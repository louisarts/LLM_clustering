#!/usr/bin/env python
"""The thesis hbars figure (raw + two partials, sorted horizontal panels), rendered once
per embedding test bed with 10-run statistics. Filenames carry the embedder tag:


for tag in instructor, instructor_gen, gemini - plus instructor_gen_glm5, the same
instructor_gen bed judged by glm-5 (all 10 runs from judge_replicates_instructor_gen_glm5.csv,
gold metrics from the instructor_gen analysis table; rendered only when the csv holds
complete runs). Pure local compute."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.stats import spearmanr, rankdata

RQ1C = Path(__file__).resolve().parents[2]
FIGS = RQ1C / 'figures'

DISPL = {'banking77': 'Banking77', 'clinc150': 'CLINC(I)', 'clinc150_domain': 'CLINC(D)',
         'mtop_intent': 'MTOP(I)', 'mtop_domain': 'MTOP(D)', 'massive_intent': 'MASSIVE(I)',
         'massive_domain': 'MASSIVE(D)', 'tweet89': 'Tweet89', 'stackoverflow20': 'StackOverflow',
         'mcid': 'M-CID', 'snips': 'SNIPS', 'dbpedia': 'DBPedia', 'reddit_s2s': 'Reddit-S2S',
         'arxiv_fine': 'ArXiv-S2S', 'stackexchange_cl': 'StackEx-S2S', 'fewrel': 'FewRel',
         'fewnerd': 'FewNerd', 'fewevent': 'FewEvent', 'bbcnews': 'BBC News'}
POSC, NEGC = '#FFB74D', '#ae282c'


def rho(g, x, y):
    return spearmanr(g[x], g[y]).statistic


def fast_perm_p(xv, yv, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    rx = rankdata(xv).astype(float); ry = rankdata(yv).astype(float)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    obs = float(np.mean(rx * ry))
    null = np.array([np.mean(rng.permutation(rx) * ry) for _ in range(n)])
    return obs, float((null >= obs).mean())


def resid_ranks(vals, ctrl):
    r_c = rankdata(ctrl); r_x = rankdata(vals).astype(float)
    return r_x - np.polyval(np.polyfit(r_c, r_x, 1), r_c)


def resid_multi(vals, C):
    """Rank residual on several controls at once (rank regression on all of them)."""
    r = rankdata(vals).astype(float)
    X = np.column_stack([np.ones(len(r))] + [rankdata(c) for c in C.T])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    return r - X @ beta


def per_run_stats(tag):
    if tag.endswith(('_glm5', '_gpt35', '_gpt4omini', '_gpt4o')):
        # alternative judge LLM on an existing bed: gold metrics from the bed's analysis
        # table, ALL judge runs (1-10) from that judge's replicates file.
        base_tag = tag.rsplit('_', 1)[0]
        A = pd.read_csv(RQ1C / 'results' / 'rq1' / f'analysis_table_{base_tag}.csv') \
            .dropna(subset=['ami'])
        rep = pd.read_csv(RQ1C / 'results' / 'rq1' / f'judge_replicates_{tag}.csv')
        J = {}
    else:
        suf = '' if tag == 'instructor' else f'_{tag}'
        A = pd.read_csv(RQ1C / 'results' / 'rq1' / f'analysis_table{suf}.csv') \
            .dropna(subset=['judge', 'ami'])
        rep = pd.read_csv(RQ1C / 'results' / 'rq1' / f'judge_replicates{suf}.csv')
        J = {1: A[['bench', 'partition_id', 'judge']]}
    # a run enters if every benchmark was EXECUTED (rows exist, complete run);
    # a reference that failed the 50-doc scoring minimum (NaN judge) drops only
    # its own benchmark, not the run's other 18.
    for r in sorted(rep.run.unique()):
        jr = rep[rep.run == r]
        if jr.bench.nunique() == A.bench.nunique():
            J[int(r)] = jr.dropna(subset=['judge'])[['bench', 'partition_id', 'judge']]
    base = A[['bench', 'partition_id', 'ami', 'silhouette', 'calinski']]
    rows = []
    for r, jr in J.items():
        M = base.merge(jr, on=['bench', 'partition_id']) \
                .dropna(subset=['judge', 'ami', 'silhouette', 'calinski'])
        for bench, g in M.groupby('bench'):
            _, p_raw = fast_perm_p(g.judge.to_numpy(), g.ami.to_numpy(), seed=r)
            row = {'run': r, 'bench': bench, 'raw': rho(g, 'judge', 'ami'), 'p_raw': p_raw}
            for c in ('silhouette', 'calinski'):
                rj = resid_ranks(g.judge.to_numpy(), g[c].to_numpy())
                ra = resid_ranks(g.ami.to_numpy(), g[c].to_numpy())
                row[c] = float(np.corrcoef(rankdata(rj), rankdata(ra))[0, 1])
                _, row[f'p_{c}'] = fast_perm_p(rj, ra, seed=r)
            C = g[['silhouette', 'calinski']].to_numpy(float)
            rj = resid_multi(g.judge.to_numpy(), C)
            ra = resid_multi(g.ami.to_numpy(), C)
            row['both'] = float(np.corrcoef(rankdata(rj), rankdata(ra))[0, 1])
            _, row['p_both'] = fast_perm_p(rj, ra, seed=r)
            rows.append(row)
    return pd.DataFrame(rows), len(J)


def agg(S, metric):
    pcol = 'p_raw' if metric == 'raw' else f'p_{metric}'
    g = S.groupby('bench')
    return pd.DataFrame({
        'mean': g[metric].mean(), 'lo': g[metric].min(), 'hi': g[metric].max(),
        'sig': g[pcol].apply(lambda p: (p < 0.05).sum() >= int(np.ceil(len(p) / 2)))})


def pad_for_labels(xmin):
    """Nudge xmin left so no band label sits bisected by the y axis."""
    for xc in (0.0, -0.24, -0.53, -0.785, -1.03):
        if xc > xmin + 0.05 and xmin > xc - 0.15:
            xmin = xc - 0.15
    return xmin


def hbars(ax, a, order, xlabel, title, xmin, mean_val=None, label_ticks=True,
          bands=True, pos_color=None, fs=1.0, band_head=1.15):
    pc = pos_color or POSC
    a = a.loc[order]
    n = len(a)
    y0 = -2.30                       # extra air under the median label
    y1 = n + band_head if bands else n + 0.35   # headroom for the band-label row
    # keep the label row clear of lines (no label row when bands are off)
    line_top = (n - 0.42 - y0) / (y1 - y0) if bands else 1.0
    for i, (b, row) in enumerate(a.iterrows()):
        v, lo, hi, sig = row['mean'], row['lo'], row['hi'], bool(row['sig'])
        base = NEGC if v < 0 else pc
        if sig:
            face = mcolors.to_rgba(base, 0.15 + 0.85 * min(abs(v), 1.0) ** 2.5)
            ax.barh(i, v, .6, color=face, edgecolor='black', lw=.7, zorder=2)
        else:
            ax.barh(i, v, .6, color='white', edgecolor=base, lw=1.1, hatch='////', zorder=2)
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt='none', ecolor='0.5',
                    elinewidth=.7, capsize=1.8, zorder=3)
    ax.axvline(0, color='black', lw=.8, ymax=line_top)
    if mean_val is not None:
        # median over the 19 per-benchmark means, distinct from the grey cutoff dots
        ax.axvline(mean_val, color='black', lw=1.2, linestyle=(0, (4, 2.2)),
                   ymax=line_top, zorder=4)
        ax.text(mean_val, -1.22, f'median {mean_val:+.2f}', ha='center', va='center',
                fontsize=8.2 * fs, fontweight='bold', color='black', zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', alpha=1.0,
                          boxstyle='round,pad=0.15'))
    for t in (-0.89, -0.68, -0.38, -0.10, 0.10, 0.38, 0.68, 0.89):
        if t > xmin + 0.02:
            ax.axvline(t, color='0.55', lw=.6, linestyle=(0, (1.5, 2.5)),
                       zorder=1, ymax=line_top)
    ax.set_ylim(y0, y1)
    if bands:
        for xc, lab in ((0.0, 'negl.'),
                        (0.24, 'weak\npos.'), (0.53, 'mod.\npos.'),
                        (0.785, 'strong\npos.'), (1.03, 'very strong\npos.'),
                        (-0.24, 'weak\nneg.'), (-0.53, 'mod.\nneg.'),
                        (-0.785, 'strong\nneg.'), (-1.03, 'very strong\nneg.')):
            if xc > xmin + 0.05:
                ax.text(xc, n + 0.12 + max(0.0, (band_head - 1.15) * 0.5), lab,
                        ha='center', va='center',
                        fontsize=6.4 * fs, color='0.35', style='italic')
    ax.set_yticks(range(n))
    ax.set_yticklabels([DISPL.get(b, b) for b in order] if label_ticks else [],
                       fontsize=8 * fs)
    ax.set_xlim(xmin, 1.17)
    ax.set_xlabel(xlabel, fontsize=9 * fs)
    ax.set_title(title, fontsize=10.5 * fs)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def render_judge_page(tag, name, outstem, xmin, pre, order, gxlim):
    """One compact three-panel row for one judge (raw, joint control, and the
    gap between them), sized so three stack on an A4 thesis page. Shared xmin,
    gap domain and benchmark order keep the five judges directly comparable."""
    S, n_runs = pre
    Ra, Ba = agg(S, 'raw'), agg(S, 'both')
    gp = S.assign(gap=S['raw'] - S['both']).groupby('bench')['gap']
    Ga = pd.DataFrame({'mean': gp.mean(), 'lo': gp.min(), 'hi': gp.max()})
    FS = 1.0
    fig, axs = plt.subplots(1, 3, figsize=(9.9, 4.15),
                            gridspec_kw={'width_ratios': [1, 1, 0.8]})
    hbars(axs[0], Ra, order, 'Spearman ρ (judge vs. AMI)',
          'Judge–AMI correlation\nper benchmark', xmin,
          mean_val=Ra['mean'].median(), fs=FS)
    hbars(axs[1], Ba, order, 'partial ρ (judge vs. AMI)',
          'Controlling for silhouette\n+ Caliński–Harabasz jointly', xmin,
          mean_val=Ba['mean'].median(), label_ticks=False, fs=FS,
          pos_color='#ABD9E9')
    ax = axs[2]
    G = Ga.loc[order]
    n = len(G)
    for i, (b, row) in enumerate(G.iterrows()):
        v, lo, hi = row['mean'], row['lo'], row['hi']
        ax.barh(i, v, .6, color='#E57373', alpha=0.9, edgecolor='black',
                lw=.6, zorder=2)
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt='none', ecolor='0.5',
                    elinewidth=.7, capsize=1.8, zorder=3)
        off = (hi - v) + .03 if v >= 0 else -((v - lo) + .03)
        ax.text(v + off, i, f'{v:+.2f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=7 * FS,
                color=('#2ca02c' if v >= 0 else '#d62728'),
                zorder=5, bbox=dict(facecolor='white', edgecolor='none',
                                    alpha=1.0, boxstyle='round,pad=0.12'))
    ax.axvline(0, color='black', lw=.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([])
    ax.set_ylim(-2.30, n + 1.15)
    ax.set_xlim(gxlim)
    ax.set_xlabel('Δ Spearman ρ', fontsize=9 * FS)
    ax.set_title('Gap between the two\n(raw − controlled)', fontsize=10.5 * FS)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    axs[0].annotate(name, xy=(-0.30, 0.5), xycoords='axes fraction',
                    rotation=90, ha='center', va='center', fontsize=11,
                    fontweight='bold', family='monospace', annotation_clip=False)
    plt.tight_layout()
    fig.savefig(FIGS / f'{outstem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outstem}.pdf', flush=True)


if __name__ == '__main__':
    ALT = ['instructor_gen', 'instructor_gen_glm5', 'instructor_gen_gpt4o',
           'instructor_gen_gpt4omini', 'instructor_gen_gpt35']
    SHORT = ['flash', 'glm5', 'gpt4o', 'gpt4omini', 'gpt35']
    NAMES = ['gemini-2.5-flash', 'glm-5', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
    pre = {t: per_run_stats(t) for t in ALT}
    gm = min(min(agg(pre[t][0], m)['lo'].min() for m in ('raw', 'both'))
             for t in ALT)
    xmin_shared = pad_for_labels(min(-0.15, gm - 0.08))
    shared = pd.concat([agg(pre[t][0], 'raw')['mean'] for t in ALT],
                       axis=1).mean(axis=1)
    order_shared = shared.sort_values().index.tolist()
    glo = min((pre[t][0]['raw'] - pre[t][0]['both']).min() for t in ALT)
    ghi = max((pre[t][0]['raw'] - pre[t][0]['both']).max() for t in ALT)
    gxlim = (glo - 0.22, ghi + 0.22)
    for t, sh, nm in zip(ALT, SHORT, NAMES):
        render_judge_page(t, nm, f'rq1_hbars_judge_{sh}', xmin_shared,
                          pre[t], order_shared, gxlim)
