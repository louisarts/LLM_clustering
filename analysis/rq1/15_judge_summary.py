#!/usr/bin/env python
"""RQ1 main-text summary: the judge-LLM comparison on the generic-INSTRUCTOR bed.

One strip figure (each row a judge LLM, each dot one benchmark's mean raw
judge~AMI correlation over its runs, red diamond the grand mean) and one
booktabs scoreboard table. The per-bed hbars figures live in the appendix.

    figures/rq1_judgebeds_summary.pdf   analysis/plots/rq1_judgebeds_summary.png
    results/table_rq1_judgebeds.tex

Beds are included as their replicate CSVs exist; incomplete runs are excluded by
the shared per_run_stats rule. Pure local compute."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RQ1C = HERE.parents[1]          # repo root
spec = importlib.util.spec_from_file_location('hbars', HERE / '10_hbars_all_beds.py')
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

def joint_partial_stats(tag):
    """Per-bench mean and per-run medians of the judge~AMI partial controlling for
    silhouette AND Calinski-Harabasz jointly (rank regression on both)."""
    from scipy.stats import spearmanr, rankdata
    A = pd.read_csv(RQ1C / 'results' / 'rq1' / 'analysis_table_instructor_gen.csv')
    base = A[['bench', 'partition_id', 'ami', 'silhouette', 'calinski']].dropna()
    if tag == 'instructor_gen':
        rep = pd.read_csv(RQ1C / 'results' / 'rq1' / 'judge_replicates_instructor_gen.csv')
        J = {1: A.dropna(subset=['judge'])[['bench', 'partition_id', 'judge']]}
    else:
        rep = pd.read_csv(RQ1C / 'results' / 'rq1' / f'judge_replicates_{tag}.csv')
        J = {}
    for r in sorted(rep.run.unique()):
        jr = rep[rep.run == r]
        if jr.bench.nunique() == 19:
            J[int(r)] = jr.dropna(subset=['judge'])[['bench', 'partition_id', 'judge']]

    def resid(vals, C):
        rk = rankdata(vals).astype(float)
        X = np.column_stack([np.ones(len(rk))] + [rankdata(c) for c in C.T])
        beta, *_ = np.linalg.lstsq(X, rk, rcond=None)
        return rk - X @ beta

    rows_ = []
    for r, jr in J.items():
        M = base.merge(jr, on=['bench', 'partition_id']).dropna()
        for b, g in M.groupby('bench'):
            C = g[['silhouette', 'calinski']].to_numpy(float)
            rows_.append({'run': r, 'bench': b,
                          'rho': spearmanr(resid(g.judge.to_numpy(), C),
                                           resid(g.ami.to_numpy(), C)).statistic})
    D = pd.DataFrame(rows_)
    bench_means = D.groupby('bench').rho.mean()
    return bench_means, bench_means.median(), D.groupby('run').rho.median().std()


BEDS = [('instructor_gen', 'gemini-2.5-flash'),
        ('instructor_gen_glm5', 'glm-5'),
        ('instructor_gen_gpt4o', 'gpt-4o'),
        ('instructor_gen_gpt4omini', 'gpt-4o-mini'),
        ('instructor_gen_gpt35', 'gpt-3.5-turbo')]
MEANC = '#ae282c'

rows = []
for tag, name in BEDS:
    f = (RQ1C / 'results' / 'rq1' / ('analysis_table_instructor_gen.csv' if tag == 'instructor_gen'
                             else f'judge_replicates_{tag}.csv'))
    if not f.exists():
        continue
    S, n_runs = hb.per_run_stats(tag)
    Ra, Ca, Sa = hb.agg(S, 'raw'), hb.agg(S, 'calinski'), hb.agg(S, 'silhouette')
    # run-to-run spread: the statistic recomputed within each replicate run
    per_run = S.groupby('run').agg(med=('raw', 'median'), pch=('calinski', 'median'),
                                   psil=('silhouette', 'median'))
    rows.append({'name': name, 'bench_means': Ra['mean'], 'sig': int(Ra['sig'].sum()),
                 'mean': S['raw'].mean(), 'median': Ra['mean'].median(),
                 'p_ch': Ca['mean'].median(), 'p_sil': Sa['mean'].median(),
                 'median_sd': per_run['med'].std(), 'p_ch_sd': per_run['pch'].std(),
                 'p_sil_sd': per_run['psil'].std(), 'n_runs': n_runs})
    rows[-1]['both_bench'], rows[-1]['p_both'], rows[-1]['p_both_sd'] = joint_partial_stats(tag)

_RCA = {'font.family': 'serif', 'mathtext.fontset': 'stix',
        'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 0.8, 'font.size': 9}
with plt.rc_context(_RCA):
    # two bars per judge: raw judge~AMI median and the joint geometry-controlled
    # median, each with its own 19 per-benchmark points overlaid.
    from matplotlib.patches import Patch
    disp = sorted(rows, key=lambda r: r['median'])
    n = len(disp)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for t in (0.10, 0.38, 0.68, 0.89):
        ax.axhline(t, color='0.62', lw=.6, linestyle=(0, (1.5, 2.5)), zorder=1)
    for yc, lab in ((0.24, 'weak'), (0.53, 'moderate'),
                    (0.785, 'strong'), (0.945, 'very strong')):
        ax.annotate(lab, xy=(1.012, yc), xycoords=('axes fraction', 'data'),
                    va='center', ha='left', fontsize=6.6, color='0.4',
                    style='italic', annotation_clip=False)
    rng = np.random.RandomState(0)
    W, OFF = 0.34, 0.19
    for i, r in enumerate(disp):
        for dx, med, bm, face in ((-OFF, r['median'], r['bench_means'], '#FFB74D'),
                                  (+OFF, r['p_both'], r['both_bench'], '#ABD9E9')):
            ax.bar(i + dx, med, W, color=face, alpha=0.5, edgecolor='black',
                   lw=.8, zorder=2)
            jit = rng.uniform(-0.10, 0.10, len(bm))
            ax.scatter(i + dx + jit, bm, s=13, color=face, edgecolor='black',
                       lw=.35, zorder=3)
            low = bm < 0.38            # name the dots that fall in the weak band
            for x_, y_, b_ in zip((i + dx + jit)[low.values], bm[low], bm.index[low]):
                ax.annotate(hb.DISPL.get(b_, b_), (x_, y_), xytext=(6, 0),
                            textcoords='offset points', va='center', fontsize=6,
                            color='0.4', style='italic')
            mc = '#2ca02c' if med >= 0 else '#d62728'
            ax.text(i + dx, med + 0.018, f"{med:+.2f}", ha='center', va='bottom',
                    fontsize=7.8, fontweight='bold', color=mc, zorder=6,
                    bbox=dict(facecolor='white', edgecolor=mc, lw=0.9,
                              alpha=1.0, boxstyle='round,pad=0.16'))
    handles = [Patch(facecolor='#FFB74D', alpha=0.5, edgecolor='black', lw=.8,
                     label='judge vs. AMI'),
               Patch(facecolor='#ABD9E9', alpha=0.5, edgecolor='black', lw=.8,
                     label='controlling silhouette + Caliński–Harabasz jointly')]
    ax.legend(handles=handles, frameon=False, fontsize=7.8, ncol=2,
              loc='lower center', bbox_to_anchor=(0.5, -0.26))
    ax.set_xticks(range(n))
    ax.set_xticklabels([r['name'] for r in disp], fontsize=8.6, family='monospace',
                       rotation=25, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.72, n - 0.28)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('Spearman ρ and partial ρ (judge vs. AMI),\n'
                  'per-benchmark mean over runs', fontsize=9)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='x', length=0)
    plt.tight_layout()
    fig.savefig(RQ1C / 'figures' / 'rq1_judgebeds_summary.pdf', bbox_inches='tight')
    plt.close(fig)

lines = ['\\begin{table}[htbp]', '\\centering', '\\small',
         '\\begin{tabular}{lccccc}', '\\toprule',
         'Judge LLM & Median $\\rho$ & Significant & '
         'Median $\\rho$ $|$ Calinski-Harabasz & Median $\\rho$ $|$ silhouette & '
         'Median $\\rho$ $|$ both \\\\',
         '\\midrule']
def pm(v, sd):
    return f'{v:+.2f}' if not sd == sd else f'{v:+.2f} $\\pm$ {sd:.2f}'

for r in rows:
    lines.append(f"{{\\small\\ttfamily {r['name']}}} & {pm(r['median'], r['median_sd'])} & "
                 f"{r['sig']}/19 & {pm(r['p_ch'], r['p_ch_sd'])} & "
                 f"{pm(r['p_sil'], r['p_sil_sd'])} & "
                 f"{pm(r['p_both'], r['p_both_sd'])} \\\\")
lines += ['\\bottomrule', '\\end{tabular}',
          '\\caption{The codebook judge across judge LLMs on the generic-INSTRUCTOR '
          'bed. The median is over the 19 per-benchmark mean '
          'correlations (10 independent judge runs each); Significant counts '
          'benchmarks where the correlation reaches $p < 0.05$ in a majority of '
          'runs; the last three columns are the median partial correlations '
          'controlling for the Calinski-Harabasz index and the silhouette score '
          'singly, and for both jointly. '
          'Each $\\pm$ is the standard deviation of that statistic recomputed '
          'within each replicate run. '
          'Per-benchmark detail in the appendix.}',
          '\\label{tab:rq1-judgebeds}', '\\end{table}']
(RQ1C / 'results' / 'rq1' / 'table_rq1_judgebeds.tex').write_text('\n'.join(lines))

for r in rows:
    print(f"{r['name']:18s} mean {r['mean']:+.3f} | median {r['median']:+.3f} | "
          f"sig {r['sig']}/19 | pCH {r['p_ch']:+.2f} | psil {r['p_sil']:+.2f} | "
          f"{r['n_runs']} runs")
print('saved rq1_judgebeds_summary.pdf and table_rq1_judgebeds.tex')
