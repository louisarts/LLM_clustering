"""Figures and summary tables. Imported by the analysis notebook; nothing here calls an LLM.

Figures are written to figures/ at true print size (6.3 in = A4 textwidth at 2.5 cm margins),
serif, title-free vector PDF - the title is added only for the on-screen copy so the caption
can live in LaTeX.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config

_RC = {'font.family': 'serif', 'mathtext.fontset': 'stix',
       'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
       'axes.linewidth': 0.8}

_OURS = '#3B6EA5'
_SOTA = '0.78'
_UP = '#1A7F37'      # ahead of the best published value
_DOWN = '#C0392B'    # behind it


def published_best():
    """benchmark display name -> best published value, per metric."""
    df = pd.read_csv(config.PUBLISHED)
    return {m: df[df.metric == m].groupby('benchmark').value.max().to_dict()
            for m in ('NMI', 'ACC')}


_BEST = None


def sota_for(bench, metric):
    global _BEST
    if _BEST is None:
        _BEST = published_best()
    return _BEST[metric].get(config.BENCH_DISPLAY.get(bench, bench), np.nan)


def barplot(summary, metric, title, outfile, show=True):
    """Grouped bars per benchmark: best published value against this method's mean +- std.

    Each pair is annotated with the signed difference (green when this method is ahead, red
    when behind), and the count of benchmarks beaten goes in the title.
    """
    key = metric.lower()
    d = summary.copy()
    d['display'] = d.bench.map(lambda b: config.BENCH_DISPLAY.get(b, b))
    d['sota'] = d.bench.map(lambda b: sota_for(b, metric))
    d['delta'] = d[f'{key}_mean'] - d['sota']
    d = d.sort_values('display').reset_index(drop=True)

    beaten = int((d['delta'] > 0).sum())
    total = int(d['sota'].notna().sum())
    full_title = f'{title} | {metric}: beats best published on {beaten}/{total}'

    x = np.arange(len(d))
    width = 0.38
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.3, 3.2))
        ax.bar(x - width / 2, d['sota'], width, color=_SOTA, edgecolor='black',
               linewidth=0.5, label='best published')
        ax.bar(x + width / 2, d[f'{key}_mean'], width, yerr=d[f'{key}_std'], capsize=2,
               color=_OURS, edgecolor='black', linewidth=0.5,
               error_kw={'elinewidth': 0.7}, label='this method')

        # signed difference above each pair
        for i, r in d.iterrows():
            if np.isnan(r['delta']):
                continue
            top = max(r['sota'], r[f'{key}_mean'] + (r[f'{key}_std'] or 0))
            ax.text(i, top + 1.5, f"{r['delta']:+.1f}", ha='center', va='bottom',
                    fontsize=5.6, fontweight='bold',
                    color=_UP if r['delta'] > 0 else _DOWN)

        ax.set_xticks(x)
        ax.set_xticklabels(d['display'], rotation=45, ha='right', fontsize=6.5)
        ax.set_ylabel(metric, fontsize=8.5)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc='upper left', ncol=2,
                  bbox_to_anchor=(0.0, 1.02))
        # PDF first, title-free: the caption lives in LaTeX
        fig.savefig(config.FIGURES / outfile, bbox_inches='tight')
        # PNG second, with the title baked in - only when a PNG dir is configured
        if config.PLOTS_PNG is not None:
            ax.set_title(full_title, fontsize=9, pad=10)
            png = config.PLOTS_PNG / (Path(outfile).stem + '.png')
            fig.savefig(png, bbox_inches='tight', dpi=200)
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)

    print(f'{full_title}   (mean delta {d["delta"].mean():+.2f})')
    print(f'  saved {config.FIGURES.name}/{outfile} and analysis/plots/{Path(outfile).stem}.png')
    return d


def baseline_comparison(algo, setting, metric, outfile, show=True):
    """CritClust_<algo> against plain k-means on the IDENTICAL embeddings.

    This is the comparison that separates the algorithm's contribution from the encoder's.
    The published-SOTA plots are confounded, because gemini-embedding-001 is stronger than any
    competitor's encoder; this one is not, since both bars share an encoder.
    """
    key = metric.lower()
    base = pd.read_csv(config.RESULTS / f'baseline_{setting}_summary.csv')
    ours = pd.read_csv(config.RESULTS / f'{config.run_name(algo, setting)}_summary.csv')

    d = base[['bench', 'k', f'{key}_mean', f'{key}_std']].rename(
        columns={f'{key}_mean': 'km', f'{key}_std': 'km_std'}).merge(
        ours[['bench', f'{key}_mean', f'{key}_std']].rename(
            columns={f'{key}_mean': 'ours', f'{key}_std': 'ours_std'}), on='bench')
    d['display'] = d.bench.map(lambda b: config.BENCH_DISPLAY.get(b, b))
    d['delta'] = d['ours'] - d['km']
    d = d.sort_values('display').reset_index(drop=True)

    up = int((d['delta'] > 0).sum())
    full_title = (f'CritClust_{algo} vs same-encoder k-means | {metric}: '
                  f'improves on {up}/{len(d)} (mean {d["delta"].mean():+.2f})')

    x = np.arange(len(d))
    width = 0.38
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.3, 3.2))
        ax.bar(x - width / 2, d['km'], width, yerr=d['km_std'], capsize=2,
               color=_SOTA, edgecolor='black', linewidth=0.5,
               error_kw={'elinewidth': 0.7}, label='k-means, same embedder')
        ax.bar(x + width / 2, d['ours'], width, yerr=d['ours_std'], capsize=2,
               color=_OURS, edgecolor='black', linewidth=0.5,
               error_kw={'elinewidth': 0.7}, label=f'CritClust_{algo}')

        for i, r in d.iterrows():
            top = max(r['km'] + (r['km_std'] or 0), r['ours'] + (r['ours_std'] or 0))
            ax.text(i, top + 1.5, f"{r['delta']:+.1f}", ha='center', va='bottom',
                    fontsize=5.6, fontweight='bold',
                    color=_UP if r['delta'] > 0 else _DOWN)

        ax.set_xticks(x)
        ax.set_xticklabels(d['display'], rotation=45, ha='right', fontsize=6.5)
        ax.set_ylabel(metric, fontsize=8.5)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc='upper left', ncol=2,
                  bbox_to_anchor=(0.0, 1.02))

        fig.savefig(config.FIGURES / outfile, bbox_inches='tight')
        ax.set_title(full_title, fontsize=9, pad=10)
        png = config.PLOTS_PNG / (Path(outfile).stem + '.png')
        fig.savefig(png, bbox_inches='tight', dpi=200)
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)

    print(full_title)
    print(f'  saved {config.FIGURES.name}/{outfile} and analysis/plots/{Path(outfile).stem}.png')
    return d


def sota_table():
    """The comparison target for every benchmark: who holds it, on what encoder and LLM.

    Values are the best published NMI and ACC from thesis/published_results.csv; the method
    named is the holder of the NMI record (on four benchmarks the ACC record is held by a
    different method, which is why the two columns are not always the same paper's numbers).
    """
    cfg = config.sota_config()
    rows = []
    for bench, c in cfg.items():
        disp = config.BENCH_DISPLAY.get(bench, bench)
        rows.append({
            'Benchmark': disp,
            'SOTA method': c['method'],
            'NMI': sota_for(bench, 'NMI'),
            'ACC': sota_for(bench, 'ACC'),
            'SOTA embedder': c['embedder'],
            'SOTA LLM': c['llm'],
            'note': c.get('note', ''),
        })
    t = pd.DataFrame(rows).sort_values('Benchmark').reset_index(drop=True)
    print(f'{len(t)} benchmarks | {t["SOTA method"].nunique()} distinct methods | '
          f'{t["SOTA embedder"].nunique()} encoders | {t["SOTA LLM"].nunique()} language models')
    return t


def candidate_table(candidates, runs, title):
    """Which candidate the judge selected, per benchmark and overall."""
    table = candidates.copy()
    table.index = [config.BENCH_DISPLAY.get(b, b) for b in table.index]
    table.columns = [config.CAND_DISPLAY.get(c, c) for c in table.columns]
    table = table.loc[:, table.sum(0) > 0]
    table['total seeds'] = table.sum(1)

    share = (runs.winner.value_counts(normalize=True) * 100).round(1)
    share.index = [config.CAND_DISPLAY.get(c, c) for c in share.index]

    print(f'\n{title}: candidate chosen, count over seeds')
    print(f'mean repair rounds kept: {runs.rounds_kept.mean():.2f}')
    return table, share.to_frame('share %')


def granularity_table(runs):
    """CritClust_B only: did "aim for k" land near k?"""
    g = (runs.groupby('bench')
             .agg(k=('k', 'first'), categories=('n_categories', 'first'))
             .reset_index())
    g['ratio'] = (g.categories / g.k).round(2)
    g['bench'] = g.bench.map(lambda b: config.BENCH_DISPLAY.get(b, b))
    print(f'median categories / k = {g.ratio.median():.2f}')
    return g.sort_values('ratio')


def compare(summary_a, summary_b, label):
    """Paired A-vs-B table within one setting."""
    m = summary_a.merge(summary_b, on='bench', suffixes=('_A', '_B'))
    m['display'] = m.bench.map(lambda b: config.BENCH_DISPLAY.get(b, b))
    m['d_nmi'] = (m.nmi_mean_B - m.nmi_mean_A).round(2)
    m['d_acc'] = (m.acc_mean_B - m.acc_mean_A).round(2)
    print(f'{label}: B beats A on NMI {int((m.d_nmi > 0).sum())}/{len(m)} '
          f'(mean {m.d_nmi.mean():+.2f}), '
          f'ACC {int((m.d_acc > 0).sum())}/{len(m)} (mean {m.d_acc.mean():+.2f})')
    cols = ['display', 'nmi_mean_A', 'nmi_mean_B', 'd_nmi',
            'acc_mean_A', 'acc_mean_B', 'd_acc']
    return m[cols].sort_values('d_nmi', ascending=False)
