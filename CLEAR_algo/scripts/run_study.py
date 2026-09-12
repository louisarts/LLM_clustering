#!/usr/bin/env python
"""Run one (algorithm, setting) combination and write its results.

    python scripts/run_study.py A unmatched        # CritClust_A, 19 benchmarks
    python scripts/run_study.py B unmatched        # CritClust_B, 19 benchmarks
    python scripts/run_study.py A matched          # CritClust_A, 8 benchmarks
    python scripts/run_study.py B matched          # CritClust_B, 8 benchmarks

Options:
    --benches a,b,c    restrict to these benchmarks (default: all for the setting)
    --seeds 0,1,2      restrict to these seeds (default: 0-4)
    --dry-run          print the plan and the model availability, run nothing

Every stage is cached on disk, so an interrupted run resumes where it stopped. All LLM calls
are budget-guarded at config.BUDGET_STOP.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from critclust import config                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('algo', choices=['A', 'B', 'C'])
    ap.add_argument('setting', choices=['unmatched', 'matched', 'embmatched', 'llmmatched'])
    ap.add_argument('--benches', default=None, help='comma-separated subset')
    ap.add_argument('--seeds', default=None, help='comma-separated subset')
    ap.add_argument('--aligned', action='store_true',
                    help='RQ1-aligned judge protocol (writes *_aligned stems)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    benches = args.benches.split(',') if args.benches else config.benchmarks_for(args.setting)
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else config.SEEDS

    print(f'CritClust_{args.algo} / {args.setting}')
    print(f'  {len(benches)} benchmarks, {len(seeds)} seeds')
    for b in benches:
        embedder, llm = config.models_for(b, args.setting)
        print(f'    {b:18} {embedder:24} {llm}')

    if args.dry_run:
        from critclust.data import gateway_spend
        spend = gateway_spend()
        print(f'\n  gateway spend: {"unknown" if spend is None else f"${spend:.2f}"} '
              f'of ${config.BUDGET_STOP:.2f} stop')
        print('  dry run, nothing executed')
        return

    from critclust import runners
    t0 = time.time()
    runner = {'A': runners.run_A, 'B': runners.run_B, 'C': runners.run_C}[args.algo]
    df = runner(args.setting, benches=benches, seeds=seeds, aligned=args.aligned)
    if df.empty:
        print('no rows produced')
        return
    summary, candidates = runners.save(df, args.algo, args.setting, aligned=args.aligned)
    print(summary.to_string(index=False))
    print(f'\nelapsed {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
