#!/usr/bin/env python
"""Pre-flight check. Verifies every input a run needs before spending anything.

    python scripts/check_ready.py

Reports, per (setting, benchmark): whether the cached generation artifacts and embeddings
exist, and whether the models that setting requires are reachable through the gateway. Costs
one tiny call per distinct model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from critclust import config                      # noqa: E402


def artifact_report(setting):
    gen = config.GLM5_GEN if setting == 'unmatched' else config.MATCHED_GEN
    print(f'\n== {setting}: generation artifacts in {gen.name}/')
    missing = []
    for bench in config.benchmarks_for(setting):
        have = {
            'reference': (gen / f'{bench}_reference.csv').exists(),
            'rewrites': (gen / f'{bench}_rewrites.csv').exists(),
            'exemplars': (gen / f'{bench}_exemplars.json').exists(),
        }
        if setting == 'unmatched':
            have['raw emb'] = (config.GEMINI_EMB / f'{bench}_raw.npy').exists()
            have['rew emb'] = (config.GLM5_EMB / f'{bench}_rew.npy').exists()
        else:
            embedder = config.MATCHED_CFG[bench][0]
            if embedder == 'e5-large-v2':
                have['raw emb'] = True          # computed locally on demand
                have['rew emb'] = True
            else:
                have['raw emb'] = (config.MATCHED_EMB / f'A_{bench}_raw.npy').exists()
                have['rew emb'] = (config.MATCHED_EMB / f'A_{bench}_rew.npy').exists()
        gaps = [k for k, v in have.items() if not v]
        flag = 'ok' if not gaps else 'MISSING ' + ', '.join(gaps)
        print(f'  {bench:18} {flag}')
        if gaps:
            missing.append((bench, gaps))
    return missing


def model_report():
    from critclust.llm import CLIENT
    needed = {config.UNMATCHED_LLM}
    embedders = {config.UNMATCHED_EMBEDDER}
    for embedder, llm, _ in config.MATCHED_CFG.values():
        needed.add(llm)
        embedders.add(embedder)

    print('\n== chat models')
    blocked = []
    for m in sorted(needed):
        try:
            CLIENT.chat.completions.create(model=m, max_tokens=3,
                                           messages=[{'role': 'user', 'content': 'ok'}])
            print(f'  {m:34} ok')
        except Exception as e:
            print(f'  {m:34} FAIL {str(e)[:80]}')
            blocked.append(m)

    print('\n== embedders')
    local = {'instructor-large', 'e5-large-v2', 'paraphrase-mpnet'}
    for m in sorted(embedders):
        if m in local:
            print(f'  {m:34} local (CPU)')
            continue
        try:
            CLIENT.embeddings.create(model=m, input=['hello'])
            print(f'  {m:34} ok')
        except Exception as e:
            print(f'  {m:34} FAIL {str(e)[:80]}')
            blocked.append(m)
    return blocked


def main():
    config.load_env()
    from critclust.data import gateway_spend
    spend = gateway_spend()
    print(f'gateway spend: {"unknown" if spend is None else f"${spend:.2f}"} '
          f'(stop at ${config.BUDGET_STOP:.2f})')

    missing = artifact_report('unmatched') + artifact_report('matched')
    blocked = model_report()

    print('\n== summary')
    print(f'  benchmarks with missing artifacts: {len(missing)}')
    print(f'  unreachable models: {len(blocked)}')
    if blocked:
        affected = [b for b in config.BENCH8
                    if config.MATCHED_CFG[b][1] in blocked
                    or config.MATCHED_CFG[b][0] in blocked]
        print(f'  matched benchmarks blocked by those models: {affected}')


if __name__ == '__main__':
    main()
