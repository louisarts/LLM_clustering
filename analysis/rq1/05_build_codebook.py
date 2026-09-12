"""Step 5 - PAID but tiny (~$0.30 total): the DISCOVERY stage.

One thinking-on LLM call per benchmark: the model reads N_DISCOVERY sampled
documents and writes the codebook (category names + one-line definitions) at
the granularity it judges natural. Output is saved as readable JSON and printed
in full, because inspectability is the point: if a codebook is degenerate
("1. Technical questions, 2. Other"), you can see it before spending anything
on assignment.

N_SAMPLES > 1 switches to the map-reduce ablation (disjoint samples, one draft
codebook each, merged by a final LLM call). REJECTED for the thesis judge -
it correlated worse with humans (pooled judge~AMI +0.77 vs +0.80) - but kept
reproducible.

Usage:
  python 05_build_codebook.py            dry run: what would be built
  python 05_build_codebook.py --run      spends (~8 calls total)
  python 05_build_codebook.py --run --bench banking77   one benchmark only

Idempotent: benchmarks whose codebook file already exists are skipped.
To rebuild one, delete data/codebooks/<bench>.json first.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RQ1 = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).parent))

from judge import CRITERIA, build_codebook, merge_codebooks         # noqa: E402

OUT = RQ1 / 'data' / 'rq1' / 'codebooks'
OUT.mkdir(parents=True, exist_ok=True)

N_DISCOVERY = 300      # documents shown to the codebook writer
N_SAMPLES = 1          # 1 = the thesis judge; >1 = the rejected map-reduce ablation
SEED = 0

RUN = '--run' in sys.argv
ONLY = sys.argv[sys.argv.index('--bench') + 1] if '--bench' in sys.argv else None

benches = [ONLY] if ONLY else sorted(CRITERIA)
todo = [b for b in benches if not (OUT / f'{b}.json').exists()]
done = [b for b in benches if b not in todo]
for b in done:
    print(f'== {b}: codebook exists, skipping (delete {OUT / (b + ".json")} to rebuild)')
calls_each = 1 if N_SAMPLES == 1 else N_SAMPLES + 1
print(f'\n{len(todo)} codebooks to build ({", ".join(todo) if todo else "none"}), '
      f'{calls_each} thinking-on call(s) each (~$0.03-0.05 per call)')
if not RUN:
    print('dry run only - rerun with --run to spend.')
    sys.exit(0)

failed = []
for bench in todo:
    spec = CRITERIA[bench]
    texts = pd.read_csv(RQ1 / 'data' / 'rq1' / 'texts' / f'{bench}.csv')['text'] \
        .fillna('').astype(str).tolist()
    # one permutation cut into N_SAMPLES disjoint chunks (shrunk for small corpora)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(texts))
    chunk = min(N_DISCOVERY, len(texts) // N_SAMPLES)
    samples = [[texts[i] for i in perm[j * chunk:(j + 1) * chunk]]
               for j in range(N_SAMPLES)]
    what = (f'{chunk} docs' if N_SAMPLES == 1
            else f'{N_SAMPLES} disjoint samples of {chunk} docs')
    print(f'\n== {bench}: {what} | criterion: {spec["criterion"]}', flush=True)
    (OUT / 'raw').mkdir(exist_ok=True)
    try:
        if N_SAMPLES == 1:
            codebook = build_codebook(samples[0], spec,
                                      raw_out=OUT / 'raw' / f'{bench}.txt')
            drafts = None
        else:
            drafts = []
            for j, sample in enumerate(samples, 1):
                draft = build_codebook(sample, spec,
                                       raw_out=OUT / 'raw' / f'{bench}_draft{j}.txt')
                print(f'   draft {j}: {len(draft)} categories', flush=True)
                drafts.append(draft)
            codebook = merge_codebooks(drafts, spec,
                                       raw_out=OUT / 'raw' / f'{bench}_merge.txt')
    except Exception as e:
        failed.append(bench)
        print(f'   !! FAILED: {type(e).__name__}: {e}')
        print(f'   raw replies saved in {OUT / "raw"} - inspect them, then rerun '
              f'(only failed benchmarks are retried)')
        continue
    payload = {'bench': bench, 'criterion': spec['criterion'],
               'n_discovery_docs': chunk * N_SAMPLES, 'n_samples': N_SAMPLES,
               'discovery_seed': SEED, 'categories': codebook}
    if drafts is not None:
        payload['draft_codebooks'] = drafts
    (OUT / f'{bench}.json').write_text(json.dumps(payload, indent=2))
    if drafts is not None:
        print(f'   merged: {" + ".join(str(len(d)) for d in drafts)} -> '
              f'{len(codebook)} categories -> {OUT / (bench + ".json")}')
    else:
        print(f'   {len(codebook)} categories -> {OUT / (bench + ".json")}')
    for c in codebook:
        print(f'     {c["id"]:>3}. {c["name"]} - {c["definition"]}')

print('\nINSPECT THE CODEBOOKS ABOVE before running 06 - a degenerate codebook '
      '(a couple of giant categories, or obvious nonsense) means the judge cannot '
      'work on that corpus, and you can see it now for free.')
if failed:
    print(f'\n!! failed and needing a rerun: {", ".join(failed)} '
          f'(raw replies in {OUT}/raw/)')
    sys.exit(1)
print('next: 06_assign_documents.py')
