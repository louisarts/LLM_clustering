"""LLM-generated artifacts: the reference labelling, criterion rewrites and exemplars.

Rewrites and exemplars are read from the original study's caches - they are identical for both
algorithms and cost one LLM call per document to regenerate. The reference labelling is built
here, per (algorithm, benchmark, setting), and cached under artifacts/codebooks/.

Both algorithms label exactly the same document ids; only the codebook they are labelled
against differs.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import config
from .codebooks import codebook_A, codebook_B_aim_k
from .data import budget_guard
from .llm import classify_batch, _clip


def _gen_dir(setting):
    return config.GLM5_GEN if setting == 'unmatched' else config.MATCHED_GEN


def reference_doc_ids(bench, setting):
    """The ~800 documents that get reference-labelled, as sampled by the original study.

    The sample is a property of the benchmark, not of the setting, so where a setting has no
    file of its own the unmatched sample is reused. Both algorithms and both settings then
    label the same documents, and only the codebook and the labelling model differ.
    """
    for src in (_gen_dir(setting) / f'{bench}_reference.csv',
                config.GLM5_GEN / f'{bench}_reference.csv'):
        if src.exists():
            d = pd.read_csv(src)
            return sorted(int(r.doc_idx) for r in d.itertuples()
                          if r.category and r.category > 0)
    raise FileNotFoundError(f'{bench}: no reference sample in {setting} or unmatched')


def build_reference(bench, setting, algo, docs, criterion, k, ask):
    """-> ({doc_idx: category}, codebook). Cached; a second call costs nothing.

    algo 'A' labels against V0's single-shot codebook, algo 'B' against the aim-for-k
    codebook. Documents with category 0 ("fits none") are dropped from the returned mapping
    but kept in the CSV.
    """
    tag = f'{algo}_{bench}_{setting}'
    cb_file = config.CODEBOOKS / f'{tag}_codebook.json'
    ref_file = config.CODEBOOKS / f'{tag}_reference.csv'

    method_file = config.CODEBOOKS / f'{tag}_method.txt'
    if ref_file.exists() and cb_file.exists():
        d = pd.read_csv(ref_file)
        fm = {int(r.doc_idx): int(r.category) for r in d.itertuples()
              if r.category and r.category > 0}
        return fm, json.loads(cb_file.read_text())

    budget_guard()
    if cb_file.exists():
        codebook = json.loads(cb_file.read_text())
    else:
        if algo == 'A':
            codebook, method = codebook_A(docs, criterion, ask=ask)
            if method != 'single_shot':
                print(f'  !! {tag}: codebook built by {method}, not V0 single-shot')
        else:
            codebook, capped = codebook_B_aim_k(docs, criterion, k, ask)
            # `capped` distinguishes two outcomes, and the difference is a result rather than a
            # warning: 'capped_at_k' means the model kept proposing categories and the ceiling
            # cut it off, 'converged_below_k' means it stopped on its own before reaching k.
            method = 'capped_at_k' if capped else 'converged_below_k'
            print(f'  {tag}: codebook {method.replace("_", " ")} '
                  f'({len(codebook)} categories, k={k})')
        cb_file.write_text(json.dumps(codebook, indent=1))
        (config.CODEBOOKS / f'{tag}_method.txt').write_text(method)

    ids = reference_doc_ids(bench, setting)
    labels, n_calls = classify_batch(ids, docs, codebook, {'criterion': criterion}, ask,
                                     label=f'{tag} reference')
    pd.DataFrame([{'doc_idx': i, 'category': c} for i, c in sorted(labels.items())]) \
      .to_csv(ref_file, index=False)

    fm = {i: c for i, c in labels.items() if c > 0}
    print(f'  {tag}: {len(fm)}/{len(ids)} usable, {len(codebook)} categories '
          f'(k={k}), {n_calls} calls')
    return fm, codebook


def judge_reference(bench, llm_src, algo, docs, criterion, k, ask_judge, seed):
    """The RQ1-ALIGNED judge instrument: -> ({doc_idx: category}, codebook, cb_method).

    Codebook: unmatched-source reuses the cached glm-5 single-shot A codebook (already an
    RQ1 build_codebook artifact); matched-source builds ONE fresh Flash single-shot codebook
    per benchmark via RQ1's own code path (cached). Reference: a FRESH 1,000-document sample
    PER SEED (RQ1's assignment size; seeds are independent judge labellings), classified in
    batches with the judge LLM. Everything cached; documents with category 0 are dropped.
    """
    if llm_src == 'unmatched':
        cb_file = config.CODEBOOKS / f'{algo}_{bench}_unmatched_codebook.json'
        codebook = json.loads(cb_file.read_text())
        cb_method = 'glm5_single_shot_shared'
    else:
        cb_file = config.CODEBOOKS / f'{algo}_{bench}_matched_flashcb.json'
        if cb_file.exists():
            codebook = json.loads(cb_file.read_text())
        else:
            budget_guard()
            from .codebooks import codebook_A
            codebook, method = codebook_A(docs, criterion, ask=ask_judge)
            cb_file.write_text(json.dumps(codebook, indent=1))
            (config.CODEBOOKS / f'{algo}_{bench}_matched_flashcb_method.txt').write_text(method)
            print(f'  {algo}_{bench}: Flash judge codebook built '
                  f'({len(codebook)} categories, {method})')
        cb_method = 'flash_single_shot'

    ref_file = config.CODEBOOKS / f'{algo}_{bench}_{llm_src}_s{seed}_ref1000.csv'
    if ref_file.exists():
        d = pd.read_csv(ref_file)
        fm = {int(r.doc_idx): int(r.category) for r in d.itertuples()
              if r.category and r.category > 0}
        return fm, codebook, cb_method

    budget_guard()
    import numpy as np
    rng = np.random.RandomState(9000 + seed)
    ids = sorted(int(i) for i in rng.choice(len(docs),
                                            min(config.ALIGNED_REF_N, len(docs)),
                                            replace=False))
    labels, n_calls = classify_batch(ids, docs, codebook, {'criterion': criterion},
                                     ask_judge, label=f'{algo}_{bench}_{llm_src}_s{seed} ref')
    pd.DataFrame([{'doc_idx': i, 'category': c} for i, c in sorted(labels.items())]) \
      .to_csv(ref_file, index=False)
    fm = {i: c for i, c in labels.items() if c > 0}
    print(f'  {algo}_{bench}_{llm_src} s{seed}: aligned reference {len(fm)}/{len(ids)} '
          f'usable ({len(codebook)} categories, {n_calls} calls)')
    return fm, codebook, cb_method


def rewrites(bench, setting, docs, criterion=None, ask=None):
    """Each document rewritten as a <=15-word phrase expressing only the criterion.

    Reuses the original study's cache when it exists; otherwise generates and caches into
    artifacts/. Generation is batched - one call per REWRITE_BATCH documents rather than one
    per document, which is the difference between ~30k calls and ~1.5k across the suite.
    """
    src = _gen_dir(setting) / f'{bench}_rewrites.csv'
    if src.exists():
        rw = pd.read_csv(src).drop_duplicates('doc_idx').set_index('doc_idx').rewrite
        return [str(rw.get(i, docs[i])) for i in range(len(docs))]

    own = config.ARTIFACTS / 'generation' / f'{bench}_{setting}_rewrites.csv'
    own.parent.mkdir(parents=True, exist_ok=True)
    if own.exists():
        rw = pd.read_csv(own).drop_duplicates('doc_idx').set_index('doc_idx').rewrite
        return [str(rw.get(i, docs[i])) for i in range(len(docs))]

    if ask is None or criterion is None:
        raise RuntimeError(f'{bench}/{setting}: no cached rewrites and nothing to generate with')
    budget_guard()
    out = generate_rewrites(docs, criterion, ask, label=f'{bench}/{setting}')
    pd.DataFrame([{'doc_idx': i, 'rewrite': out.get(i, docs[i])} for i in range(len(docs))]) \
      .to_csv(own, index=False)
    print(f'  {bench}/{setting}: generated {len(out)}/{len(docs)} rewrites')
    return [str(out.get(i, docs[i])) for i in range(len(docs))]


def generate_rewrites(docs, criterion, ask, batch=25, workers=8, label=''):
    """-> {doc_idx: rewrite}. Batched; a batch that fails to parse is simply dropped, and the
    caller falls back to the original text for those documents."""
    idx = list(range(len(docs)))
    chunks = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    out, failed = {}, 0

    def one(chunk):
        budget_guard()
        body = '\n'.join(f'{n}. {_clip(docs[i], 600)}' for n, i in enumerate(chunk, 1))
        prompt = (
            f"For each document below, write a phrase of at most 15 words capturing ONLY "
            f"{criterion}. Drop everything else - tone, detail, phrasing. Keep the same order.\n"
            f'Return ONLY a JSON array of {len(chunk)} strings.\n\nDocuments:\n{body}')
        try:
            txt = ask(prompt, max_tokens=40 * len(chunk)) or ''
            arr = json.loads(txt[txt.find('['): txt.rfind(']') + 1])
            if not isinstance(arr, list):
                return {}, 1
            return {chunk[n]: str(v)[:200] for n, v in enumerate(arr) if n < len(chunk)}, 0
        except Exception:
            return {}, 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, c) for c in chunks]):
            got, bad = fut.result()
            out.update(got); failed += bad
    if failed:
        print(f'    {label}: {failed}/{len(chunks)} rewrite batches failed, originals kept')
    return out


def exemplars(bench, setting, codebook=None, criterion=None, ask=None):
    """category id -> list of LLM-written example documents for that category."""
    src = _gen_dir(setting) / f'{bench}_exemplars.json'
    if src.exists():
        return json.loads(src.read_text())

    own = config.ARTIFACTS / 'generation' / f'{bench}_{setting}_exemplars.json'
    own.parent.mkdir(parents=True, exist_ok=True)
    if own.exists():
        return json.loads(own.read_text())

    if ask is None or codebook is None:
        raise RuntimeError(f'{bench}/{setting}: no cached exemplars and nothing to generate with')
    budget_guard()
    out = generate_exemplars(codebook, criterion, ask)
    own.write_text(json.dumps(out, indent=1))
    print(f'  {bench}/{setting}: generated exemplars for {len(out)} categories')
    return out


def generate_exemplars(codebook, criterion, ask, n=5, workers=8):
    """-> {category_id: [n short documents that would belong to it]}."""
    def one(cat):
        prompt = (
            f"A corpus is organised by {criterion}. One category is:\n"
            f"{cat['name']}" + (f" - {cat['definition']}" if cat.get('definition') else '') +
            f"\n\nWrite {n} short documents that would clearly belong to this category and "
            f"read like real corpus entries.\nReturn ONLY a JSON array of {n} strings.")
        try:
            txt = ask(prompt, max_tokens=60 * n) or ''
            arr = json.loads(txt[txt.find('['): txt.rfind(']') + 1])
            return str(cat['id']), [str(x)[:300] for x in arr][:n]
        except Exception:
            return str(cat['id']), []

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, c) for c in codebook]):
            cid, texts = fut.result()
            out[cid] = texts
    return out
