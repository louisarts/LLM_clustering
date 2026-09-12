"""Codebook discovery - the one stage where the two algorithms genuinely differ in prompt.

CritClust_A  one call over a 300-document sample; the model chooses its own granularity and
             is told nothing about k.
CritClust_B  chunked incremental discovery that AIMS FOR k. This replaces V4's `1.4k + 4`
             category cap: the target is stated in the prompt instead of enforced afterwards,
             and discovery stops on the empty-batch rule alone.
"""
import json

import numpy as np

from . import config
from .llm import _clip


def codebook_A(docs, criterion, ask=None, attempts=3):
    """V0's single-shot codebook. Granularity is the model's choice.

    -> (categories, method). method is 'single_shot' normally, or 'chunked_no_target' when
    the single call truncated at the token limit on every attempt. On fine taxonomies the
    reply can exceed the budget (stackexchange_cl, k=121, is the known case), and V0's
    build_codebook raises rather than returning a partial list. Re-sampling usually fixes it;
    the fallback accumulates categories over chunks instead, which keeps the defining property
    of A's codebook - the model chooses the granularity, and is never told k - while removing
    the single oversized reply. Any benchmark that falls back is recorded in the results.
    """
    import judge as judgemod
    from judge import build_codebook
    from .llm import CURRENT_LLM, context_of, max_output_of
    if context_of(CURRENT_LLM) < 32000:
        # V0's single call asks for 60k output tokens over a 300-document prompt, which cannot
        # fit a small-context model at all. Chunked discovery keeps the defining property -
        # the model picks the granularity and is never told k - within a small window.
        print(f'    codebook_A: {CURRENT_LLM} has a '
              f'{context_of(CURRENT_LLM)}-token context, using chunked discovery')
        return codebook_A_chunked(docs, criterion, ask), 'chunked_small_context'
    # V0 asks _ask_thinking for 60k output tokens; the matched models refuse that, so cap the
    # request to what this one accepts. The prompt itself is unchanged.
    cap = max_output_of(CURRENT_LLM)
    original = judgemod._ask_thinking
    judgemod._ask_thinking = (
        lambda prompt, max_tokens=cap, retries=3: original(prompt, min(max_tokens, cap), retries))
    try:
        for attempt in range(attempts):
            rng = np.random.RandomState(attempt)
            idx = rng.choice(len(docs), min(300, len(docs)), replace=False)
            try:
                return (build_codebook([docs[i] for i in idx], {'criterion': criterion}),
                        'single_shot')
            except ValueError as e:
                print(f'    codebook_A attempt {attempt + 1}/{attempts} failed: {e}')
    finally:
        judgemod._ask_thinking = original

    if ask is None:
        raise RuntimeError('codebook_A: single-shot failed and no ask callable for fallback')
    print('    codebook_A: falling back to chunked discovery (no k target)')
    return codebook_A_chunked(docs, criterion, ask), 'chunked_no_target'


def codebook_A_chunked(docs, criterion, ask):
    """Fallback for codebook_A: chunked accumulation, granularity still the model's choice.

    Stopping is by SATURATION, not by a strict-zero rule. Asked "which categories do these 60
    documents need that are not already listed?", a model can always name something, so a
    zero-new-categories batch effectively never happens and discovery runs to the end of the
    sample - producing 700+ near-duplicate categories that paraphrase individual documents
    rather than describing a taxonomy. Stopping once batches stop contributing *materially*
    fixes that without telling the model k, which A must never see.
    """
    from .llm import CURRENT_LLM, context_of
    small = context_of(CURRENT_LLM) < 32000
    cb_batch = 20 if small else config.CB_BATCH
    clip = 150 if small else 250
    rng = np.random.RandomState(0)
    idx = rng.choice(len(docs), min(config.CB_SAMPLE, len(docs)), replace=False)
    cats, seen, quiet = [], set(), 0
    for start in range(0, len(idx), cb_batch):
        if quiet >= config.CB_SATURATE_ROUNDS:
            break
        if len(cats) >= config.CB_HARD_CAP:
            print(f'    codebook_A_chunked: hard cap {config.CB_HARD_CAP} reached')
            break
        batch = [docs[i] for i in idx[start:start + cb_batch]]
        known = ', '.join(c['name'] for c in cats) if cats else '(none yet)'
        body = '\n'.join(f'- {_clip(d, clip)}' for d in batch)
        prompt = (
            f"We are designing a codebook for organising a corpus by {criterion}.\n"
            f"Categories discovered so far: {known}.\n\n"
            f"Below are more documents. List ONLY the NEW categories (by {criterion}) that "
            f"these documents need and that are NOT already in the list above. Choose the "
            f"granularity you judge most natural for this corpus; do not force a round "
            f"number. If none are new, return [].\n"
            f'Return ONLY a JSON array: [{{"name":...,"definition":...}}, ...].\n\n'
            f"Documents:\n{body}")
        added = 0
        for c in _parse_array(ask(prompt) or ''):
            name = str(c.get('name', '')).strip()
            if name and _norm(name) not in seen:
                seen.add(_norm(name))
                cats.append({'id': len(cats) + 1, 'name': name[:80],
                             'definition': str(c.get('definition', ''))[:200]})
                added += 1
        quiet = quiet + 1 if added <= config.CB_SATURATE_NEW else 0

    return consolidate(cats, criterion, ask)


def consolidate(cats, criterion, ask, chunk=80):
    """Merge the accumulated categories into a coherent codebook.

    Chunked discovery asks "what is new in these documents?" over and over and never revisits
    what it already wrote, so near-duplicates accumulate and the list drifts from a taxonomy
    into paraphrases of individual documents. A single-shot codebook does not have this problem
    because the model writes the whole list at once and can see it. This restores that: the
    model is shown its own accumulated categories and asked to merge duplicates and
    over-specific entries into one mutually exclusive list.

    k is never mentioned, so the granularity remains the model's own choice - the model is only
    asked to be consistent with itself.
    """
    if len(cats) <= config.CB_CONSOLIDATE_MIN:
        return cats

    working = cats
    for _ in range(config.CB_CONSOLIDATE_ROUNDS):
        merged, before = [], len(working)
        for start in range(0, len(working), chunk):
            block = working[start:start + chunk]
            listing = '\n'.join(
                f"- {c['name']}" + (f": {c['definition']}" if c.get('definition') else '')
                for c in block)
            prompt = (
                f"Below is a draft list of categories for organising a corpus by {criterion}. "
                f"It was built up in passes and contains duplicates, near-duplicates, and "
                f"entries that describe one document rather than a category.\n\n"
                f"Rewrite it as a clean list: merge categories that mean the same thing, "
                f"generalise entries that are too specific to a single document, and drop "
                f"nothing that is genuinely distinct. Do not force any particular number of "
                f"categories - use however many the corpus needs.\n"
                f'Return ONLY a JSON array: [{{"name":...,"definition":...}}, ...]\n\n'
                f"Draft list:\n{listing}")
            from .llm import CURRENT_LLM, max_output_of
            budget = min(6000, max_output_of(CURRENT_LLM))
            got = _parse_array(ask(prompt, max_tokens=budget) or '')
            if got:
                merged.extend({'name': str(c.get('name', '')).strip()[:80],
                               'definition': str(c.get('definition', ''))[:200]}
                              for c in got if str(c.get('name', '')).strip())
            else:
                merged.extend(block)

        seen, out = set(), []
        for c in merged:
            key = _norm(c['name'])
            if key not in seen:
                seen.add(key)
                out.append({'id': len(out) + 1, 'name': c['name'], 'definition': c['definition']})
        working = out
        print(f'    consolidate: {before} -> {len(working)} categories')
        if len(working) >= before * config.CB_CONSOLIDATE_STOP or len(working) <= chunk:
            break
    return working


_STOP_WORDS = {'inquiry', 'request', 'assistance', 'management', 'information', 'query',
               'issue', 'status', 'the', 'a', 'an', 'of', 'for', 'and', 'to'}


def _norm(name):
    """Normalise a category name for duplicate detection.

    Exact-string matching lets 'Credit Card Replacement Status Inquiry' and 'Card Usage Issue
    Inquiry' both through as distinct categories. Stripping parenthesised qualifiers and the
    filler nouns that models attach to almost every name collapses those onto their content
    words, so near-duplicates are recognised as duplicates.
    """
    import re
    n = re.sub(r'\([^)]*\)', ' ', name.lower())
    n = re.sub(r'[^a-z0-9 ]+', ' ', n)
    words = sorted(w for w in n.split() if w not in _STOP_WORDS)
    return ' '.join(words) or name.lower()


def _parse_array(txt):
    try:
        return json.loads(txt[txt.find('['): txt.rfind(']') + 1])
    except Exception:
        return []


def codebook_B_aim_k(docs, criterion, k, ask):
    """Chunked discovery with k as a stated target. -> (categories, hit_runaway).

    Each batch is shown the categories found so far and asked only for genuinely new ones, with
    the target k and the running count in the prompt so the model can calibrate how finely to
    slice. Discovery ends when the codebook reaches k categories, or earlier if CB_EMPTY_STOP
    consecutive batches add nothing. The returned list is truncated to exactly k, since a batch
    can complete after the ceiling is reached.

    The second return value reports whether the ceiling bound the result: True means the model
    kept proposing categories and was cut off, False means it converged below k on its own.
    """
    from .llm import CURRENT_LLM, context_of
    small = context_of(CURRENT_LLM) < 32000
    cb_batch = 20 if small else config.CB_BATCH
    clip = 150 if small else 250
    rng = np.random.RandomState(0)
    idx = rng.choice(len(docs), min(config.CB_SAMPLE, len(docs)), replace=False)
    cats, seen, empties, hit_runaway = [], set(), 0, False

    for start in range(0, len(idx), cb_batch):
        if empties >= config.CB_EMPTY_STOP:
            break
        if len(cats) >= config.CB_RUNAWAY(k):
            hit_runaway = True
            break   # the cap bound the result rather than the model converging

        batch = [docs[i] for i in idx[start:start + cb_batch]]
        known = ', '.join(c['name'] for c in cats) if cats else '(none yet)'
        if small and len(known) > 4000:
            known = ', '.join(c['name'] for c in cats[-120:]) + ' (earlier ones omitted)'
        body = '\n'.join(f'- {_clip(d, clip)}' for d in batch)
        prompt = (
            f"We are building a codebook that organises a corpus by {criterion}. "
            f"The target is {k} categories in total.\n"
            f"Categories discovered so far ({len(cats)} of a target {k}): {known}.\n\n"
            f"Below are more documents. List ONLY the NEW categories (by {criterion}) that "
            f"these documents need and that are NOT already in the list above. Aim for a "
            f"final list of about {k} categories, so choose a level of detail that will get "
            f"there. If none are new, return [].\n"
            f'Return ONLY a JSON array: [{{"name":...,"definition":...}}, ...].\n\n'
            f"Documents:\n{body}")

        added = 0
        for c in _parse_array(ask(prompt) or ''):
            name = str(c.get('name', '')).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                cats.append({'id': len(cats) + 1,
                             'name': name[:80],
                             'definition': str(c.get('definition', ''))[:200]})
                added += 1
        empties = empties + 1 if added == 0 else 0

    return cats[:k], hit_runaway
