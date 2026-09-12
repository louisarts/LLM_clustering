"""The LLM judge: codebook-first weak annotation.

Three stages:

  1. DISCOVERY  (script 05, one call per corpus)
     The LLM reads a sample of documents and writes a CODEBOOK - category names
     with one-line definitions - guided by a one-sentence statement of what the
     taxonomy is organised by (CRITERIA below). The model chooses the
     granularity itself; nothing in the prompt says how fine to slice.
     (A map-reduce variant - N_SAMPLES > 1 disjoint samples merged by
     merge_codebooks - was tried and REJECTED: it correlated worse with humans,
     pooled judge~AMI +0.77 vs +0.80. Kept as a reproducible ablation.)

  2. ASSIGNMENT (script 06, one cheap call per sampled document)
     The LLM classifies ~1,000 sampled documents against the codebook, producing
     a reference labelling of the sample.

  3. SCORING    (script 07, free)
     A candidate clustering's judge score is the AMI between its labels and the
     reference labelling. AMI is chance-corrected for granularity, so clusterings
     with different numbers of clusters are directly comparable.

No gold labels are used at any stage. The codebook is saved as readable JSON so
the judge's understanding of a corpus can be inspected before any scoring.

Model: vertex_ai/gemini-2.5-flash, temperature 0. Assignments run with
reasoning_effort='none'; the single codebook call runs with default reasoning.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))   # LLM_call.py sits next to this file

from LLM_call import ask_llm, client, MODEL                         # noqa: E402

# Per benchmark: what the taxonomy is organised by. One line of human intent -
# never labels, never example pairs, and nothing about how fine to slice.
# Criteria for the 19-benchmark literature suite (single source of truth:
# criteria.json next to this file, loaded verbatim).
import json as _json
from pathlib import Path as _P
CRITERIA = {b: {'criterion': c} for b, c in _json.loads(
    (_P(__file__).resolve().parent / 'criteria.json')
    .read_text()).items()}


def _clip(doc, max_chars=1000):
    return ' '.join(str(doc).split())[:max_chars]


def _ask_thinking(prompt, max_tokens=60000, retries=3):
    """One call WITH default reasoning (thinking on). Used only for the codebook -
    a single quality-critical call per dataset, so the extra cents are justified.

    max_tokens is generous because thinking tokens count against it; a tight
    budget truncates the JSON mid-list. Returns (text, finish_reason).
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            return (choice.message.content or ''), (choice.finish_reason or '')
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)


def _extract_json(text):
    """Parse the first {...} block in a reply (tolerates markdown fences)."""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        raise ValueError(f'no JSON object found in reply: {text[:300]!r}')
    return json.loads(m.group())


def _parse_categories(reply):
    """Layered parsing of the codebook reply, from strict to salvage.

    LLM JSON fails in predictable ways (trailing commas, an unescaped quote
    inside one definition). Layer 3 recovers every well-formed entry
    individually, so one bad definition does not cost the whole codebook.
    """
    # 1) strict
    try:
        return _extract_json(reply)['categories']
    except Exception:
        pass
    # 2) strict after removing trailing commas
    try:
        m = re.search(r'\{.*\}', reply, re.DOTALL)
        if m:
            return json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))['categories']
    except Exception:
        pass
    # 3) entry-level salvage
    pat = re.compile(
        r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"name"\s*:\s*"(.*?)"\s*,'
        r'\s*"definition"\s*:\s*"(.*?)"\s*\}', re.DOTALL)
    entries = [{'id': int(i), 'name': n, 'definition': d}
               for i, n, d in pat.findall(reply)]
    if len(entries) >= 2:
        return entries
    raise ValueError(f'could not parse a codebook from the reply; it starts: '
                     f'{reply[:400]!r}')


def build_codebook(sample_docs, spec, clip_chars=400, raw_out=None):
    """DISCOVERY: one thinking-on call. Returns [{'id', 'name', 'definition'}, ...].

    The sample shows the model WHAT is in the corpus; the criterion says what the
    taxonomy is organised by. The model chooses the granularity itself - the
    prompt says nothing about how finely to slice. It is told to cover what it
    sees and not to invent absent categories or force a round count.

    If raw_out is given, the raw reply is saved there (provenance + debugging).
    """
    body = '\n'.join(f'{i}. {_clip(d, clip_chars)}' for i, d in enumerate(sample_docs, 1))
    prompt = (
        f"You are designing a codebook for organising a corpus of documents by "
        f"{spec['criterion']}.\n\n"
        f"Below are {len(sample_docs)} documents sampled from the corpus. Read them all, "
        f"then write the complete list of categories needed to organise this corpus by "
        f"{spec['criterion']}.\n"
        f"- Choose the granularity you judge most natural for this corpus and purpose; "
        f"use however many categories the corpus needs.\n"
        f"- Cover every category you can see evidence for in the sample.\n"
        f"- Do NOT invent categories that are absent, and do not force a round number.\n"
        f"- Definitions must be one line and mutually exclusive where possible.\n"
        f"- Plain text only inside names and definitions: no double quotes, no "
        f"backslashes.\n\n"
        f'Respond only with JSON:\n'
        f'{{"categories": [{{"id": 1, "name": "...", "definition": "..."}}, ...]}}\n\n'
        f'Documents:\n{body}'
    )
    reply, finish = _ask_thinking(prompt)
    if raw_out is not None:
        Path(raw_out).write_text(reply)
    if finish == 'length':
        raise ValueError('codebook reply was truncated at the token limit - '
                         'rerun (or raise max_tokens in _ask_thinking)')
    cats = _parse_categories(reply)
    out = []
    for i, c in enumerate(cats, 1):
        out.append({'id': i, 'name': str(c.get('name', f'category_{i}')).strip(),
                    'definition': str(c.get('definition', '')).strip()})
    if len(out) < 2:
        raise ValueError(f'degenerate codebook ({len(out)} categories) - inspect the reply')
    return out


def merge_codebooks(drafts, spec, raw_out=None):
    """REDUCE step of map-reduce discovery (ABLATION ONLY - not used by the
    final judge). One thinking-on call that merges independently-drafted
    codebooks into one.

    Tried with 3 disjoint 300-doc samples and rejected: the merge consolidated
    codebooks too coarsely where fineness mattered (banking77, stackoverflow)
    and concatenated where it did not (20newsgroups), dropping pooled
    judge~AMI from +0.80 to +0.77. Kept so the ablation is reproducible
    (set N_SAMPLES > 1 in 05_build_codebook.py).
    """
    blocks = []
    for i, cb in enumerate(drafts, 1):
        blocks.append(f'Draft codebook {i}:\n' +
                      '\n'.join(f"- {c['name']} - {c['definition']}" for c in cb))
    prompt = (
        f"Several annotators each read a DIFFERENT sample of documents from the same "
        f"corpus and independently drafted a codebook for organising the corpus by "
        f"{spec['criterion']}.\n\n"
        f"Merge the drafts below into ONE final codebook for organising the corpus by "
        f"{spec['criterion']}.\n"
        f"- Combine categories that refer to the same thing, even when named or worded "
        f"differently; write one clean name and definition for the combined category.\n"
        f"- KEEP categories that are genuinely distinct, including those appearing in "
        f"only one draft - each annotator saw different documents, so a category seen "
        f"once is still real.\n"
        f"- Choose the granularity you judge most natural for this corpus and purpose; "
        f"use however many categories the corpus needs.\n"
        f"- Do NOT invent categories absent from every draft, and do not force a round "
        f"number.\n"
        f"- Definitions must be one line and mutually exclusive where possible.\n"
        f"- Plain text only inside names and definitions: no double quotes, no "
        f"backslashes.\n\n"
        f'Respond only with JSON:\n'
        f'{{"categories": [{{"id": 1, "name": "...", "definition": "..."}}, ...]}}\n\n'
        + '\n\n'.join(blocks)
    )
    reply, finish = _ask_thinking(prompt)
    if raw_out is not None:
        Path(raw_out).write_text(reply)
    if finish == 'length':
        raise ValueError('merge reply was truncated at the token limit - rerun')
    cats = _parse_categories(reply)
    out = [{'id': i, 'name': str(c.get('name', f'category_{i}')).strip(),
            'definition': str(c.get('definition', '')).strip()}
           for i, c in enumerate(cats, 1)]
    if len(out) < 2:
        raise ValueError(f'degenerate merged codebook ({len(out)} categories) - '
                         f'inspect the reply')
    return out


def format_codebook(codebook):
    return '\n'.join(f"{c['id']}. {c['name']} - {c['definition']}" for c in codebook)


def classify_document(doc, codebook, spec):
    """ASSIGNMENT: one cheap (thinking-off) call. Returns a category id, or 0 for
    'fits none of these', or None if the reply is unusable."""
    prompt = (
        f"You are classifying one document by {spec['criterion']}.\n\n"
        f"Categories:\n{format_codebook(codebook)}\n\n"
        f"Which single category does the document belong to? If it truly fits none of "
        f"them, answer 0.\n"
        f'Respond only with JSON: {{"category": <number>}}\n\n'
        f'Document: {_clip(doc)}'
    )
    reply = ask_llm(prompt) or ''
    m = re.search(r'"category"\s*:\s*(\d+)', reply)
    if not m:
        m = re.search(r'\b(\d{1,3})\b', reply)
        if not m:
            return None
    cat = int(m.group(1))
    return cat if 0 <= cat <= len(codebook) else None


def score_partition(ref_doc_idx, ref_categories, membership, min_docs=50):
    """SCORING (free): AMI between the reference labelling and a candidate clustering,
    over the reference documents the candidate covers (membership >= 0).

    AMI is chance-corrected for granularity, so scores are comparable across
    candidates with different numbers of clusters. Returns (ami, n_docs_used).
    """
    from sklearn.metrics import adjusted_mutual_info_score
    mem = np.asarray(membership)[np.asarray(ref_doc_idx, dtype=int)]
    ok = mem >= 0
    if ok.sum() < min_docs:
        return np.nan, int(ok.sum())
    ref = np.asarray(ref_categories)[ok]
    lab = mem[ok]
    if len(np.unique(ref)) < 2 or len(np.unique(lab)) < 2:
        return np.nan, int(ok.sum())
    return float(adjusted_mutual_info_score(ref, lab)), int(ok.sum())
