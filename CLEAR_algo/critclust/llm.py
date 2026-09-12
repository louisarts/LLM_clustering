"""Gateway client, per-model prompting, and batched document classification.

Classification is the dominant cost of this study, and the expensive part of each call is the
codebook rather than the document: a fine codebook runs to 100-190 named categories with
definitions and, under the original per-document scheme, was re-transmitted once per document.
`classify_batch` sends CLASSIFY_BATCH documents per call so the codebook is amortised, which
reduces the classification stages by roughly that factor.

Because batching changes the protocol, BOTH algorithms use it - CritClust_A's reference is
rebuilt here rather than read from V0's per-document cache - so the only differences between
the two algorithms remain the two intended ones.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from . import config
from .data import budget_guard

config.load_env()
config.add_helper_paths()

import LLM_call                                   # noqa: E402
import judge as judgemod                          # noqa: E402
from judge import _clip, classify_document        # noqa: E402

import os                                         # noqa: E402

CLIENT = OpenAI(base_url=os.environ['OPENAI_BASE_URL'],
                api_key=os.environ['OPENAI_API_KEY'],
                timeout=240, max_retries=2)


def make_ask(llm):
    """Return a callable prompt -> text for this model.

    Model-specific parameters: gemini needs reasoning_effort='none', glm-5 needs thinking
    disabled through extra_body, and the OpenAI and gemma models need neither.
    """
    def ask(prompt, max_tokens=2000, retries=4):
        kwargs = {}
        if 'gemini' in llm:
            kwargs['reasoning_effort'] = 'none'
        elif 'glm' in llm:
            kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
        for attempt in range(retries):
            try:
                r = CLIENT.chat.completions.create(
                    model=llm, temperature=0, max_tokens=max_tokens,
                    messages=[{'role': 'user', 'content': prompt}], **kwargs)
                return r.choices[0].message.content or ''
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)
    return ask


# Total context window per model. Anything not listed is assumed large. gemma-2-9b-it is the
# tight one at 8k, which constrains how much codebook plus how many documents fit in one call.
MODEL_CONTEXT = {
    'gemma-2-9b-it': 8192,
    'gemma-2-9b-it-embed': 8192,
    'gpt-3.5-turbo': 16385,
    'gpt-4o': 128000,
    'gpt-4o-mini': 128000,
    'gpt-4.1-mini': 128000,
}
DEFAULT_CONTEXT = 120000

# Largest completion allowed per call. V0's codebook asks for 60k thinking tokens, which the
# matched models refuse; the request has to be capped to what each one accepts.
MAX_OUTPUT = {
    'gemma-2-9b-it': 4096,
    'gpt-3.5-turbo': 4096,
    'gpt-4o': 16384,
    'gpt-4o-mini': 16384,
    'gpt-4.1-mini': 32768,
}
DEFAULT_MAX_OUTPUT = 60000


def max_output_of(llm):
    return MAX_OUTPUT.get((llm or '').split('/')[-1], DEFAULT_MAX_OUTPUT)
CHARS_PER_TOKEN = 4          # deliberately conservative

CURRENT_LLM = None           # set by bind_llm, read by the batch planner


def context_of(llm):
    return MODEL_CONTEXT.get((llm or '').split('/')[-1], DEFAULT_CONTEXT)


def plan_batch(codebook, llm=None, doc_clip=1000, reserve=600):
    """Choose (batch_size, doc_clip, codebook_text) that fit this model's context.

    Large codebooks are the binding constraint: 435 categories with definitions is ~9k tokens
    on its own, which does not fit an 8k model at any batch size. So definitions are dropped
    first (names alone still identify a category), then the per-document clip is tightened, and
    only then the batch shrinks. Returns None if even one document cannot be made to fit.
    """
    llm = llm or CURRENT_LLM
    budget = context_of(llm) - reserve
    full = codebook_text(codebook)
    names_only = '\n'.join(f"{c['id']}. {c['name']}" for c in codebook)

    for cbtxt in (full, names_only):
        cb_tokens = len(cbtxt) / CHARS_PER_TOKEN
        for clip in (doc_clip, 400, 250, 150):
            per_doc = clip / CHARS_PER_TOKEN + 12          # +12 for numbering and the reply entry
            room = budget - cb_tokens - 120                # 120 for the instruction block
            n = int(room / per_doc)
            if n >= 1:
                return min(config.CLASSIFY_BATCH, n), clip, cbtxt
    return None


def bind_llm(llm):
    """Point the shared RQ1 judge helpers at this model, and return its ask callable.

    `classify_document` and `build_codebook` call module-level `ask_llm`, so the binding has
    to happen before either is used.
    """
    ask = make_ask(llm)
    LLM_call.ask_llm = ask
    judgemod.ask_llm = ask
    LLM_call.MODEL = llm
    judgemod.MODEL = llm
    global CURRENT_LLM
    CURRENT_LLM = llm
    return ask


def codebook_text(codebook):
    return '\n'.join(
        f"{c['id']}. {c['name']}" + (f" - {c['definition']}" if c.get('definition') else '')
        for c in codebook)


def parse_labels(txt):
    """Accept [{"doc": 1, "category": 3}, ...] or a bare [3, 0, 12, ...]. -> {position: id}."""
    try:
        arr = json.loads(txt[txt.find('['): txt.rfind(']') + 1])
    except Exception:
        return None
    out = {}
    for pos, item in enumerate(arr, start=1):
        if isinstance(item, dict):
            doc, cat = item.get('doc'), item.get('category')
            if doc is None or cat is None:
                continue
            out[int(doc)] = int(cat)
        elif isinstance(item, (int, float)):
            out[pos] = int(item)
    return out or None


def classify_batch(indices, docs, codebook, spec, ask, batch_size=None, workers=8, label=''):
    """Classify many documents with few calls.

    Returns ({doc_idx: category}, n_calls). Category 0 means "fits none". A batch whose reply
    cannot be parsed is retried once and then falls back to per-document classification for
    that batch only, so one malformed reply costs a batch rather than the run.
    """
    plan = plan_batch(codebook, doc_clip=1000)
    if plan is None:
        raise RuntimeError(
            f'{label}: codebook of {len(codebook)} categories cannot fit the context of '
            f'{CURRENT_LLM} ({context_of(CURRENT_LLM)} tokens) even one document at a time')
    planned, clip_chars, cbtxt = plan
    batch_size = batch_size or planned
    idx = list(indices)
    chunks = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    result, fell_back = {}, 0
    if batch_size < config.CLASSIFY_BATCH:
        print(f'    {label}: batch {batch_size} @ {clip_chars} chars '
              f'({len(codebook)} categories, {context_of(CURRENT_LLM)}-token context)')

    def one_chunk(chunk):
        budget_guard()
        body = '\n'.join(f'{n}. {_clip(docs[i], clip_chars)}' for n, i in enumerate(chunk, 1))
        prompt = (
            f"Assign each document below to exactly one category from the codebook, "
            f"organising by {spec['criterion']}.\n\n"
            f"CODEBOOK:\n{cbtxt}\n\n"
            f"Use category id 0 if a document fits none of them. Do not invent categories.\n"
            f"Return ONLY a JSON array with one entry per document, in order:\n"
            f'[{{"doc":1,"category":<id>}}, ...]\n\n'
            f"DOCUMENTS ({len(chunk)}):\n{body}")
        budget = 60 + 22 * len(chunk)
        got = parse_labels(ask(prompt, max_tokens=budget) or '')
        if got is None:
            got = parse_labels(ask(prompt, max_tokens=budget) or '')
        if got is not None:
            return {chunk[p - 1]: c for p, c in got.items() if 1 <= p <= len(chunk)}, 0
        out = {}
        for i in chunk:
            try:
                c = classify_document(docs[i], codebook, spec)
            except Exception:
                c = None
            if c is not None:
                out[i] = int(c)
        return out, 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one_chunk, c) for c in chunks]):
            try:
                got, fb = fut.result()
            except Exception:
                got, fb = {}, 1
            result.update(got)
            fell_back += fb

    if fell_back:
        print(f'    {label}: {fell_back}/{len(chunks)} batches fell back to per-document')
    return result, len(chunks)
