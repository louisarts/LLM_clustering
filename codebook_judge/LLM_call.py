"""Single entry point for LLM calls used across the codebase.

ask_llm(prompt) sends a prompt to the configured model and returns its text
response. Import this everywhere an LLM call is needed rather than re-creating
the client.
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- PROVIDER TOGGLE ---------------------------------------------------------
# 'google'      TEMPORARY direct-Google mode: a Vertex/Gemini API key (AQ....)
#               in OPENAI_API_KEY, hitting the Gemini API's OpenAI-compatible
#               endpoint. NOTE: this key cannot access the 2.5 generation
#               (gated for new accounts), so it runs gemini-3-flash-preview —
#               results produced in this mode come from a NEWER model
#               generation than the cached 2.5-flash artifacts; documented in
#               the project notes.
# 'chattermill' ORIGINAL setup (flip back when the new gateway key arrives):
#               LiteLLM virtual key (sk-...) in OPENAI_API_KEY and
#               OPENAI_BASE_URL=<your gateway base URL> in
#               .env (the bare OpenAI() client reads both), model
#               'vertex_ai/gemini-2.5-flash'.
PROVIDER = "chattermill"  # waiting for the new gateway key; google mode was never used for any saved artifact

if PROVIDER == "google":
    client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    MODEL = "gemini-3-flash-preview"
else:  # 'chattermill'
    client = OpenAI()  # base_url + key from .env
    MODEL = "vertex_ai/gemini-2.5-flash"

# --- THINKING DISABLED (2026-07-20) ------------------------------------------
# Gemini 2.5 Flash thinks by default, and thinking tokens bill at the output
# rate ($2.50/M vs $0.30/M input). On the judge prompts that meant ~249 output
# tokens to answer a multiple-choice question, i.e. ~85% of the per-call cost
# was deliberation we do not need: these are pattern-matching tasks (spot the
# intruder, split eight documents in two), not reasoning tasks.
#
# Measured on a representative intruder prompt through the gateway:
#   baseline                       384 in / 249 out  = $0.000738/call
#   reasoning_effort='none'        384 in /   7 out  = $0.000133/call  <-- 5.5x
#   extra_body thinking configs    silently IGNORED, full price
# Same answer in every case; 'none' also returns bare JSON instead of a
# ```json fence, so replies parse more reliably.
#
# This changes the measuring instrument: judge scores produced before this date
# (all of experiments 1-7, and any RQ1 rows judged before it) are NOT
# comparable with scores produced after, and must not be mixed in one analysis.
REASONING_EFFORT = "none" if PROVIDER == "chattermill" else None


def ask_llm(prompt, max_tokens=2000, retries=4):
    """Send a prompt to the LLM and return its text response.

    Retries transient failures (rate limits, timeouts, gateway hiccups) with
    exponential backoff — essential for long unattended runs, where a single
    unhandled blip would otherwise crash a whole benchmark stage or silently
    poison a cache entry. Raises only after all attempts fail.
    """
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                **extra,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)  # 2s, 4s, 8s
