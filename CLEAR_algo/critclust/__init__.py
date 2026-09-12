"""CritClust: the CLEAR pipeline package.

    CritClust_A   V0, unchanged.
    CritClust_B   V4 with the codebook prompt aiming for k, and boundary repair restored.

Typical use from a script:

    from critclust import config, runners
    df = runners.run_B('unmatched')
    summary, candidates = runners.save(df, 'B', 'unmatched')

Typical use from the analysis notebook (no LLM calls, reads results/ from disk):

    from critclust import plots, runners
    runs, summary, candidates = runners.load('B', 'unmatched')
    plots.barplot(summary, 'NMI', 'CritClust_B / unmatched', 'ccb_unmatched_nmi.pdf')

`critclust.plots` and `critclust.config` are safe to import without gateway credentials;
importing `runners`, `llm`, `generation`, `embeddings` or `repair` constructs the gateway
client and therefore needs a readable .env.
"""
from . import config  # noqa: F401

__all__ = ['config', 'data', 'llm', 'codebooks', 'generation', 'embeddings',
           'pools', 'judges', 'repair', 'runners', 'plots']
