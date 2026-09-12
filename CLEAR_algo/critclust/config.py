"""Paths, constants and benchmark configuration for CLEAR (thesis variant C).

Settings:
    'unmatched'    19 benchmarks, gemini-embedding-001 + glm-5 (the default configuration).
    'matched'      each benchmark on its NMI record holder's own encoder and LLM.
    'embmatched'   encoder matched, LLM left at glm-5 (ablation).
    'llmmatched'   LLM matched, encoder left at gemini-embedding-001 (ablation).
"""
from pathlib import Path
import json
import os
import sys

# --------------------------------------------------------------------------- paths
PKG = Path(__file__).resolve().parent
CLEAN = PKG.parent                      # CLEAR_algo/
ROOT = CLEAN.parent                     # repository root
DATA = ROOT / 'data'

# read-only inputs (embedding caches regenerate on demand into artifacts/)
NEWBENCH = DATA / 'benchmarks'
GEMINI_EMB = CLEAN / 'artifacts' / 'gemini_emb'    # gemini-embedding-001 raw document embeddings
GLM5_EMB = CLEAN / 'artifacts' / 'glm5_emb'        # glm-5 rewrite and exemplar embeddings
GLM5_GEN = DATA / 'glm5_gen'          # unmatched generation artifacts
MATCHED_GEN = DATA / 'matched_gen'    # matched generation artifacts
MATCHED_EMB = CLEAN / 'artifacts' / 'matched_emb'  # matched embeddings
E5_CACHE = CLEAN / 'artifacts' / 'seed_cache'      # locally computed e5-large-v2 embeddings
PUBLISHED = DATA / 'published_results.csv'

# run outputs go to the repo-level results/ and figures/ folders
RESULTS = ROOT / 'results' / 'clear'
ARTIFACTS = CLEAN / 'artifacts'
CODEBOOKS = ARTIFACTS / 'codebooks'           # codebooks + reference labellings
REPAIR_CACHE = ARTIFACTS / 'repair'           # cluster names + classify caches
EMB_CACHE = ARTIFACTS / 'embeddings'           # embeddings this study computes itself
FIGURES = ROOT / 'figures'                    # vector PDFs for LaTeX
PLOTS_PNG = None                              # set to a Path to also write titled PNG copies

for _d in (RESULTS, ARTIFACTS, CODEBOOKS, REPAIR_CACHE, EMB_CACHE, FIGURES,
           GEMINI_EMB, GLM5_EMB, MATCHED_EMB, E5_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- credentials
def load_env():
    """Populate os.environ from the repository .env (gateway base URL and key)."""
    env = ROOT / '.env'
    if not env.exists():
        raise FileNotFoundError(f'no .env at {env}')
    for line in env.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())


HELPERS = ROOT / 'codebook_judge'   # judge.py and LLM_call.py live here


def add_helper_paths():
    """judge.py and LLM_call.py both live in codebook_judge/."""
    for d in (HELPERS,):
        p = str(d)
        if p not in sys.path:
            sys.path.append(p)


add_rq1_to_path = add_helper_paths   # backwards-compatible alias


# --------------------------------------------------------------------------- run constants
SEEDS = [0, 1, 2, 3, 4]
BUDGET_STOP = 999.0            # dollars of gateway spend; runs abort above this
CLASSIFY_BATCH = 20            # documents per classification call

REPAIR_ROUNDS = 3              # both algorithms
REPAIR_DOSE = 200              # lowest-margin documents re-classified per round

TIE_EPS = 0.01                 # CritClust_A: less-LLM tie-break tolerance
B_BOOT = 15                    # CritClust_B: bootstrap resamples of the judging set
JUDGE_SPLIT = 0.70             # CritClust_B: train / judge fraction of the reference

# CritClust_B codebook discovery. V4 capped the codebook at 1.4k+4 categories; CritClust_B
# caps it at exactly k. The prompt states k as the target and the cap enforces it, so the
# reference labelling ends up at the same granularity as the clustering being judged.
CB_SAMPLE = 1200
CB_BATCH = 60
CB_EMPTY_STOP = 3
# codebook_A's chunked fallback: stop once batches stop contributing materially, and bound the
# total. Neither reveals k, so A's protocol (the model chooses its own granularity) survives.
CB_SATURATE_NEW = 2        # a batch adding <= this many genuinely new categories counts as quiet
CB_SATURATE_ROUNDS = 2     # this many quiet batches in a row ends discovery
CB_HARD_CAP = 250          # backstop; a codebook this large is a failure, not a taxonomy
# After chunked discovery, the model is shown its own accumulated list and asked to merge
# duplicates and over-specific entries. This is what a single-shot codebook gets for free and
# what the accumulating prompt never gets. k is not mentioned, so granularity stays the
# model's choice.
CB_CONSOLIDATE_MIN = 60       # lists shorter than this are left alone
CB_CONSOLIDATE_ROUNDS = 3     # repeat while it keeps shrinking materially
CB_CONSOLIDATE_STOP = 0.85    # stop once a round removes less than 15%
def CB_RUNAWAY(k):
    """Hard ceiling on CritClust_B's codebook: exactly k categories."""
    return k

# CritClust_A candidate preference order for the tie-break (least LLM involvement first)
CAND_ORDER_A = ['kmeans_raw', 'gmm_raw', 'exemplar_raw', 'gmm_concat', 'exemplar_concat']

# --------------------------------------------------------------------------- benchmarks
BENCH19 = [
    'banking77', 'clinc150', 'clinc150_domain', 'mtop_intent', 'mtop_domain',
    'massive_intent', 'massive_domain', 'tweet89', 'stackoverflow20', 'mcid',
    'snips', 'dbpedia', 'reddit_s2s', 'arxiv_fine', 'stackexchange_cl',
    'fewrel', 'fewnerd', 'fewevent', 'bbcnews',
]

# bench -> (embedder, llm, sota holder)
MATCHED_CFG = {
    'banking77'         : ('text-embedding-3-small', 'gpt-4o', 'k-LLMmeans'),
    'massive_domain'    : ('text-embedding-3-small', 'gpt-4o', 'k-LLMmeans'),
    'massive_intent'    : ('instructor-large', 'gemma-2-9b-it', 'SPILL'),
    'mtop_domain'       : ('instructor-large', 'gpt-3.5-turbo', 'ClusterLLM-I'),
    'arxiv_fine'        : ('e5-large-v2', 'gpt-3.5-turbo', 'ClusterLLM-E-iter'),
    'stackexchange_cl'  : ('instructor-large', 'gpt-3.5-turbo', 'ClusterLLM-I-iter'),
    'clinc150'          : ('instructor-large', 'gemma-2-9b-it', 'LUMI'),
    'clinc150_domain'   : ('instructor-large', 'gpt-3.5-turbo', 'LLMEdgeRefine'),
    'mtop_intent'       : ('instructor-large', 'gemma-2-9b-it', 'LUMI'),
    'fewrel'            : ('e5-large-v2', 'gpt-3.5-turbo', 'ClusterLLM-E-iter'),
    'fewevent'          : ('instructor-large', 'gpt-3.5-turbo', 'ClusterLLM-I-iter'),
    'stackoverflow20'   : ('paraphrase-mpnet', 'gpt-3.5-turbo', 'IDAS'),
    'tweet89'           : ('instructor-large', 'gpt-4o-mini', 'Cequel'),
    'bbcnews'           : ('instructor-large', 'gpt-4o-mini', 'Cequel'),
    'reddit_s2s'        : ('instructor-large', 'gpt-4o-mini', 'Cequel'),
    'mcid'              : ('instructor-large', 'gpt-4o-mini', 'NILC'),
    'snips'             : ('instructor-large', 'gpt-4o-mini', 'NILC'),
    'dbpedia'           : ('instructor-large', 'gpt-4o-mini', 'NILC'),
    'fewnerd'           : ('gemini-embedding-001', 'gpt-4.1-mini', 'LLM-MemCluster'),
}
BENCH8 = list(MATCHED_CFG)

UNMATCHED_EMBEDDER = 'gemini-embedding-001'
UNMATCHED_LLM = 'vertex_ai/zai-org/glm-5-maas'

# Hybrid settings complete the 2x2 factorial that decomposes the ingredient confound:
#   embmatched    embedder matched per benchmark, LLM the unmatched best (glm-5)
#   llmmatched    embedder the unmatched best (gemini), LLM matched per benchmark
# LLM-side artifacts (codebook, reference labelling, rewrite and exemplar texts) depend
# only on the LLM, so each hybrid reuses them from the run that shares its LLM.
HYBRID_LLM_SRC = {'embmatched': 'unmatched', 'llmmatched': 'matched'}
HYBRID_EMB_SRC = {'embmatched': 'matched', 'llmmatched': 'unmatched'}
SETTINGS = ['unmatched', 'matched', 'embmatched', 'llmmatched']


def llm_source_setting(setting):
    """The setting whose LLM (and cached LLM artifacts) this setting uses."""
    return HYBRID_LLM_SRC.get(setting, setting)


def emb_source_setting(setting):
    """The setting whose embedder this setting uses."""
    return HYBRID_EMB_SRC.get(setting, setting)


def models_for(bench, setting):
    """-> (embedder, llm) for this benchmark under this setting."""
    embedder = (UNMATCHED_EMBEDDER if emb_source_setting(setting) == 'unmatched'
                else MATCHED_CFG[bench][0])
    llm = (UNMATCHED_LLM if llm_source_setting(setting) == 'unmatched'
           else MATCHED_CFG[bench][1])
    return embedder, llm


def benchmarks_for(setting):
    return BENCH19 if setting == 'unmatched' else BENCH8


# --------------------------------------------------------------------------- criteria
SOTA_CONFIG_FILE = CLEAN / 'sota_config.json'

def sota_config():
    """bench -> {method, embedder, llm, note}: who holds the record and what they ran it on."""
    return json.loads(SOTA_CONFIG_FILE.read_text())


CRITERIA_FILE = CLEAN / 'criteria.json'

def criteria():
    """bench -> the one-sentence clustering criterion given to every LLM call."""
    return json.loads(CRITERIA_FILE.read_text())


# --------------------------------------------------------------------------- display names
BENCH_DISPLAY = {
    'banking77': 'Banking77', 'clinc150': 'CLINC(I)', 'clinc150_domain': 'CLINC(D)',
    'mtop_intent': 'MTOP(I)', 'mtop_domain': 'MTOP(D)', 'massive_intent': 'MASSIVE(I)',
    'massive_domain': 'MASSIVE(D)', 'tweet89': 'Tweet89', 'stackoverflow20': 'StackOverflow',
    'mcid': 'M-CID', 'snips': 'SNIPS', 'dbpedia': 'DBPedia', 'reddit_s2s': 'Reddit-S2S',
    'arxiv_fine': 'ArXiv-S2S', 'stackexchange_cl': 'StackEx-S2S', 'fewrel': 'FewRel',
    'fewnerd': 'FewNerd', 'fewevent': 'FewEvent', 'bbcnews': 'BBC News',
}

CAND_DISPLAY = {
    'kmeans_raw': 'k-means (raw)', 'gmm_raw': 'GMM (raw)', 'gmm_concat': 'GMM (avg)',
    'exemplar_raw': 'exemplar init (raw)', 'exemplar_concat': 'exemplar init (avg)',
    'consensus': 'consensus', 'seeded': 'reference-seeded', 'lda': 'LDA projection',
    'km_concat': 'k-means (avg)', 'km_rewrite': 'k-means (rewrite)',
    'agglo_concat': 'Ward (avg)', 'agglo_raw': 'Ward (raw)',
    'bisect_raw': 'bisecting k-means', 'km_pca50': 'k-means (PCA-50)',
    'head': 'contrastive head',
}


def run_name(algo, setting, aligned=False):
    """Canonical stem for every file a run produces. aligned=True marks runs made under
    the RQ1-aligned judge protocol (separate stems; legacy results are never clobbered)."""
    return f'critclust_{algo}_{setting}' + ('_aligned' if aligned else '')


# ------------------------------------------------------------------ aligned judge protocol
# The judge inside CritClust, aligned with the RQ1 instrument (see results/ALIGNED_PROTOCOL.md):
#   - judge LLM: glm-5 wherever the LLM-source is 'unmatched' (kept deliberately), and
#     gemini-2.5-flash (the RQ1-validated instance) wherever it is 'matched';
#   - matched-source judge codebooks are fresh Flash single-shots via RQ1's build_codebook;
#   - the reference labelling is a FRESH 1,000-document sample per seed (RQ1's N_ASSIGN),
#     classified in batches (batching is the one retained deviation, disclosed).
JUDGE_FLASH = 'vertex_ai/gemini-2.5-flash'
ALIGNED_REF_N = 1000


def judge_llm_for(setting):
    """The LLM the aligned judge runs on under this setting."""
    return UNMATCHED_LLM if llm_source_setting(setting) == 'unmatched' else JUDGE_FLASH
