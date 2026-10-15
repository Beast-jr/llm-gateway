# Prices in US dollars per 1 MILLION tokens, per model.
# Source: third-party price trackers (Oct 2026). Double-check https://groq.com/pricing
# and edit these numbers if Groq changes them.
PRICES_PER_MILLION = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}

# Exact cache (Redis)
CACHE_TTL_SECONDS = 24 * 60 * 60  # how long a saved answer is kept: 24 hours
REDIS_TIMEOUT_SECONDS = 0.5  # if Redis takes longer than this, skip the cache and ask Groq

# Semantic cache: embeddings (fastembed, runs locally, no API calls)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small makes vectors of 384 numbers; must match the DB column
EMBEDDING_CACHE_DIR = ".cache/fastembed"  # where the downloaded model is kept (git-ignored)

# Semantic cache: lookup
SEMANTIC_CUTOFF = 0.96  # fallback after the verifier did not clearly win (see eval/cutoff_results.md); numbers guard is also on
SEMANTIC_PREFILTER = 0.85  # EVALUATION ONLY: embedding score above which scripts/tune_cutoff.py runs the verifier
SEMANTIC_DUPLICATE_SIMILARITY = 0.99  # don't save a question if one this similar is already stored
SEMANTIC_TIMEOUT_SECONDS = 1.0  # embed + DB lookup together; if slower, skip it and ask Groq

# Semantic cache: verifier - evaluated but NOT used in production (see eval/cutoff_results.md).
# (a small model that reads BOTH questions together; runs locally via ONNX)
VERIFIER_REPO = "cross-encoder/quora-distilroberta-base"  # trained on Quora "is this a duplicate question?"
VERIFIER_FILE = "onnx/model_quint8_avx2.onnx"  # 8-bit version: ~83MB instead of 329MB
VERIFIER_CACHE_DIR = ".cache/verifier"  # where the downloaded model is kept (git-ignored)
