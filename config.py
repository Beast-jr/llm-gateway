# Prices in US dollars per 1 MILLION tokens, per model.
# Source: third-party price trackers (Oct 2026). Double-check https://groq.com/pricing
# and edit these numbers if Groq changes them.
PRICES_PER_MILLION = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}

# LLM answers
MAX_OUTPUT_TOKENS = 4096  # longest answer we allow. You only pay for tokens actually produced; this is a ceiling.
# (gpt-oss is a "thinking" model: its hidden reasoning tokens count too, so 2048 was cutting real answers off.)

# Exact cache (Redis)
CACHE_TTL_SECONDS = 24 * 60 * 60  # how long a saved answer is kept: 24 hours
REDIS_TIMEOUT_SECONDS = 0.5  # if Redis takes longer than this, skip the cache and ask Groq
# The budget counter gets longer: skipping it would let a user past their limit, and it only delays
# requests that are about to wait seconds for Groq anyway. A brand-new Redis connection takes ~3s from India.
BUDGET_TIMEOUT_SECONDS = 4.0

# API keys
AUTH_CACHE_SECONDS = 60  # remember a key->user lookup in memory this long (saves a DB query per request)

# Cache privacy: "per_user" = one user's saved answers are never shown to another user.
# "shared" = everyone shares one cache (cheaper, but only OK if answers are not private).
CACHE_SCOPE = "per_user"

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
