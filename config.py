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
