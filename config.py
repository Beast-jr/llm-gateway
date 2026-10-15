import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # so values in .env (like EMBEDDING_CACHE_DIR) are visible here too

BASE_DIR = Path(__file__).resolve().parent  # the project folder

# Prices in US dollars per 1 MILLION tokens, per model.
# Source: third-party price trackers (Oct 2026). Double-check https://groq.com/pricing
# and edit these numbers if Groq changes them.
PRICES_PER_MILLION = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    # PLACEHOLDER: unverified list price for the default Gemini model. Check https://ai.google.dev/pricing
    # and fix it (the free tier costs nothing, but this shows what the paid tier would charge).
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

# LLM answers
# Groq is tried first. Gemini answers only when Groq fails (error, can't connect, or too slow).
GROQ_CONNECT_TIMEOUT_SECONDS = 3  # can't even connect to Groq within this -> switch to Gemini
GROQ_TOTAL_TIMEOUT_SECONDS = 25  # generous: a healthy 7s answer is fine; only a hung request is cut off
GEMINI_TIMEOUT_SECONDS = 25
# Default Gemini model. UNVERIFIED guess: run `python -m scripts.list_gemini_models`, then set GEMINI_MODEL in .env.
GEMINI_MODEL = "gemini-2.5-flash-lite"
# Circuit breaker: after this many Groq failures in a row, skip Groq for the cooldown and use Gemini directly.
BREAKER_FAILURES = 3
BREAKER_COOLDOWN_SECONDS = 60
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
EMBEDDING_THREADS = 1  # one CPU thread: for a model this small it is faster (~40ms vs ~90ms) and kinder to a small server
EMBEDDING_DIM = 384  # bge-small makes vectors of 384 numbers; must match the DB column
# Where the downloaded embedding model is kept (git-ignored). On Render the model is downloaded into this folder during
# the BUILD (scripts/download_model.py), so the running server never has to download it. Override with the env var.
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR") or str(BASE_DIR / ".cache" / "fastembed")
# Startup warm-up (model + database + Redis) may take at most this long per step. After that the server starts anyway,
# so a slow or unreachable service can never stop it from opening its port. (Render free tier has only 0.1 CPU.)
STARTUP_WARMUP_TIMEOUT_SECONDS = 60

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
