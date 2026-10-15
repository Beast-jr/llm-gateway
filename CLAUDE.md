# LLM Gateway — project memory

Owner: Jahnavi, final-year CS (AI/ML) student. Portfolio project for placements.
Learning as I go: explain things in simple, non-technical language. Keep code small
and readable, with short comments.

## Goal
A FastAPI service that sits between apps and LLM APIs. Apps call the gateway instead
of calling Groq directly. The gateway adds:
- **Exact cache (Redis):** the same question again → instant saved answer
- **Semantic cache (pgvector + sentence-transformers):** a *similar* question
  ("what is RAG?" vs "explain RAG to me") → reuse the saved answer. **Main selling point.**
- **API keys + per-user daily token budgets** (401 / 429 errors)
- **Fallback:** if Groq fails or takes >5s, retry with Gemini
- **/stats:** cache hit rate, money saved, latency with vs without cache
- Tests + GitHub Actions CI, deployed on Render, load-tested with Locust

Final goal: a live URL and a README headline like
"semantic cache cut LLM calls by X% and response time from 1.8s to 20ms."

## Stack
FastAPI · Postgres (Neon) + pgvector · Redis (Upstash) · Groq (main) + Gemini (fallback)
· sentence-transformers · pytest · Locust · GitHub Actions · Render

## 14-day plan (each phase ends with a checkpoint test)
- Day 1 – Setup: repo, folders, .env, GET /health
- Day 2 – POST /chat sends question to Groq, returns answer
- Day 3 – Connect Neon, `calls` table via Alembic, log every call (question, model, tokens, latency, cost)
- Day 4 – Exact cache in Redis (hash question, TTL), add cache_hit column
- Days 5–7 – Semantic cache: pgvector + embeddings, similarity cutoff, tune on 20 question pairs (0.85/0.90/0.95)
- Days 8–9 – users table with API keys (401), daily token budget in Redis (429)
- Day 10 – Fallback routing Groq → Gemini, log which provider answered
- Day 11 – GET /stats
- Day 12 – pytest (mocked LLM calls) + GitHub Actions, 80%+ coverage
- Days 13–14 – Deploy to Render, Locust benchmark, README

## Environment
- Windows 11, PowerShell, VS Code. Project at `C:\llm-gateway`, Python 3.13 venv at `.venv`
- GitHub: github.com/Beast-jr/llm-gateway (branch: main)
- Folders: `api/` (routes), `services/` (business logic, e.g. LLM calls), `models/` (Pydantic schemas), `tests/`
- `.env` keys: GROQ_API_KEY, GEMINI_API_KEY, GROQ_MODEL (=openai/gpt-oss-20b), DATABASE_URL, REDIS_URL
- Run server: `.venv\Scripts\activate` then `uvicorn api.main:app --reload` → http://127.0.0.1:8000/docs
- Run tests: `python -m pytest -v` (tests use in-memory SQLite via tests/conftest.py, never Neon)
- DB migrations: `alembic upgrade head` (reads DATABASE_URL from .env). New column → edit models/, then
  `alembic revision --autogenerate -m "..."`, review the file, `alembic upgrade head`.
- DB driver: SQLAlchemy 2.x + psycopg v3 (sync). Chosen because it accepts Neon's `?sslmode=require`
  as-is (asyncpg rejects it) and avoids psycopg-async's Windows event-loop issue.
- Prices for cost_usd live in `config.py` (PRICES_PER_MILLION), plus cache TTL and Redis timeout.
- Request flow: route (api/main.py) → `services/gateway.answer_question()` (cache → LLM) → background `save_call()`.
  New steps (semantic cache, Gemini fallback) go in gateway.py. Tests patch `services.gateway.ask_llm`.
- Redis/DB failures must never break /chat: log a warning and carry on.
- Embeddings: fastembed (ONNX, no PyTorch, for Render 512MB). Model loaded lazily once; embed() via asyncio.to_thread;
  vectors normalized. Tests use an autouse fake embedder (tests/conftest.py), never the real model.

## Rules
- **NEVER print, read aloud, or commit the contents of `.env`.**
- Do only the current day's task. Don't jump ahead.
- Keep logic out of routes: LLM calls live in `services/`, schemas in `models/`.
- Tests must mock LLM calls (no real API calls, no credits spent).
- Don't `git commit`; Jahnavi commits. Just list which files changed.
- End each session with a 3-line plain-English summary.
- `requirements.txt` must be UTF-8 (PowerShell `pip freeze > file` writes UTF-16; use
  `pip freeze | Out-File -Encoding utf8 requirements.txt` instead).

## Progress
- [x] Day 1 – Setup + GET /health
- [x] Day 2 – POST /chat → Groq (services/llm.py, models/schemas.py, tests/test_chat.py) — committed, works in /docs
- [x] Day 3 – Neon + Alembic + call logging (psycopg v3 sync driver, `calls` table, cost from config.py, saved in a background task) — committed, verified 5 rows in Neon
- [x] Day 4 – Exact cache (Redis): services/cache.py (SHA-256 of normalized question + model, TTL in config.py),
  flow in services/gateway.py, cache_hit column (migration 0002), fakeredis in tests — committed, verified 1954ms miss → 22ms hit
- [x] Day 5 – Semantic cache part 1: services/embeddings.py (fastembed, BAAI/bge-small-en-v1.5, 384 dims),
  `semantic_cache` table + HNSW cosine index (migration 0003), saved in background on a miss
  (gateway.remember_answer), scripts/similarity_check.py
  Days 4–5 committed; semantic_cache has 3 rows in Neon.
- [x] Day 6 – Semantic cache part 2: lookup by similarity — committed. exact → semantic → Groq in gateway.py,
  SEMANTIC_CUTOFF + 1s time limit in config.py, cache_type/similarity columns (migration 0004), startup warm-up
  (model+DB+Redis), reconnect/retry logic, scripts/semantic_demo.py
- [x] Day 7 – Semantic cache part 3: cutoff tuning. FINAL DECISION: SEMANTIC_CUTOFF 0.96 + numbers-only guard
  (services/match_guard.check_numbers). Tried score-only → word-list guard (overfit, caught by holdout) → Quora cross-encoder
  verifier (services/verifier.py: eval only, NOT loaded by the server, +105MB RAM) — none reached zero wrong answers, verifier
  didn't clearly win. Mitigations: /chat returns `matched_question`; request field `semantic_cache: false` opts out of the
  semantic step; similarity logged on every call (incl. near-misses); near-duplicates (>=0.99) not re-saved.
  Evidence: eval/cutoff_results.md, eval/question_pairs.csv + holdout_pairs.csv + holdout2_pairs.csv, scripts/tune_cutoff.py.
  Don't tune the cutoff further; check calls.similarity on real traffic after deploy.
- [x] Day 8 – API keys — committed, verified (401 without key, 200 with key). services/auth.py, users table (migration 0005),
  per-user cache scope via CACHE_SCOPE, user_id on calls + semantic_cache, 60s in-memory key cache, scripts/create_key.py
- [ ] Day 9 – Daily token budgets (IN PROGRESS: services/budget.py, Redis counter budget:{user_id}:{UTC date} with 48h TTL, 429 + Retry-After
  before Groq, cache hits free, response has tokens_used_today + daily_budget, fail OPEN if Redis is down)
- [ ] Day 10 – Groq → Gemini fallback
- [ ] Day 11 – GET /stats
- [ ] Day 12 – Tests + CI (80%+ coverage)
- [ ] Days 13–14 – Deploy, Locust, README

Notes / numbers to remember:
- Day 9 fixes: (1) budget counter read/write use BUDGET_TIMEOUT_SECONDS (4s), longer than the cache's 0.5s, because a slow read must not
  silently skip the check; free cache hits don't wait for the counter (tokens_used_today may be null there). Still fail OPEN if Redis is down.
  (2) Groq answers cut off by the length limit (finish_reason "length") come back with truncated=true and are NEVER saved to Redis or
  semantic_cache (still logged and still counted in the budget). MAX_OUTPUT_TOKENS=4096 (was Groq's default ~2048; gpt-oss's hidden
  reasoning tokens count too). Truncated rows already saved before this fix: semantic_cache rows with completion_tokens >= 2048.
- Day 9 budgets: checked BEFORE calling Groq, counted AFTER (we only know the tokens afterwards), so one request (or a few at the same
  moment) can overshoot a little. Cache hits cost 0 tokens: never blocked, never counted. Budget 0 = cache-only user. FAIL OPEN: if Redis
  is down/slow we allow the request (log a warning) because blocking everyone whenever the counter is down is worse than some unmetered
  calls, and Groq has its own rate limits. Budget changes in the users table take up to 60s (auth cache). Time comes from budget.utcnow().
- Day 8 auth: services/auth.py. `Authorization: Bearer gw_...`; only the SHA-256 hash is stored (users table, migration 0005 also adds
  user_id to calls + semantic_cache). Key→user lookups cached in memory 60s (AUTH_CACHE_SECONDS), so deactivating a key can take up to 60s.
  CACHE_SCOPE="per_user" puts user id in the Redis key and filters the semantic lookup; old rows without user_id never match.
  Tests: autouse fixture in conftest fakes a logged-in user #1; tests/test_auth.py removes it to test real login.
  Make a key: `python -m scripts.create_key --name test --budget 500`. DB down during login → 503 (never a free pass).
  Known limit: pgvector HNSW + WHERE user_id=... filters AFTER the index search, so with many users a user's own near rows can be
  missed (look at hnsw.iterative_scan / per-user partial indexes if it ever matters).
- Day 7 outcome (3 pair sets: train 47, holdout 25, holdout2 38): no approach got zero wrong answers. Shipped 0.96 + numbers guard:
  train 5/17 reused · 0 wrong, holdout 3/12 · 1, holdout2 6/16 · 5 (reused/wrong). App RAM ~260MB (365MB if verifier were loaded).
  Residual risk: swapped-order / unit / plural pairs score 0.99+. Our sets are adversarial; real traffic should be far better.
- Day 6 latency: from India, Neon (us-east-2) and Upstash are each ~250ms per round trip; a NEW connection ~3s.
  Fixes: warm-up of model+DB+Redis at startup, no pool_pre_ping, AUTOCOMMIT read for the lookup (DB lookup ~990 → ~260ms).
  Locally an exact hit ≈ 270ms, a semantic hit ≈ 540ms: that is distance, not code. Take README numbers from the DEPLOYED
  service (Render Ohio, next to Neon). A cancelled (timed-out) Redis connect never finishes, so cold Redis = no cache.
- Day 4: exact cache 1954ms miss → 22ms hit.
- Day 5 similarity (bge-small): paraphrases 0.91–0.95; related-but-different 0.47–0.78; unrelated 0.32–0.52.
  Acronym vs full name ("RAG" vs "retrieval augmented generation") only 0.50 — the model doesn't know the acronym.
  Embedding ~9ms/question on laptop; model load ~25s first time (incl. download), cached in .cache/.
