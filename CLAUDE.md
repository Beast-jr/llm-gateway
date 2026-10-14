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
- Prices for cost_usd live in `config.py` (PRICES_PER_MILLION).

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
- [x] Day 3 – Neon + Alembic + call logging (psycopg v3 sync driver, `calls` table, cost from config.py, saved in a background task)
- [ ] Day 4 – Exact cache (Redis)
- [ ] Days 5–7 – Semantic cache
- [ ] Days 8–9 – API keys + token budgets
- [ ] Day 10 – Groq → Gemini fallback
- [ ] Day 11 – GET /stats
- [ ] Day 12 – Tests + CI (80%+ coverage)
- [ ] Days 13–14 – Deploy, Locust, README

Notes / numbers to remember:
-
