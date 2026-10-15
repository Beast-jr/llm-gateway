# LLM Gateway

[![CI](https://github.com/Beast-jr/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Beast-jr/llm-gateway/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

A FastAPI service that sits between apps and LLM APIs (Groq first, Gemini as a fallback) and adds:
an exact cache (Redis), a semantic cache (pgvector), API keys with daily token budgets, a circuit-breaker
fallback, and a `/stats` endpoint showing cache hit rate, money saved and speed-up.

> The full README (architecture, benchmark numbers, how to deploy) comes on Day 14.

## Run the tests

```
pip install -r requirements.txt
python -m pytest --cov --cov-fail-under=80
```

The tests need no `.env`, database, Redis or API keys: everything external is faked.

**About the badges.** The CI badge is live (green only when the tests pass and coverage is at least 80%).
The coverage badge is typed in by hand (97% when it was last updated), so refresh it when coverage changes a lot.
The exact current numbers are on every CI run's summary page.
