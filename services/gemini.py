"""Gemini: the backup LLM, used only when Groq fails (see services/gateway.py).

Returns the same fields as services/llm.ask_llm, so the rest of the app doesn't care who answered.
Uses Google's official SDK (google-genai).
"""
import asyncio
import os
import time

from dotenv import load_dotenv

import config
from services.llm import LLMError

load_dotenv()

_client = None  # a google.genai Client, created on first use


def preload() -> None:
    """Import Google's SDK (about 0.6s on a fast laptop, much more on a small server). It is only needed when Groq fails, so
    the server does this in the background AFTER it is ready, instead of making every cold start wait for it."""
    try:
        import google.genai  # noqa: F401
    except Exception:
        pass  # not fatal: ask_gemini will try again when it is actually needed


def _get_client():
    # Create the client only when first needed, so importing this file (e.g. in tests) needs no key.
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=int(config.GEMINI_TIMEOUT_SECONDS * 1000)),  # milliseconds
        )
    return _client


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL") or config.GEMINI_MODEL


async def ask_gemini(question: str) -> dict:
    """Send one question to Gemini and return the answer plus usage info (same shape as ask_llm)."""
    from google.genai import errors, types

    model = get_gemini_model()
    start = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            _get_client().aio.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(max_output_tokens=config.MAX_OUTPUT_TOKENS),
            ),
            config.GEMINI_TIMEOUT_SECONDS,
        )
    except TimeoutError as e:
        raise LLMError(f"Gemini request timed out after {config.GEMINI_TIMEOUT_SECONDS}s", "gemini_timeout") from e
    except errors.APIError as e:  # bad key, rate limit (429), server errors...
        raise LLMError(f"Gemini request failed: {e.code} {e.message or ''}".strip(), f"gemini_error_{e.code}") from e
    except Exception as e:  # network trouble and anything else from the SDK
        raise LLMError(f"Gemini request failed: {e!r}", "gemini_error") from e
    latency_ms = (time.perf_counter() - start) * 1000

    candidate = response.candidates[0] if response.candidates else None
    finish = candidate.finish_reason.name if candidate and candidate.finish_reason else None
    answer = response.text or ""
    if not answer and finish != "MAX_TOKENS":  # e.g. blocked by a safety filter: nothing useful to return
        raise LLMError(f"Gemini returned no answer (finish reason: {finish})", "gemini_no_answer")

    usage = response.usage_metadata
    prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
    # Gemini's hidden "thinking" tokens are billed as output too, so they count as completion tokens.
    completion_tokens = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
    return {
        "answer": answer,
        "model": model,
        "provider": "gemini",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(latency_ms, 2),
        "truncated": finish == "MAX_TOKENS",  # cut off at the length limit
    }
