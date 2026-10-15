import os
import time

import httpx
from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, APITimeoutError, AsyncGroq, GroqError

import config

load_dotenv()  # read keys from .env into environment variables

_client: AsyncGroq | None = None


class LLMError(Exception):
    """Raised when an LLM provider fails. `reason` is a short code such as "groq_timeout" or "groq_error_401"."""

    def __init__(self, message: str, reason: str = "error"):
        super().__init__(message)
        self.reason = reason


def _groq_reason(e: GroqError) -> str:
    if isinstance(e, APITimeoutError):  # (a kind of connection error, so check it first)
        return "groq_timeout"
    if isinstance(e, APIConnectionError):
        return "groq_connection_error"
    if isinstance(e, APIStatusError):  # the server answered with an error: 401, 429, 5xx...
        return f"groq_error_{e.status_code}"
    return "groq_error"


def _get_client() -> AsyncGroq:
    # Create the Groq client only when first needed, so importing this file
    # (e.g. in tests) doesn't require a real API key.
    global _client
    if _client is None:
        _client = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=httpx.Timeout(config.GROQ_TOTAL_TIMEOUT_SECONDS, connect=config.GROQ_CONNECT_TIMEOUT_SECONDS),
            max_retries=0,  # no hidden retries: if Groq fails we want to switch to Gemini right away
        )
    return _client


def get_model() -> str:
    """The Groq model we ask (also used in the cache key)."""
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


async def ask_llm(question: str) -> dict:
    """Send one question to Groq and return the answer plus usage info."""
    model = get_model()
    start = time.perf_counter()
    try:
        response = await _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_completion_tokens=config.MAX_OUTPUT_TOKENS,
        )
    except GroqError as e:  # covers bad key, network errors, rate limits, etc.
        raise LLMError(f"Groq request failed: {e}", _groq_reason(e)) from e
    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage
    # "length" means the model hit the token limit and the answer stops mid-sentence.
    truncated = response.choices[0].finish_reason == "length"
    return {
        "answer": response.choices[0].message.content or "",
        "model": response.model,
        "provider": "groq",  # Day 10 adds "gemini" as a fallback
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "latency_ms": round(latency_ms, 2),
        "truncated": truncated,
    }
