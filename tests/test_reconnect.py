import asyncio

import pytest
from sqlalchemy.exc import OperationalError

import config
from services import cache, db


def test_db_retries_once_on_dropped_connection():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise OperationalError("SELECT 1", {}, Exception("server closed the connection unexpectedly"))
        return "ok"

    assert db.retry_once_if_dropped(flaky) == "ok"
    assert len(calls) == 2


def test_db_does_not_retry_forever():
    def always_broken():
        raise OperationalError("SELECT 1", {}, Exception("down"))

    with pytest.raises(OperationalError):
        db.retry_once_if_dropped(always_broken)


def test_redis_timeout_does_not_cancel_the_work(monkeypatch):
    """A slow reconnect must keep going after the request gave up, so the next request benefits."""
    monkeypatch.setattr(config, "REDIS_TIMEOUT_SECONDS", 0.05)
    finished = []

    async def slow_connect():
        await asyncio.sleep(0.2)
        finished.append(True)

    async def run():
        with pytest.raises(TimeoutError):
            await cache._within_timeout(slow_connect())
        assert finished == []  # we gave up waiting...
        await asyncio.sleep(0.3)
        assert finished == [True]  # ...but the work still completed

    asyncio.run(run())
