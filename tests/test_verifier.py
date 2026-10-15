import asyncio

import numpy as np
import pytest

from services import verifier


class FakeTokenizer:
    def encode(self, a, b):
        class Encoded:
            ids = [0, 1, 2]
            attention_mask = [1, 1, 1]
        return Encoded()


class FakeSession:
    def __init__(self, logit):
        self.logit = logit

    def get_inputs(self):
        return [type("I", (), {"name": "input_ids"})(), type("I", (), {"name": "attention_mask"})()]

    def run(self, outputs, inputs):
        assert set(inputs) == {"input_ids", "attention_mask"}
        return [np.array([[self.logit]], dtype=np.float32)]


@pytest.mark.parametrize("logit, expected", [(0.0, 0.5), (6.0, 0.9975), (-6.0, 0.0025)])
def test_logit_becomes_probability(monkeypatch, logit, expected):
    monkeypatch.setattr(verifier, "_load", lambda: (FakeSession(logit), FakeTokenizer()))
    assert verifier.same_question_probability("a", "b") == pytest.approx(expected, abs=1e-3)


def test_async_wrapper_returns_the_same_score(monkeypatch):
    monkeypatch.setattr(verifier, "_load", lambda: (FakeSession(0.0), FakeTokenizer()))
    assert asyncio.run(verifier.same_question("a", "b")) == pytest.approx(0.5)
