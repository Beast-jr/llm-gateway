"""Verifier: a small cross-encoder that reads two questions TOGETHER and says if they are the same question.

The embedding model turns each question into numbers separately, so "zoom in" and "zoom out" look alike.
This model sees both questions side by side, so it can notice the one word that changes the meaning.
It is slower than comparing embeddings, so we only run it on close candidates.
"""
import asyncio
import threading

import numpy as np

import config

_session = None
_tokenizer = None
_lock = threading.Lock()  # stops two first requests from loading the model twice


def _load():
    # Load once, on first use (downloads ~83MB the very first time). Imported here so tests never load it.
    global _session, _tokenizer
    with _lock:
        if _session is None:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            kwargs = {"repo_id": config.VERIFIER_REPO, "cache_dir": config.VERIFIER_CACHE_DIR}
            model_path = hf_hub_download(filename=config.VERIFIER_FILE, **kwargs)
            tokenizer = Tokenizer.from_file(hf_hub_download(filename="tokenizer.json", **kwargs))
            tokenizer.enable_truncation(max_length=128)
            tokenizer.enable_padding()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1  # small model; fewer threads = less memory and no fighting the server
            _tokenizer = tokenizer
            _session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    return _session, _tokenizer


def same_question_probability(question_a: str, question_b: str) -> float:
    """0.0 - 1.0: how likely the two questions ask the same thing. Sync (~tens of ms)."""
    session, tokenizer = _load()
    encoded = tokenizer.encode(question_a, question_b)
    inputs = {"input_ids": np.array([encoded.ids], dtype=np.int64),
              "attention_mask": np.array([encoded.attention_mask], dtype=np.int64)}
    wanted = {i.name for i in session.get_inputs()}
    logit = float(session.run(None, {k: v for k, v in inputs.items() if k in wanted})[0].reshape(-1)[0])
    return 1.0 / (1.0 + np.exp(-logit))  # logit -> probability


async def same_question(question_a: str, question_b: str) -> float:
    """Same as above, in a worker thread so the server keeps handling other requests meanwhile."""
    return await asyncio.to_thread(same_question_probability, question_a, question_b)


async def warm_up() -> None:
    """Load the model now (at server start) so the first real request isn't slow."""
    await asyncio.to_thread(_load)
