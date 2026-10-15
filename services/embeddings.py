import asyncio
import threading

import numpy as np

import config

_model = None
_lock = threading.Lock()  # stops two first requests from loading the model twice


def _get_model():
    # Load the model once, on first use (takes a few seconds; downloads ~67MB the very first time).
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding  # imported here so tests never load it

            _model = TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=config.EMBEDDING_CACHE_DIR)
    return _model


def embed_many(texts: list[str]) -> list[list[float]]:
    """Turn texts into vectors of EMBEDDING_DIM numbers, each scaled to length 1.

    Length-1 ("normalized") vectors mean cosine similarity is just a dot product.
    """
    vectors = np.array(list(_get_model().embed(texts)))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.tolist()


async def embed(text: str) -> list[float]:
    """Embed one text in a worker thread, so the server keeps handling other requests meanwhile."""
    vectors = await asyncio.to_thread(embed_many, [text])
    return vectors[0]
