"""Build step for Render: download the embedding model into EMBEDDING_CACHE_DIR.

Run from the project root:  python -m scripts.download_model

The running server then finds the model on disk and never downloads anything on a cold start.
It uses the same code as the server, so the files land exactly where the server will look.
"""
import time

import config
from services.embeddings import embed_many


def main() -> None:
    print(f"Downloading {config.EMBEDDING_MODEL} into {config.EMBEDDING_CACHE_DIR} ...")
    started = time.perf_counter()
    vectors = embed_many(["build-time check"])  # loading the model downloads it; embedding proves it works
    if len(vectors[0]) != config.EMBEDDING_DIM:
        raise SystemExit(f"Wrong vector size {len(vectors[0])}, expected {config.EMBEDDING_DIM}")
    print(f"OK: model ready, {len(vectors[0])}-number vectors, took {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
