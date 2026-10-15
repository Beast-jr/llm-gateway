"""Print how similar pairs of questions are, to build intuition for Day 7's cutoff.

Run from the project root:  python -m scripts.similarity_check
"""
import time

import numpy as np

import config
from services.embeddings import embed_many

# (question A, question B, what we expect)
PAIRS = [
    ("what is RAG?", "explain RAG to me", "same meaning"),
    ("what is RAG?", "What is retrieval augmented generation?", "same meaning"),
    ("how do I reverse a list in python?", "python reverse list", "same meaning"),
    ("what is the capital of France?", "which city is France's capital?", "same meaning"),
    ("what is RAG?", "how do I build a RAG pipeline?", "related, different question"),
    ("what is RAG?", "what is fine-tuning?", "related, different question"),
    ("how do I reverse a list in python?", "how do I sort a list in python?", "related, different question"),
    ("what is RAG?", "what is Docker?", "unrelated"),
    ("what is the capital of France?", "how do I reverse a list in python?", "unrelated"),
]


def main():
    print(f"Model: {config.EMBEDDING_MODEL}  (first run downloads it, then it's cached)\n")
    start = time.perf_counter()
    embed_many(["warm-up"])  # load the model once so the timing below is fair
    print(f"Model loaded in {time.perf_counter() - start:.1f}s\n")

    print(f"{'similarity':>10}  {'expected':<28} pair")
    for a, b, expected in PAIRS:
        va, vb = np.array(embed_many([a, b]))
        similarity = float(va @ vb)  # vectors are normalized, so dot product = cosine similarity
        print(f"{similarity:>10.3f}  {expected:<28} {a!r} vs {b!r}")

    start = time.perf_counter()
    embed_many(["how long does one embedding take?"])
    print(f"\nOne embedding takes ~{(time.perf_counter() - start) * 1000:.0f} ms on this machine.")


if __name__ == "__main__":
    main()
