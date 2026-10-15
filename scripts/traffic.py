"""Send ~50 realistic requests to the RUNNING local server, so GET /stats has something to show.

Start the server first:   uvicorn api.main:app --reload
Then run:                 python -m scripts.traffic
Options:                  --delay 0.4   seconds between requests      --url http://127.0.0.1:8000

The key comes from DEV_API_KEY in .env (make one with scripts/create_key.py). Stops by itself if the
server rejects the key (401) or the daily token budget runs out (429).

The mix, in order (questions come from eval/question_pairs.csv):
  1. ~20 new questions        -> answered by the LLM (cache misses)
  2. 12 exact repeats         -> exact cache hits
  3. 12 paraphrases           -> semantic hits when they are close enough (some will miss, like real life)
  4.  6 repeats of those      -> exact hits again
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

PAIRS_FILE = Path(__file__).resolve().parent.parent / "eval" / "question_pairs.csv"
SUFFIX = " Answer in 2-3 sentences."  # keeps answers short, so this costs few tokens


def build_plan() -> list[tuple[str, str]]:
    """Returns [(phase, question), ...] in the order they will be sent."""
    rows = list(csv.DictReader(open(PAIRS_FILE, encoding="utf-8")))
    pairs = [(r["question_a"], r["question_b"]) for r in rows if r["category"] == "paraphrase"]
    used = {q for pair in pairs for q in pair}
    others = []
    for r in rows:
        for q in (r["question_a"], r["question_b"]):
            if r["category"] in ("related", "unrelated") and q not in used and q not in others:
                others.append(q)

    rng = random.Random(7)  # fixed seed: every run sends the same mix
    new = [a for a, _ in pairs] + others[:8]
    repeats = rng.sample(new, k=min(12, len(new)))
    paraphrases = [b for _, b in pairs][:12]
    paraphrase_repeats = paraphrases[:6]
    plan = ([("new", q) for q in new] + [("repeat", q) for q in repeats]
            + [("paraphrase", q) for q in paraphrases] + [("repeat of paraphrase", q) for q in paraphrase_repeats])
    return [(phase, q + SUFFIX) for phase, q in plan]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a realistic mix of requests to the local gateway.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--delay", type=float, default=0.4, help="seconds to wait between requests")
    args = parser.parse_args()

    key = os.getenv("DEV_API_KEY")
    if not key:
        sys.exit("DEV_API_KEY is missing in .env. Make a key with: python -m scripts.create_key --name dev --budget 100000")

    plan = build_plan()
    print(f"Sending {len(plan)} requests to {args.url}/chat\n")
    print(f"{'#':>3} {'phase':<21} {'result':<9} {'sim':>6} {'ms':>8}  question")

    results, latencies, stopped = [], {}, None
    last = {}
    with httpx.Client(timeout=90, headers={"Authorization": f"Bearer {key}"}) as client:
        for i, (phase, question) in enumerate(plan, 1):
            try:
                res = client.post(f"{args.url}/chat", json={"question": question})
            except httpx.TransportError as e:
                stopped = f"Could not reach the server ({e.__class__.__name__}). Is uvicorn running?"
                break
            if res.status_code == 401:
                stopped = "401: the server rejected DEV_API_KEY (wrong, or the key is disabled)."
                break
            if res.status_code == 429:
                stopped = f"429: daily token budget used up. {res.json().get('detail', '')}"
                break
            if res.status_code != 200:
                print(f"{i:>3} {phase:<21} ERROR {res.status_code}: {res.text[:80]}")
                results.append(("error", None))
                continue

            body = res.json()
            kind = body["cache_type"] or f"LLM/{body['provider']}"
            results.append((kind, body))
            latencies.setdefault(kind, []).append(body["latency_ms"])
            last = body
            sim = f"{body['similarity']:.3f}" if body.get("similarity") is not None else ""
            print(f"{i:>3} {phase:<21} {kind:<9} {sim:>6} {body['latency_ms']:>8.0f}  {question[:48]}")
            time.sleep(args.delay)

    print("\n=== summary ===")
    counts = Counter(kind for kind, _ in results)
    for kind, n in counts.most_common():
        avg = sum(latencies[kind]) / len(latencies[kind]) if kind in latencies else 0
        print(f"  {kind:<12} {n:>3} requests   average {avg:,.0f} ms")
    bodies = [b for _, b in results if b]
    print(f"  fallbacks to Gemini: {sum(1 for b in bodies if b.get('fallback_reason'))}"
          f"   cut-off answers: {sum(1 for b in bodies if b.get('truncated'))}")
    if last.get("daily_budget") is not None:
        print(f"  tokens used today: {last.get('tokens_used_today')} of {last['daily_budget']}")
    if stopped:
        print(f"\nSTOPPED EARLY: {stopped}")
    print("\nNow look at the totals: GET /stats in http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    main()
