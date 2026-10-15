"""Day 6 checkpoint: send a few questions to the RUNNING local server and show how each was answered.

Start the server first:  uvicorn api.main:app --reload
Then run:                python scripts/semantic_demo.py
"""
import httpx

URL = "http://127.0.0.1:8000/chat"

# Run order matters: the first question is answered by Groq, the paraphrases should be semantic hits,
# the repeat should be an exact hit, and the unrelated one should miss.
QUESTIONS = [
    "What is RAG?",
    "Explain RAG to me",                   # paraphrase -> expect semantic hit
    "Explain RAG to me",                   # same wording again -> expect exact hit
    "How do I reverse a list in Python?",
    "What's the way to reverse a Python list?",  # paraphrase -> expect semantic hit
    "What is the capital of Japan?",       # new topic -> expect miss
]


def main() -> None:
    print(f"{'question':45} {'cache_type':10} {'similarity':10} {'latency_ms':>10}")
    print("-" * 80)
    with httpx.Client(timeout=60) as client:
        for q in QUESTIONS:
            res = client.post(URL, json={"question": q})
            if res.status_code != 200:
                print(f"{q[:45]:45} ERROR {res.status_code}: {res.text[:60]}")
                continue
            body = res.json()
            print(f"{q[:45]:45} {str(body['cache_type']):10} {str(body['similarity']):10} {body['latency_ms']:>10}")


if __name__ == "__main__":
    main()
