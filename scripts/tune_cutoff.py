"""Day 7: pick how the semantic cache decides "same question", using labelled question pairs.

Uses our real local models only (no Groq, no database, no server).
Run from the project root:  python -m scripts.tune_cutoff
Writes eval/cutoff_results.md (for the README) and prints the same numbers.

Three label sets (each pair says should_match yes/no):
  train    eval/question_pairs.csv  - used to build the word-list guard and look at verifier scores
  holdout  eval/holdout_pairs.csv   - written before the word-list guard
  holdout2 eval/holdout2_pairs.csv  - written before the verifier, never looked at while building it

Strategies compared:
  (a) embedding only                      reuse if embedding score >= cutoff
  (b) embedding + word-list guard         ... and the full guard (numbers, names, opposite words) agrees
  (c) embedding + verifier + numbers      embedding >= prefilter, numbers match, verifier probability >= threshold
  (d) like (c) but with the full guard    (for comparison only)
  (e) fallback: embedding 0.96 + numbers guard
"""
import csv
from pathlib import Path

import numpy as np

import config
from services import match_guard, verifier
from services.embeddings import embed_many

ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "train": ROOT / "eval" / "question_pairs.csv",
    "holdout": ROOT / "eval" / "holdout_pairs.csv",
    "holdout2": ROOT / "eval" / "holdout2_pairs.csv",
}
RESULTS_FILE = ROOT / "eval" / "cutoff_results.md"
CUTOFFS = [round(0.80 + i * 0.01, 2) for i in range(20)]  # 0.80 ... 0.99
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]  # verifier "same question" probability
PREFILTER = config.SEMANTIC_PREFILTER
FALLBACK_CUTOFF = 0.96


def load_scored_pairs(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    questions = sorted({q for r in rows for q in (r["question_a"], r["question_b"])})
    vectors = dict(zip(questions, np.array(embed_many(questions))))
    for r in rows:
        a, b = r["question_a"], r["question_b"]
        r["score"] = float(vectors[a] @ vectors[b])  # vectors are length 1, so dot product = cosine similarity
        r["match"] = r["should_match"] == "yes"
        r["guard"] = match_guard.check(a, b)  # full word-list guard: None = allows
        r["numbers"] = match_guard.check_numbers(a, b)  # numbers-only guard
        # The verifier only ever sees candidates above the prefilter; skipping it elsewhere mirrors production.
        r["prob"] = verifier.same_question_probability(a, b) if r["score"] >= PREFILTER else None
    return rows


# ---- strategies: each returns True when the cache WOULD reuse the saved answer ----
def hit_embedding(r, cutoff):
    return r["score"] >= cutoff


def hit_guard(r, cutoff):
    return r["score"] >= cutoff and r["guard"] is None


def hit_verifier(r, threshold):
    return r["prob"] is not None and r["numbers"] is None and r["prob"] >= threshold


def hit_verifier_full_guard(r, threshold):
    return r["prob"] is not None and r["guard"] is None and r["prob"] >= threshold


def hit_fallback(r, _):
    return r["score"] >= FALLBACK_CUTOFF and r["numbers"] is None


STRATEGIES = {  # key: (label, hit function, parameter values, parameter name)
    "a": ("embedding only", hit_embedding, CUTOFFS, "cutoff"),
    "b": ("embedding + word-list guard", hit_guard, CUTOFFS, "cutoff"),
    "c": (f"embedding ≥ {PREFILTER} + verifier + numbers guard", hit_verifier, THRESHOLDS, "threshold"),
    "d": (f"embedding ≥ {PREFILTER} + verifier + full word-list guard", hit_verifier_full_guard, THRESHOLDS, "threshold"),
    "e": (f"fallback: embedding {FALLBACK_CUTOFF} + numbers guard", hit_fallback, [None], ""),
}


# Measured on the dev laptop (Windows, 1 CPU thread for the verifier), 2026-10-15.
VERIFIER_COST = (
    f"`{config.VERIFIER_REPO}` as an 8-bit ONNX file: **83 MB** on disk (+3.6 MB tokenizer). "
    "RAM for the whole app process: **93 MB** with no models, **260 MB** with the bge-small embedder, "
    "**365 MB** with the verifier added (+105 MB), which leaves little room on Render's 512 MB free tier. "
    "Speed: **about 13 ms per pair** (max 29 ms), only for candidates above the prefilter, plus about 1.8 s to load at startup."
)

CAVEATS = "\n".join([
    "- **Nothing reached zero wrong answers on all three sets**, with or without the guard or the verifier. "
    "The hardest pairs swap two things around (kilometers to miles vs miles to kilometers, 5 dollars in rupees vs "
    "5 rupees in dollars) or change singular to plural; the embedding scores them 0.99+ and even the verifier is fooled by some.",
    "- **The verifier helps but is not enough.** It rejects many danger pairs the other approaches accept "
    "(Celsius/Fahrenheit, km/miles, enable/disable), but it calls some opposites the same question "
    "(mute/unmute 0.99, zoom in/out 0.98, pros/cons 0.97) and rejects many real paraphrases "
    "(\"What is RAG?\" vs \"Explain RAG to me\" scores 0.002). It also costs 105 MB of RAM.",
    "- **These sets are deliberately adversarial**: more than half the pairs are danger pairs built to fool the cache. "
    "Real users ask far fewer one-word-different questions, so real-traffic wrong-answer rates should be much lower "
    "than these counts. Check the `similarity` column in the `calls` table on live traffic.",
    "- **One label was corrected after the first holdout2 run**: \"before eating or after eating\" vs "
    "\"after eating or before eating\" is the same question in a different word order, so it was changed from *no* to *yes* "
    "(my labelling mistake). Every other label is unchanged.",
    "- **Small sets**: 47, 25 and 38 pairs. A difference of one or two pairs is noise.",
])


def count(rows, fn, param) -> dict:
    """reused = true matches answered from cache; wrong = another question's answer shown; extra = missed matches."""
    hits = [fn(r, param) for r in rows]
    return {
        "reused": sum(h and r["match"] for h, r in zip(hits, rows)),
        "wrong": sum(h and not r["match"] for h, r in zip(hits, rows)),
        "extra": sum((not h) and r["match"] for h, r in zip(hits, rows)),
    }


def best_setting(sets, key):
    """Zero wrong answers on EVERY set, then the most reuse. Ties go to the safer (higher) parameter."""
    _, fn, params, _ = STRATEGIES[key]
    best = None
    for p in params:
        per_set = {n: count(rows, fn, p) for n, rows in sets.items()}
        if any(c["wrong"] for c in per_set.values()):
            continue
        reused = sum(c["reused"] for c in per_set.values())
        if best is None or reused >= best[0]:
            best = (reused, p, per_set)
    return best  # (total reused, parameter, per-set counts) or None


def fmt(c, total_yes):
    return f"{c['reused']}/{total_yes} · {c['wrong']} · {c['extra']}"


def main() -> None:
    print("loading models and scoring pairs...")
    sets = {name: load_scored_pairs(path) for name, path in SETS.items()}
    yes = {n: sum(r["match"] for r in rows) for n, rows in sets.items()}
    for n, rows in sets.items():
        print(f"  {n}: {len(rows)} pairs ({yes[n]} should match, {len(rows) - yes[n]} should not)")

    for key, (label, fn, params, pname) in STRATEGIES.items():
        print(f"\n=== ({key}) {label} === reused / WRONG / extra")
        print(f"{pname or '':>10} | " + " | ".join(f"{n:^16}" for n in sets))
        for p in params:
            counts = {n: count(rows, fn, p) for n, rows in sets.items()}
            cells = " | ".join(f"{c['reused']:>5}/{yes[n]:<2} {c['wrong']:>3} {c['extra']:>3}" for n, c in counts.items())
            print(f"{'' if p is None else p:>10} | {cells}")

    best = {k: best_setting(sets, k) for k in STRATEGIES}
    print("\nBest setting per strategy (zero wrong on all three sets, then most reuse):")
    for k, b in best.items():
        label = STRATEGIES[k][0]
        print(f"  ({k}) {label}: " + ("NONE has zero wrong answers on all three sets" if b is None
              else f"{STRATEGIES[k][3] or 'fixed'} {b[1] if b[1] is not None else ''} -> {b[0]} true matches reused"))

    print("\nVerifier probability for pairs that must NOT match (embedding >= prefilter), highest first:")
    for n, rows in sets.items():
        for r in sorted((r for r in rows if not r["match"] and r["prob"] is not None), key=lambda r: -r["prob"]):
            print(f"  [{n}] prob {r['prob']:.3f}  emb {r['score']:.3f}  numbers:{r['numbers'] or '-':<18} "
                  f"{r['question_a']!r} vs {r['question_b']!r}")
    print("\nTrue matches not reused at the best (c) threshold:")
    thr = best["c"][1] if best["c"] else 0.9
    for n, rows in sets.items():
        for r in rows:
            if r["match"] and not hit_verifier(r, thr):
                why = "below prefilter" if r["prob"] is None else f"prob {r['prob']:.3f}" if r["numbers"] is None else "numbers guard"
                print(f"  [{n}] emb {r['score']:.3f}  {why:<18} {r['question_a']!r} vs {r['question_b']!r}")

    write_markdown(sets, yes, best)
    print(f"\nWrote {RESULTS_FILE.relative_to(ROOT)}")


def write_markdown(sets, yes, best) -> None:
    names = list(sets)
    n_pairs = {n: len(rows) for n, rows in sets.items()}

    def row(label, key, p):
        _, fn, _, _ = STRATEGIES[key]
        return f"| {label} | " + " | ".join(fmt(count(sets[n], fn, p), yes[n]) for n in names) + " |"

    # The settings worth comparing side by side (not "best safe", because none is safe on all three sets).
    final_rows = [
        ("(a) score only: embedding cutoff 0.85", "a", 0.85),
        ("(a) score only: embedding cutoff 0.90", "a", 0.90),
        ("(a) score only: embedding cutoff 0.96", "a", 0.96),
        ("(a) score only: embedding cutoff 0.99", "a", 0.99),
        ("(b) + word-list guard, cutoff 0.96", "b", 0.96),
        (f"(c) + verifier (embedding ≥ {PREFILTER}) + numbers guard, threshold 0.97", "c", 0.97),
        ("**(e) SHIPPED: embedding cutoff 0.96 + numbers guard**", "e", None),
    ]
    zero_safe = [k for k in "abcde" if best[k]]
    lines = [
        "# Semantic cache: how we decide two questions are the same",
        "",
        "Cells show **reused / wrong answers / extra Groq calls**. *Reused* = a true match answered from cache (good). "
        "*Wrong* = the user would be shown another question's answer (very bad). "
        "*Extra* = a true match we missed, which only costs one more Groq call (cheap).",
        "",
        "Pair sets: " + "; ".join(f"**{n}** = `{SETS[n].relative_to(ROOT).as_posix()}` ({n_pairs[n]} pairs, {yes[n]} true matches)"
                                  for n in names) + ". More than half of the pairs are deliberately tricky danger pairs, "
        "so wrong-answer counts here are much higher than real traffic should produce.",
        "",
        "## The story, in order",
        "",
        "1. **Score only.** The embedding model scores how close two questions are in meaning. But it treats a "
        "one-word change (Python vs JavaScript, open vs close, 2014 vs 2018) as almost the same question, "
        "so only a very high cutoff is safe, and that throws away most true matches. On the first set, 0.96 was the lowest "
        "cutoff with no wrong answers, reusing only 5 of 17 true matches.",
        "2. **Word-list guard: overfit, caught by the holdout.** A list of names, numbers and opposite words fixed every "
        "danger pair it was built from (the train set became safe from 0.81 upward). But the holdout sets, written before "
        "the guard, had pairs it had no words for (zoom in/out 0.97, speed up/slow down, mute/unmute), and it still gave wrong answers. "
        "A word list only knows its own words, so only the numbers/years part (which did generalise) was kept.",
        f"3. **Cross-encoder verifier: did not clearly win, and costs RAM.** A small model (`{config.VERIFIER_REPO}`, trained on "
        f"Quora duplicate questions) reads both questions together. It rejected many danger pairs the others accepted, but it "
        f"still called some opposites the same question (mute/unmute 0.99), rejected many real paraphrases, never reached zero wrong "
        f"answers, and adds 105 MB of RAM (365 MB total vs 260 MB). It is used only for evaluation, never loaded by the server.",
        "4. **Shipped: cutoff 0.96 + numbers guard, with mitigations for the wrong answers that remain.** "
        "(a) The `/chat` response now includes `matched_question`, the saved question whose answer was reused, so a caller or "
        "user can see what was matched. (b) Callers can send `\"semantic_cache\": false` to skip the semantic step "
        "(the exact cache still applies). (c) Every call, including near-misses, logs its `similarity`, so real traffic can be audited.",
        "",
        "## Final table (train / holdout / holdout2)",
        "",
        "| approach | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        *[row(label, key, p) for label, key, p in final_rows],
        "",
        "## Decision",
        "",
    ]
    if zero_safe:
        lines.append("Some setting reached zero wrong answers on all three sets: " + ", ".join(f"({k})" for k in zero_safe) + ".")
    else:
        lines.append("**No approach reached zero wrong answers on all three sets** (score only, word-list guard, "
                     "verifier, or the shipped setting). The verifier did not clearly beat the simpler options, so we ship "
                     f"embedding cutoff {FALLBACK_CUTOFF} + numbers guard, plus the mitigations above. "
                     "A cutoff of 0.99 is safer but reuses almost nothing.")
    lines += ["", "## Verifier cost", "", VERIFIER_COST, "",
              "## Honest caveats", "", CAVEATS, "",
              f"## Pairs that must NOT match, with their scores (embedding ≥ {PREFILTER})", "",
              "| set | embedding | verifier prob | numbers guard | word-list guard | pair |", "|---|---|---|---|---|---|"]
    for n, rows in sets.items():
        for r in sorted((r for r in rows if not r["match"] and r["prob"] is not None), key=lambda r: -r["score"]):
            lines.append(f"| {n} | {r['score']:.3f} | {r['prob']:.3f} | {r['numbers'] or '—'} | {r['guard'] or '**missed**'} | "
                         f"{r['question_a']} ↔ {r['question_b']} |")
    thr = best["c"][1] if best["c"] else 0.9
    lines += ["", f"## True matches not reused by the verifier setting (threshold {thr}); cost: an extra Groq call", "",
              "| set | embedding | verifier prob | pair |", "|---|---|---|---|"]
    for n, rows in sets.items():
        for r in sorted((r for r in rows if r["match"] and not hit_verifier(r, thr)), key=lambda r: r["score"]):
            prob = "(not run: below prefilter)" if r["prob"] is None else f"{r['prob']:.3f}"
            lines.append(f"| {n} | {r['score']:.3f} | {prob} | {r['question_a']} ↔ {r['question_b']} |")
    lines.append("")
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
