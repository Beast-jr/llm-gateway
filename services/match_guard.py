"""Safety check for semantic matches.

IN PRODUCTION only `check_numbers` (numbers/years) is used. The full word-list `check` below was
evaluated (eval/cutoff_results.md): it fixed every pair it was built from but missed new kinds of
danger pairs, so it is kept only as a comparison in scripts/tune_cutoff.py.

The embedding model thinks "reverse a list in Python" and "reverse a list in JavaScript" are almost
the same question. This guard compares the two questions' *specifics* and says "don't reuse that
answer" when they differ. It is only a small word list + regex, no model, so it is fast and readable.
"""
import re

# Words that name one specific thing. If one question has one and the other doesn't, they differ.
# (Capitalised words in the middle of a sentence, like "Peru", are caught separately below.)
NAMED_TERMS = {
    # programming languages (not "go" or "r": too common as normal words)
    "python", "javascript", "java", "typescript", "rust", "ruby", "php", "kotlin", "swift", "sql",
    "c++", "c#", "golang", "perl", "html", "css",
    # operating systems / platforms
    "windows", "linux", "ubuntu", "macos", "mac", "android", "ios", "debian", "fedora",
    # products and tools
    "docker", "kubernetes", "react", "angular", "vue", "django", "flask", "node", "nodejs", "mysql",
    "postgres", "mongodb", "redis", "aws", "azure", "gcp", "git", "github", "excel", "chrome", "firefox",
    # countries and cities
    "india", "france", "germany", "japan", "china", "brazil", "usa", "uk", "canada", "australia",
    "russia", "italy", "spain", "mumbai", "delhi", "london", "paris", "tokyo", "berlin",
}

# Pairs of opposite actions. A question with one must not reuse the answer of a question with the other.
OPPOSITES = [
    ("open", "close"), ("add", "remove"), ("enable", "disable"), ("install", "uninstall"),
    ("reverse", "sort"), ("encrypt", "decrypt"), ("stage", "unstage"), ("start", "stop"),
    ("create", "delete"), ("increase", "decrease"), ("push", "pull"), ("import", "export"),
    ("lock", "unlock"), ("connect", "disconnect"), ("upload", "download"), ("compress", "decompress"),
    ("show", "hide"), ("allow", "block"), ("buy", "sell"), ("on", "off"), ("merge", "split"),
    ("before", "after"), ("accept", "reject"), ("join", "leave"), ("win", "lose"),
]

NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million", "billion",
}


def _stem(word: str) -> str:
    """Crude: 'installing', 'installed', 'installs' -> 'install', so word forms still match."""
    return re.sub(r"(ing|ed|es|s|e)$", "", word) if len(word) > 4 else word


_OPPOSITE_OF: dict[str, set[str]] = {}
for _a, _b in OPPOSITES:
    _OPPOSITE_OF.setdefault(_stem(_a), set()).add(_stem(_b))
    _OPPOSITE_OF.setdefault(_stem(_b), set()).add(_stem(_a))


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9+#]*", text)


def _numbers(text: str) -> set[str]:
    digits = set(re.findall(r"\d+(?:[.,]\d+)*", text))  # 3, 2024, 15, 3.5
    spelled = {w.lower() for w in _words(text) if w.lower() in NUMBER_WORDS}
    return digits | spelled


def _named_terms(text: str) -> set[str]:
    words = _words(text)
    named = {w.lower() for w in words if w.lower() in NAMED_TERMS}
    # Any capitalised word that isn't the first word or "I" is probably a name (Peru, Ibuprofen...).
    named |= {w.lower() for w in words[1:] if w[0].isupper() and w != "I"}
    return named


def _actions(text: str) -> set[str]:
    return {_stem(w.lower()) for w in _words(text)}


def check(question_a: str, question_b: str) -> str | None:
    """Return why the two questions must NOT share an answer, or None if they look safe to match."""
    if _numbers(question_a) != _numbers(question_b):
        return "different numbers"
    if _named_terms(question_a) != _named_terms(question_b):
        return "different named terms"
    a, b = _actions(question_a), _actions(question_b)
    for word in a ^ b:  # words that only one question has
        for opposite in _OPPOSITE_OF.get(word, ()):
            if opposite in (b if word in a else a):
                return "opposite actions"
    return None


def is_safe_to_reuse(question_a: str, question_b: str) -> bool:
    return check(question_a, question_b) is None


def check_numbers(question_a: str, question_b: str) -> str | None:
    """Only the numbers/years part of the guard (the part that generalised): 2023 vs 2024, 3 vs 10 tips."""
    return "different numbers" if _numbers(question_a) != _numbers(question_b) else None
