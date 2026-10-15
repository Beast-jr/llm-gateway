import pytest

from services.match_guard import check, check_numbers, is_safe_to_reuse


@pytest.mark.parametrize("a, b, reason", [
    ("How do I reverse a list in Python?", "How do I reverse a list in JavaScript?", "different named terms"),
    ("Who won the World Cup in 2023?", "Who won the World Cup in 2024?", "different numbers"),
    ("Give me 3 tips", "Give me 10 tips", "different numbers"),
    ("How do I open a port in the firewall?", "How do I close a port in the firewall?", "opposite actions"),
    ("How do I install Node?", "How do I uninstall Node?", "opposite actions"),
    ("What is the capital of Peru?", "What is the capital of Chile?", "different named terms"),  # capitalised names
    ("How do I reverse a list?", "How do I sort a list?", "opposite actions"),
])
def test_guard_blocks_different_specifics(a, b, reason):
    assert check(a, b) == reason
    assert not is_safe_to_reuse(a, b)


@pytest.mark.parametrize("a, b", [
    ("What is RAG?", "Explain RAG to me"),
    ("How do I reverse a list in Python?", "What's the way to reverse a Python list?"),
    ("What is the capital of France?", "Which city is the capital of France?"),
    ("How do I install Docker on Ubuntu?", "What are the steps to install Docker on Ubuntu?"),
    ("Why is the sky blue?", "What makes the sky look blue?"),
])
def test_guard_allows_true_paraphrases(a, b):
    assert check(a, b) is None


def test_open_and_close_in_both_questions_is_fine():
    assert is_safe_to_reuse("How do I open and close a file in Python?", "How can I open and close a Python file?")


def test_numbers_only_guard_blocks_years_and_amounts():
    assert check_numbers("Who won in 2023?", "Who won in 2024?") == "different numbers"
    assert check_numbers("Give me 3 tips", "Give me 10 tips") == "different numbers"


def test_numbers_only_guard_ignores_names_and_actions():
    # production relies on the cutoff for these; the word lists proved too narrow
    assert check_numbers("How do I reverse a list in Python?", "How do I reverse a list in JavaScript?") is None
    assert check_numbers("How do I open a port?", "How do I close a port?") is None


def test_numbers_only_guard_allows_paraphrases():
    assert check_numbers("What is RAG?", "Explain RAG to me") is None
