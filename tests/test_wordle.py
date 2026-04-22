from wordle import build_constraints, filter_consistent_words, get_feedback, is_consistent


def test_get_feedback_handles_repeated_letters() -> None:
    assert get_feedback("allee", "eagle") == "YYBYG"


def test_constraints_filter_consistent_words() -> None:
    history = [("crane", get_feedback("crane", "crate"))]
    constraints = build_constraints(history)
    assert is_consistent("crate", constraints)
    assert not is_consistent("crone", constraints)
    filtered = filter_consistent_words(["crate", "crone", "brace"], history)
    assert filtered == ["crate"]
