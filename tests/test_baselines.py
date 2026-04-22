from baselines import solve_frequency, solve_random
from benchmark import frequency_config, random_config

TEST_WORDS = ("crane", "crate", "trace", "brace", "grace")


def test_random_solver_returns_valid_history() -> None:
    result = solve_random("crate", TEST_WORDS, random_config(seed=7))
    assert result.guess_history
    assert all(word in TEST_WORDS for word in result.guess_history)
    assert len(result.guess_history) == len(set(result.guess_history))
    assert result.solved


def test_frequency_solver_solves_small_pool() -> None:
    result = solve_frequency("crate", TEST_WORDS, frequency_config(seed=7))
    assert result.solved
    assert result.guess_history[-1] == "crate"
