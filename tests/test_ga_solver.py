from benchmark import (
    SolverConfig,
    build_full_ga_grid,
    build_parameter_isolation_configs,
    default_ga_config,
)
from baselines import build_frequency_tables
from ga_solver import _score_candidate_word, solve_ga
from wordle import get_feedback

TEST_WORDS = ("crane", "crate", "trace", "brace", "grace", "slate", "plane")


def test_ga_solver_solves_small_pool() -> None:
    result = solve_ga("crate", TEST_WORDS, default_ga_config(seed=11))
    assert result.solved
    assert result.guess_history[-1] == "crate"
    assert result.generations_used >= 0
    assert len(result.fitness_trace_by_guess) == result.guesses_used
    assert len(result.diversity_trace_by_guess) == result.guesses_used
    assert all(
        len(trace) in {0, default_ga_config(seed=11).generations_per_guess}
        for trace in result.fitness_trace_by_guess
    )


def test_feedback_aware_fitness_prefers_consistent_word() -> None:
    history = [("crane", get_feedback("crane", "crate"))]
    scoring_words = [word for word in TEST_WORDS if word != "crane"]
    consistent_words = {"crate"}
    guessed_letters = {"c", "r", "a", "n", "e"}
    unique_letter_counts, positional_counts = build_frequency_tables(scoring_words)
    consistent_score = _score_candidate_word(
        word="crate",
        history=history,
        valid_word_set=set(TEST_WORDS),
        consistent_word_set=consistent_words,
        guessed_words={"crane"},
        unique_letter_counts=unique_letter_counts,
        positional_counts=positional_counts,
        guessed_letters=guessed_letters,
    )
    inconsistent_score = _score_candidate_word(
        word="trace",
        history=history,
        valid_word_set=set(TEST_WORDS),
        consistent_word_set=consistent_words,
        guessed_words={"crane"},
        unique_letter_counts=unique_letter_counts,
        positional_counts=positional_counts,
        guessed_letters=guessed_letters,
    )
    assert consistent_score > inconsistent_score


def test_ga_config_builders_support_roulette() -> None:
    isolation_configs = build_parameter_isolation_configs()
    assert any(config.selection_type == "roulette" for config in isolation_configs)
    full_grid = build_full_ga_grid(seed=3)
    assert any(config.selection_type == "roulette" for config in full_grid)
    assert len(full_grid) == 768


def test_supported_ga_variants_execute() -> None:
    configs = [
        SolverConfig(
            population_size=20,
            mutation_rate=0.01,
            selection_type="tournament",
            tournament_size=2,
            crossover_type="single_point",
            elitism_rate=0.00,
            generations_per_guess=3,
            max_guesses=4,
            seed=1,
        ),
        SolverConfig(
            population_size=20,
            mutation_rate=0.05,
            selection_type="tournament",
            tournament_size=3,
            crossover_type="two_point",
            elitism_rate=0.05,
            generations_per_guess=3,
            max_guesses=4,
            seed=2,
        ),
        SolverConfig(
            population_size=20,
            mutation_rate=0.10,
            selection_type="tournament",
            tournament_size=5,
            crossover_type="uniform",
            elitism_rate=0.10,
            generations_per_guess=3,
            max_guesses=4,
            seed=3,
        ),
        SolverConfig(
            population_size=20,
            mutation_rate=0.20,
            selection_type="roulette",
            tournament_size=0,
            crossover_type="single_point",
            elitism_rate=0.20,
            generations_per_guess=3,
            max_guesses=4,
            seed=4,
        ),
    ]
    for config in configs:
        result = solve_ga("crate", TEST_WORDS, config)
        assert result.guess_history
        assert all(word in TEST_WORDS for word in result.guess_history)
        assert all(
            len(trace) in {0, config.generations_per_guess}
            for trace in result.fitness_trace_by_guess
        )
