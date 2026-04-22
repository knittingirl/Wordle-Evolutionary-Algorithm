from __future__ import annotations

import copy
import random
import time
from collections import Counter
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

from deap import base, creator, tools

from baselines import build_frequency_tables, word_information_score
from benchmark import SolverConfig, SolverResult
from wordle import WORD_LENGTH, HistoryEntry, filter_consistent_words, get_feedback

_CROSSOVER_PROBABILITY = 0.90
_VALID_WORD_BONUS = 20.0
_CONSISTENCY_BONUS = 30.0
_CONSISTENCY_PENALTY = 12.0
_EXACT_HISTORY_BONUS = 18.0
_INFORMATION_WEIGHT = 0.002
_MINIMUM_FITNESS = 0.01
_STATE_ALIGNMENT_SCORES = {
    ("G", "G"): 12.0,
    ("G", "Y"): 4.0,
    ("G", "B"): -10.0,
    ("Y", "Y"): 7.0,
    ("Y", "G"): -3.0,
    ("Y", "B"): -8.0,
    ("B", "B"): 3.0,
    ("B", "Y"): -5.0,
    ("B", "G"): -7.0,
}


def solve_ga(
    target: str,
    valid_words: tuple[str, ...],
    config: SolverConfig,
    seed: int | None = None,
) -> SolverResult:
    if config.solver != "ga":
        raise ValueError("solve_ga requires a GA config")

    history: list[HistoryEntry] = []
    guessed_words: set[str] = set()
    guess_history: list[str] = []
    feedback_history: list[str] = []
    best_fitness_by_guess: list[float] = []
    diversity_by_guess: list[float] = []
    fitness_trace_by_guess: list[list[float]] = []
    diversity_trace_by_guess: list[list[float]] = []
    generations_used = 0
    start = time.perf_counter()
    base_seed = config.seed if seed is None else seed

    for guess_index in range(config.max_guesses):
        guess, guess_stats = evolve_guess(
            valid_words=valid_words,
            history=history,
            guessed_words=guessed_words,
            config=config,
            seed=base_seed + guess_index,
        )
        feedback = get_feedback(guess, target)
        history.append((guess, feedback))
        guessed_words.add(guess)
        guess_history.append(guess)
        feedback_history.append(feedback)
        generations_used += int(guess_stats["generations_used"])
        best_fitness_by_guess.append(float(guess_stats["best_fitness"]))
        diversity_by_guess.append(float(guess_stats["avg_diversity"]))
        fitness_trace_by_guess.append(guess_stats["fitness_trace"])
        diversity_trace_by_guess.append(guess_stats["diversity_trace"])
        if guess == target:
            break

    solved = guess_history[-1] == target if guess_history else False
    return SolverResult(
        solver="ga",
        target=target,
        solved=solved,
        guesses_used=len(guess_history),
        generations_used=generations_used,
        runtime_seconds=time.perf_counter() - start,
        guess_history=guess_history,
        feedback_history=feedback_history,
        best_fitness_by_guess=best_fitness_by_guess,
        diversity_by_guess=diversity_by_guess,
        fitness_trace_by_guess=fitness_trace_by_guess,
        diversity_trace_by_guess=diversity_trace_by_guess,
    )


def evolve_guess(
    valid_words: tuple[str, ...],
    history: list[HistoryEntry],
    guessed_words: set[str],
    config: SolverConfig,
    seed: int,
) -> tuple[str, dict[str, object]]:
    available_words = [word for word in valid_words if word not in guessed_words]
    if not available_words:
        available_words = list(valid_words)

    consistent_words = filter_consistent_words(available_words, history)
    scoring_words = consistent_words or available_words
    valid_word_set = set(valid_words)
    consistent_word_set = set(consistent_words)
    guessed_letters = {letter for guess, _ in history for letter in guess}
    unique_letter_counts, positional_counts = build_frequency_tables(scoring_words)
    heuristic_fallback = max(
        sorted(scoring_words),
        key=lambda word: word_information_score(
            word, unique_letter_counts, positional_counts, guessed_letters
        ),
    )

    def score_word(word: str) -> float:
        return _score_candidate_word(
            word=word,
            history=history,
            valid_word_set=valid_word_set,
            consistent_word_set=consistent_word_set,
            guessed_words=guessed_words,
            unique_letter_counts=unique_letter_counts,
            positional_counts=positional_counts,
            guessed_letters=guessed_letters,
        )

    if len(consistent_words) == 1:
        best_word = consistent_words[0]
        return best_word, {
            "best_fitness": score_word(best_word),
            "avg_diversity": 1.0,
            "generations_used": 0,
            "fitness_trace": [],
            "diversity_trace": [],
        }

    elite_count = min(config.population_size, int(config.population_size * config.elitism_rate))
    fitness_trace: list[float] = []
    diversity_trace: list[float] = []

    with temporary_random_seed(seed):
        toolbox = _build_toolbox(
            available_words=available_words,
            seed_words=scoring_words,
            config=config,
            score_word=score_word,
            heuristic_fallback=heuristic_fallback,
        )
        population = toolbox.population(n=config.population_size)
        _evaluate_population(population, toolbox)

        for _ in range(config.generations_per_guess):
            elites = tools.selBest(population, elite_count) if elite_count else []
            selected = toolbox.select(population, len(population) - elite_count)
            offspring = list(map(toolbox.clone, selected))

            for left, right in zip(offspring[::2], offspring[1::2]):
                if random.random() < _CROSSOVER_PROBABILITY:
                    toolbox.mate(left, right)
                    toolbox.repair(left)
                    toolbox.repair(right)
                    if left.fitness.valid:
                        del left.fitness.values
                    if right.fitness.valid:
                        del right.fitness.values

            for individual in offspring:
                toolbox.mutate(individual)
                toolbox.repair(individual)
                if individual.fitness.valid:
                    del individual.fitness.values

            population = elites + offspring
            _evaluate_population(population, toolbox)
            fitness_trace.append(max(ind.fitness.values[0] for ind in population))
            diversity_trace.append(population_diversity(population))

        ranked_words = _rank_population(population)

    best_guess = _choose_best_guess(
        ranked_words=ranked_words,
        consistent_word_set=consistent_word_set,
        valid_word_set=valid_word_set,
        guessed_words=guessed_words,
        fallback_word=heuristic_fallback,
    )
    return best_guess, {
        "best_fitness": max(fitness_trace) if fitness_trace else score_word(best_guess),
        "avg_diversity": sum(diversity_trace) / len(diversity_trace) if diversity_trace else 1.0,
        "generations_used": config.generations_per_guess,
        "fitness_trace": fitness_trace,
        "diversity_trace": diversity_trace,
    }


@contextmanager
def temporary_random_seed(seed: int) -> Iterator[None]:
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def population_diversity(population: Sequence[Sequence[str]]) -> float:
    if not population:
        return 0.0
    unique = {"".join(individual) for individual in population}
    return len(unique) / len(population)


def _feedback_alignment_score(expected_feedback: str, observed_feedback: str) -> float:
    return sum(
        _STATE_ALIGNMENT_SCORES[(expected_state, observed_state)]
        for expected_state, observed_state in zip(expected_feedback, observed_feedback)
    )


def _score_candidate_word(
    word: str,
    history: Sequence[HistoryEntry],
    valid_word_set: set[str],
    consistent_word_set: set[str],
    guessed_words: set[str],
    unique_letter_counts: Counter[str],
    positional_counts: Sequence[Counter[str]],
    guessed_letters: set[str],
) -> float:
    if len(word) != WORD_LENGTH or not word.isalpha() or not word.islower():
        return _MINIMUM_FITNESS
    if word not in valid_word_set:
        return _MINIMUM_FITNESS

    score = _VALID_WORD_BONUS
    if word in guessed_words:
        score -= _VALID_WORD_BONUS

    exact_matches = 0
    for guess, expected_feedback in history:
        observed_feedback = get_feedback(guess, word)
        score += _feedback_alignment_score(expected_feedback, observed_feedback)
        if observed_feedback == expected_feedback:
            exact_matches += 1

    if history:
        if word in consistent_word_set:
            score += _CONSISTENCY_BONUS + len(history) * 6.0
        else:
            score -= _CONSISTENCY_PENALTY * len(history)
        score += exact_matches * _EXACT_HISTORY_BONUS

    score += word_information_score(
        word,
        unique_letter_counts=unique_letter_counts,
        positional_counts=positional_counts,
        guessed_letters=guessed_letters,
    ) * _INFORMATION_WEIGHT

    return max(score, _MINIMUM_FITNESS)


def _mutate_candidate(
    individual: creator.WordIndividual,
    mutation_rate: float,
) -> tuple[creator.WordIndividual]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for index in range(len(individual)):
        if random.random() < mutation_rate:
            individual[index] = random.choice(alphabet)
    return (individual,)


def _repair_candidate(
    individual: creator.WordIndividual,
    available_words: Sequence[str],
    available_word_set: set[str],
    score_word: Callable[[str], float],
    fallback_word: str,
) -> creator.WordIndividual:
    word = "".join(individual)
    if word in available_word_set:
        return individual
    sample_size = min(128, len(available_words))
    repair_pool = (
        random.sample(list(available_words), sample_size)
        if len(available_words) > sample_size
        else list(available_words)
    )
    repair_pool.append(fallback_word)
    repaired_word = max(
        repair_pool,
        key=lambda candidate: (score_word(candidate), -_hamming_distance(candidate, word)),
    )
    individual[:] = list(repaired_word)
    return individual


def _build_toolbox(
    available_words: Sequence[str],
    seed_words: Sequence[str],
    config: SolverConfig,
    score_word: Callable[[str], float],
    heuristic_fallback: str,
) -> base.Toolbox:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "WordIndividual"):
        creator.create("WordIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    score_cache: dict[str, float] = {}

    def cached_score(word: str) -> float:
        if word not in score_cache:
            score_cache[word] = score_word(word)
        return score_cache[word]

    def make_individual() -> creator.WordIndividual:
        use_seed_pool = bool(seed_words) and random.random() < 0.70
        pool = seed_words if use_seed_pool else available_words
        return creator.WordIndividual(list(random.choice(pool)))

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda individual: (cached_score("".join(individual)),))
    if config.selection_type == "roulette":
        toolbox.register("select", tools.selRoulette)
    else:
        toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    if config.crossover_type == "two_point":
        toolbox.register("mate", tools.cxTwoPoint)
    elif config.crossover_type == "uniform":
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    else:
        toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", _mutate_candidate, mutation_rate=config.mutation_rate)
    toolbox.register(
        "repair",
        _repair_candidate,
        available_words=tuple(available_words),
        available_word_set=set(available_words),
        score_word=cached_score,
        fallback_word=heuristic_fallback,
    )
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def _evaluate_population(population: list[creator.WordIndividual], toolbox: base.Toolbox) -> None:
    invalid = [individual for individual in population if not individual.fitness.valid]
    for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
        individual.fitness.values = fitness


def _hamming_distance(left: str, right: str) -> int:
    return sum(left_char != right_char for left_char, right_char in zip(left, right))


def _rank_population(population: Sequence[Sequence[str]]) -> list[str]:
    return [
        "".join(individual)
        for individual in sorted(
            population,
            key=lambda individual: individual.fitness.values[0],
            reverse=True,
        )
    ]


def _choose_best_guess(
    ranked_words: Sequence[str],
    consistent_word_set: set[str],
    valid_word_set: set[str],
    guessed_words: set[str],
    fallback_word: str,
) -> str:
    for word in ranked_words:
        if word in consistent_word_set and word not in guessed_words:
            return word
    for word in ranked_words:
        if word in valid_word_set and word not in guessed_words:
            return word
    return fallback_word
