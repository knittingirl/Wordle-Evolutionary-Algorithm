from __future__ import annotations

import random
import time
from collections import Counter
from typing import Callable, Iterable, Sequence

from benchmark import SolverConfig, SolverResult
from wordle import WORD_LENGTH, HistoryEntry, filter_consistent_words, get_feedback


def solve_random(
    target: str,
    valid_words: tuple[str, ...],
    config: SolverConfig,
    seed: int | None = None,
) -> SolverResult:
    rng = random.Random(config.seed if seed is None else seed)

    def chooser(
        candidates: list[str],
        history: list[HistoryEntry],
        guessed_words: set[str],
    ) -> str:
        _ = history
        unseen = [word for word in candidates if word not in guessed_words]
        pool = unseen or candidates
        return rng.choice(pool)

    return _run_solver_loop(
        target=target,
        valid_words=valid_words,
        config=config,
        chooser=chooser,
    )


def solve_frequency(
    target: str,
    valid_words: tuple[str, ...],
    config: SolverConfig,
    seed: int | None = None,
) -> SolverResult:
    _ = seed

    def chooser(
        candidates: list[str],
        history: list[HistoryEntry],
        guessed_words: set[str],
    ) -> str:
        return choose_frequency_guess(candidates, history, guessed_words)

    return _run_solver_loop(
        target=target,
        valid_words=valid_words,
        config=config,
        chooser=chooser,
    )


def choose_frequency_guess(
    candidates: list[str],
    history: list[HistoryEntry],
    guessed_words: set[str],
) -> str:
    unseen = [word for word in candidates if word not in guessed_words]
    pool = unseen or candidates
    guessed_letters = {letter for guess, _ in history for letter in guess}
    unique_letter_counts, positional_counts = build_frequency_tables(pool)
    return max(
        sorted(pool),
        key=lambda word: word_information_score(
            word, unique_letter_counts, positional_counts, guessed_letters
        ),
    )


def build_frequency_tables(
    words: Sequence[str],
) -> tuple[Counter[str], list[Counter[str]]]:
    unique_letter_counts: Counter[str] = Counter()
    positional_counts = [Counter() for _ in range(WORD_LENGTH)]
    for word in words:
        unique_letter_counts.update(set(word))
        for index, letter in enumerate(word):
            positional_counts[index][letter] += 1
    return unique_letter_counts, positional_counts


def word_information_score(
    word: str,
    unique_letter_counts: Counter[str],
    positional_counts: Sequence[Counter[str]],
    guessed_letters: Iterable[str] = (),
) -> float:
    guessed_letter_set = set(guessed_letters)
    duplicate_penalty = sum(count - 1 for count in Counter(word).values() if count > 1)
    unique_score = sum(unique_letter_counts[letter] for letter in set(word))
    positional_score = sum(positional_counts[index][letter] for index, letter in enumerate(word))
    novelty_bonus = sum(1 for letter in set(word) if letter not in guessed_letter_set)
    return positional_score * 2.0 + unique_score + novelty_bonus * 1.5 - duplicate_penalty * 2.5


def _run_solver_loop(
    target: str,
    valid_words: tuple[str, ...],
    config: SolverConfig,
    chooser: Callable[[list[str], list[HistoryEntry], set[str]], str],
) -> SolverResult:
    start = time.perf_counter()
    history: list[HistoryEntry] = []
    guess_history: list[str] = []
    feedback_history: list[str] = []
    guessed_words: set[str] = set()

    for _ in range(config.max_guesses):
        candidates = filter_consistent_words(valid_words, history)
        if not candidates:
            candidates = list(valid_words)
        guess = chooser(candidates, history, guessed_words)
        feedback = get_feedback(guess, target)
        history.append((guess, feedback))
        guess_history.append(guess)
        feedback_history.append(feedback)
        guessed_words.add(guess)
        if guess == target:
            break

    solved = guess_history[-1] == target if guess_history else False
    return SolverResult(
        solver=config.solver,
        target=target,
        solved=solved,
        guesses_used=len(guess_history),
        generations_used=0,
        runtime_seconds=time.perf_counter() - start,
        guess_history=guess_history,
        feedback_history=feedback_history,
    )
