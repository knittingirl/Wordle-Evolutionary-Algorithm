from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

WORD_LENGTH = 5
HistoryEntry = tuple[str, str]


@dataclass(slots=True)
class WordleConstraints:
    required_positions: list[str | None]
    forbidden_positions: list[set[str]]
    min_counts: dict[str, int]
    max_counts: dict[str, int]


def get_feedback(guess: str, target: str) -> str:
    _validate_word(guess)
    _validate_word(target)
    feedback = ["B"] * WORD_LENGTH
    remaining_letters = Counter()

    for index, (guess_letter, target_letter) in enumerate(zip(guess, target)):
        if guess_letter == target_letter:
            feedback[index] = "G"
        else:
            remaining_letters[target_letter] += 1

    for index, guess_letter in enumerate(guess):
        if feedback[index] == "G":
            continue
        if remaining_letters[guess_letter] > 0:
            feedback[index] = "Y"
            remaining_letters[guess_letter] -= 1

    return "".join(feedback)


def build_constraints(history: Iterable[HistoryEntry]) -> WordleConstraints:
    required_positions: list[str | None] = [None] * WORD_LENGTH
    forbidden_positions = [set() for _ in range(WORD_LENGTH)]
    min_counts: dict[str, int] = {}
    max_counts: dict[str, int] = {}

    for guess, feedback in history:
        _validate_word(guess)
        _validate_feedback(feedback)
        positive_counts = Counter(
            letter
            for letter, state in zip(guess, feedback)
            if state in {"G", "Y"}
        )
        guessed_counts = Counter(guess)

        for index, (letter, state) in enumerate(zip(guess, feedback)):
            if state == "G":
                required_positions[index] = letter
            else:
                forbidden_positions[index].add(letter)

        for letter, count in positive_counts.items():
            min_counts[letter] = max(min_counts.get(letter, 0), count)

        for letter, guessed_count in guessed_counts.items():
            positive_count = positive_counts.get(letter, 0)
            if positive_count < guessed_count:
                current_max = max_counts.get(letter, WORD_LENGTH)
                max_counts[letter] = min(current_max, positive_count)

    return WordleConstraints(
        required_positions=required_positions,
        forbidden_positions=forbidden_positions,
        min_counts=min_counts,
        max_counts=max_counts,
    )


def is_consistent(word: str, constraints: WordleConstraints) -> bool:
    _validate_word(word)
    counts = Counter(word)
    for index, letter in enumerate(word):
        required = constraints.required_positions[index]
        if required is not None and letter != required:
            return False
        if letter in constraints.forbidden_positions[index]:
            return False
    for letter, minimum in constraints.min_counts.items():
        if counts[letter] < minimum:
            return False
    for letter, maximum in constraints.max_counts.items():
        if counts[letter] > maximum:
            return False
    return True


def filter_consistent_words(
    words: Iterable[str], history: Iterable[HistoryEntry]
) -> list[str]:
    constraints = build_constraints(history)
    return [word for word in words if is_consistent(word, constraints)]


def _validate_word(word: str) -> None:
    if len(word) != WORD_LENGTH or not word.isalpha() or not word.islower():
        raise ValueError(f"Invalid Wordle word: {word!r}")


def _validate_feedback(feedback: str) -> None:
    if len(feedback) != WORD_LENGTH or any(state not in {"G", "Y", "B"} for state in feedback):
        raise ValueError(f"Invalid Wordle feedback: {feedback!r}")
