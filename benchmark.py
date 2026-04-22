from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal

SelectionType = Literal["tournament", "roulette"]
CrossoverType = Literal["single_point", "two_point", "uniform"]
SolverType = Literal["random", "frequency", "ga"]

POPULATION_SIZES = (50, 100, 200, 500)
MUTATION_RATES = (0.01, 0.05, 0.10, 0.20)
TOURNAMENT_SIZES = (2, 3, 5)
CROSSOVER_TYPES = ("single_point", "two_point", "uniform")
ELITISM_RATES = (0.00, 0.05, 0.10, 0.20)
GENERATIONS_PER_GUESS = 25
DEFAULT_SEED = 5512

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_WORDS_PATH = PROJECT_ROOT / "words.txt"
DEFAULT_TARGETS_PATH = PROJECT_ROOT / "targets.txt"


def load_word_list(path: str | Path) -> tuple[str, ...]:
    words = [
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(sorted(set(words)))


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _serialize_trace_matrix(traces: list[list[float]]) -> str:
    return "|".join(
        ",".join(f"{value:.6f}" for value in trace)
        for trace in traces
    )


@dataclass(frozen=True, slots=True)
class SolverConfig:
    solver: SolverType = "ga"
    population_size: int = 100
    mutation_rate: float = 0.05
    selection_type: SelectionType = "tournament"
    tournament_size: int = 3
    crossover_type: CrossoverType = "single_point"
    elitism_rate: float = 0.10
    generations_per_guess: int = GENERATIONS_PER_GUESS
    max_guesses: int = 6
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.solver not in {"random", "frequency", "ga"}:
            raise ValueError(f"Unsupported solver: {self.solver}")
        if self.max_guesses <= 0:
            raise ValueError("max_guesses must be positive")
        if self.solver != "ga":
            object.__setattr__(self, "population_size", 0)
            object.__setattr__(self, "mutation_rate", 0.0)
            object.__setattr__(self, "selection_type", "tournament")
            object.__setattr__(self, "tournament_size", 0)
            object.__setattr__(self, "crossover_type", "single_point")
            object.__setattr__(self, "elitism_rate", 0.0)
            object.__setattr__(self, "generations_per_guess", 0)
            return
        if self.selection_type not in {"tournament", "roulette"}:
            raise ValueError(f"Unsupported selection type: {self.selection_type}")
        if self.crossover_type not in {"single_point", "two_point", "uniform"}:
            raise ValueError(f"Unsupported crossover type: {self.crossover_type}")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0 <= self.elitism_rate <= 1:
            raise ValueError("elitism_rate must be between 0 and 1")
        if self.generations_per_guess < 0:
            raise ValueError("generations_per_guess must be non-negative")
        if self.selection_type == "roulette":
            object.__setattr__(self, "tournament_size", 0)
        elif self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")

    @property
    def config_id(self) -> str:
        if self.solver != "ga":
            return f"{self.solver}_seed{self.seed}"
        selection = self.selection_type
        if self.selection_type == "tournament":
            selection = f"tournament{self.tournament_size}"
        mutation = f"{self.mutation_rate:.2f}".replace(".", "p")
        elitism = f"{self.elitism_rate:.2f}".replace(".", "p")
        return (
            f"ga_pop{self.population_size}_mut{mutation}_sel{selection}"
            f"_cross{self.crossover_type}_elite{elitism}"
            f"_gen{self.generations_per_guess}_seed{self.seed}"
        )

    def with_overrides(self, **kwargs: object) -> "SolverConfig":
        return replace(self, **kwargs)


@dataclass(slots=True)
class SolverResult:
    solver: SolverType
    target: str
    solved: bool
    guesses_used: int
    generations_used: int
    runtime_seconds: float
    guess_history: list[str] = field(default_factory=list)
    feedback_history: list[str] = field(default_factory=list)
    best_fitness_by_guess: list[float] = field(default_factory=list)
    diversity_by_guess: list[float] = field(default_factory=list)
    fitness_trace_by_guess: list[list[float]] = field(default_factory=list)
    diversity_trace_by_guess: list[list[float]] = field(default_factory=list)

    def to_row(self, config: SolverConfig) -> dict[str, object]:
        return {
            "config_id": config.config_id,
            "solver": self.solver,
            "target": self.target,
            "solved": self.solved,
            "guesses": self.guesses_used,
            "generations_used": self.generations_used,
            "runtime": round(self.runtime_seconds, 6),
            "runtime_seconds": round(self.runtime_seconds, 6),
            "guess_history": ",".join(self.guess_history),
            "feedback_history": ",".join(self.feedback_history),
            "avg_best_fitness": round(mean(self.best_fitness_by_guess), 6),
            "avg_population_diversity": round(mean(self.diversity_by_guess), 6),
            "fitness_trace_by_guess": _serialize_trace_matrix(self.fitness_trace_by_guess),
            "diversity_trace_by_guess": _serialize_trace_matrix(
                self.diversity_trace_by_guess
            ),
            "population_size": config.population_size,
            "mutation_rate": config.mutation_rate,
            "selection_type": config.selection_type,
            "tournament_size": config.tournament_size,
            "crossover_type": config.crossover_type,
            "elitism_rate": config.elitism_rate,
            "generations_per_guess": config.generations_per_guess,
            "max_guesses": config.max_guesses,
            "seed": config.seed,
        }


def default_ga_config(seed: int = DEFAULT_SEED) -> SolverConfig:
    return SolverConfig(seed=seed)


def random_config(seed: int = DEFAULT_SEED) -> SolverConfig:
    return SolverConfig(solver="random", seed=seed)


def frequency_config(seed: int = DEFAULT_SEED) -> SolverConfig:
    return SolverConfig(solver="frequency", seed=seed)


def build_parameter_isolation_configs(
    base: SolverConfig | None = None, seed: int = DEFAULT_SEED
) -> list[SolverConfig]:
    base = base or default_ga_config(seed=seed)
    configs = [base]
    for population_size in POPULATION_SIZES:
        configs.append(base.with_overrides(population_size=population_size))
    for mutation_rate in MUTATION_RATES:
        configs.append(base.with_overrides(mutation_rate=mutation_rate))
    configs.append(base.with_overrides(selection_type="roulette", tournament_size=0))
    for tournament_size in TOURNAMENT_SIZES:
        configs.append(
            base.with_overrides(
                selection_type="tournament",
                tournament_size=tournament_size,
            )
        )
    for crossover_type in CROSSOVER_TYPES:
        configs.append(base.with_overrides(crossover_type=crossover_type))
    for elitism_rate in ELITISM_RATES:
        configs.append(base.with_overrides(elitism_rate=elitism_rate))
    return list(dict.fromkeys(configs))


def build_full_ga_grid(seed: int = DEFAULT_SEED) -> list[SolverConfig]:
    configs: list[SolverConfig] = []
    selection_variants: tuple[tuple[SelectionType, int], ...] = (
        ("roulette", 0),
        ("tournament", 2),
        ("tournament", 3),
        ("tournament", 5),
    )
    for population_size in POPULATION_SIZES:
        for mutation_rate in MUTATION_RATES:
            for selection_type, tournament_size in selection_variants:
                for crossover_type in CROSSOVER_TYPES:
                    for elitism_rate in ELITISM_RATES:
                        configs.append(
                            SolverConfig(
                                population_size=population_size,
                                mutation_rate=mutation_rate,
                                selection_type=selection_type,
                                tournament_size=tournament_size,
                                crossover_type=crossover_type,
                                elitism_rate=elitism_rate,
                                generations_per_guess=GENERATIONS_PER_GUESS,
                                seed=seed,
                            )
                        )
    return configs


def run_benchmark(
    target_words: Iterable[str],
    valid_words: tuple[str, ...],
    configs: list[SolverConfig],
    output_path: str | Path,
) -> tuple[Path, str]:
    target_list = list(target_words)
    rows: list[dict[str, object]] = []
    for config in configs:
        solver = _resolve_solver(config)
        for target in target_list:
            result = solver(target=target, valid_words=valid_words, config=config, seed=config.seed)
            rows.append(result.to_row(config))
    csv_path = write_csv(output_path, rows)
    summary = summarize_results(rows)
    summary_path = csv_path.with_suffix(".summary.txt")
    summary_path.write_text(summary, encoding="utf-8")
    return csv_path, summary


def summarize_results(rows: list[dict[str, object]]) -> str:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)

    lines = ["Benchmark Summary", ""]
    metrics: list[tuple[str, float, float, float]] = []

    for config_id in sorted(grouped):
        config_rows = grouped[config_id]
        success_rate = sum(1 for row in config_rows if row["solved"]) / len(config_rows)
        average_guesses = mean([float(row["guesses"]) for row in config_rows])
        average_runtime = mean([float(row["runtime"]) for row in config_rows])
        average_generations = mean([float(row["generations_used"]) for row in config_rows])
        lines.append(
            (
                f"{config_id}: success_rate={success_rate:.3f}, "
                f"avg_guesses={average_guesses:.3f}, "
                f"avg_generations={average_generations:.3f}, "
                f"avg_runtime={average_runtime:.6f}s"
            )
        )
        metrics.append((config_id, success_rate, average_guesses, average_runtime))

    best_config = max(metrics, key=lambda item: (item[1], -item[2], -item[3]))[0] if metrics else None
    worst_config = min(metrics, key=lambda item: (item[1], -item[2], -item[3]))[0] if metrics else None

    lines.extend(
        [
            "",
            f"Best config: {best_config}",
            f"Worst config: {worst_config}",
        ]
    )
    return "\n".join(lines)


def build_configs(mode: str, solver: str, seed: int) -> list[SolverConfig]:
    configs: list[SolverConfig] = []
    if solver in {"all", "random"}:
        configs.append(random_config(seed=seed))
    if solver in {"all", "frequency"}:
        configs.append(frequency_config(seed=seed))
    if solver not in {"all", "ga"}:
        return configs

    if mode == "default":
        configs.append(default_ga_config(seed=seed))
    elif mode == "isolation":
        configs.extend(build_parameter_isolation_configs(seed=seed))
    elif mode == "full-grid":
        configs.extend(build_full_ga_grid(seed=seed))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return configs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wordle solver benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark batches.")
    benchmark_parser.add_argument(
        "--mode",
        choices=("default", "isolation", "full-grid"),
        default="default",
        help="Configuration generation strategy.",
    )
    benchmark_parser.add_argument(
        "--solver",
        choices=("all", "random", "frequency", "ga"),
        default="all",
        help="Which solver family to benchmark.",
    )
    benchmark_parser.add_argument(
        "--valid-words",
        default=str(DEFAULT_WORDS_PATH),
        help="Path to the valid words file.",
    )
    benchmark_parser.add_argument(
        "--targets",
        default=str(DEFAULT_TARGETS_PATH),
        help="Path to the benchmark target list.",
    )
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of target words to run.",
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible runs.",
    )
    benchmark_parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "benchmark_results.csv"),
        help="CSV output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "benchmark":
        raise ValueError(f"Unsupported command: {args.command}")

    valid_words = load_word_list(args.valid_words)
    target_words = list(load_word_list(args.targets))
    if args.limit:
        target_words = target_words[: args.limit]
    if not target_words:
        raise ValueError("No target words available for benchmark run")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs = build_configs(mode=args.mode, solver=args.solver, seed=args.seed)
    csv_path, summary = run_benchmark(
        target_words=target_words,
        valid_words=valid_words,
        configs=configs,
        output_path=args.output,
    )
    print(f"Wrote benchmark results to {csv_path}")
    print(summary)
    return 0


def _resolve_solver(config: SolverConfig):
    if config.solver == "random":
        from baselines import solve_random

        return solve_random
    if config.solver == "frequency":
        from baselines import solve_frequency

        return solve_frequency
    if config.solver == "ga":
        from ga_solver import solve_ga

        return solve_ga
    raise ValueError(f"Unsupported solver: {config.solver}")
