from pathlib import Path

from benchmark import default_ga_config, frequency_config, run_benchmark

TEST_WORDS = ("crane", "crate", "trace", "brace", "grace")


def test_benchmark_writes_csv_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    csv_path, summary = run_benchmark(
        target_words=["crate", "grace"],
        valid_words=TEST_WORDS,
        configs=[frequency_config(seed=5), default_ga_config(seed=5)],
        output_path=output,
    )
    assert csv_path.exists()
    assert csv_path.with_suffix(".summary.txt").exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "runtime" in header
    assert "fitness_trace_by_guess" in header
    assert "Best config:" in summary
