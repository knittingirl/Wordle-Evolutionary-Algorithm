# Wordle GA Project

This repo is a small Wordle experiment project with three solver styles:

- a random baseline
- a frequency-based baseline
- a DEAP-backed genetic algorithm

The point is not just to solve one puzzle. The code is set up so you can run repeatable benchmarks and compare GA settings against the simpler baselines.

## What each file does

### `main.py`

This is the entrypoint. It just hands off to the benchmark CLI.

If you run:

```bash
python main.py benchmark --mode default
```

this is the first file Python enters.

### `benchmark.py`

This is the main runner for the project.

It is responsible for:

- loading `words.txt` and `targets.txt`
- building the config sets for default, isolation, and full-grid runs
- dispatching the solver runs
- writing the CSV output
- printing the summary at the end

If you want to understand the overall flow of the project, start here.

### `wordle.py`

This is the rules engine.

It handles:

- Wordle feedback with `G`, `Y`, and `B`
- constraint building from past guesses
- consistency filtering for candidate words

Both the baselines and the GA depend on this file.

### `baselines.py`

This file has the two non-GA solvers:

- `random`, which picks a valid consistent word at random
- `frequency`, which scores candidate words using letter and position frequency

These give the GA something reasonable to compete against.

### `ga_solver.py`

This is the actual genetic algorithm solver.

It:

- builds populations with DEAP
- applies selection, crossover, mutation, and repair
- scores candidate words against prior Wordle feedback
- tracks per-guess fitness and diversity traces

This is where the experimental part of the project lives.

### `words.txt`

This is the dictionary of valid 5-letter words used during solving.

### `targets.txt`

This is the benchmark target set. These are the words the solvers are tested against in benchmark runs.

### `pyproject.toml`

This is the project config file for packaging and tooling.

It tells Python tools:

- the project name
- which dependencies to install
- which dev dependencies to install
- what console command to expose
- how `pytest` should run

This is what makes `pip install -e ".[dev]"` and the `wordle-ga` command work.

## How the files work together

The short version is:

`main.py` -> `benchmark.py` -> load words and targets -> build configs -> run `baselines.py` and/or `ga_solver.py` -> both use `wordle.py` -> write results

The normal flow looks like this:

1. `main.py` starts the CLI.
2. `benchmark.py` parses the flags.
3. `benchmark.py` loads `words.txt` and `targets.txt`.
4. `benchmark.py` builds one or more solver configs.
5. `benchmark.py` sends each run to either the baseline solver or the GA solver.
6. Those solvers use `wordle.py` to compute feedback and filter valid candidates.
7. `benchmark.py` writes the CSV and summary file.

## Where `words.txt` came from

`words.txt` was generated locally from:

```text
/usr/share/dict/words
```

It was filtered down to:

- lowercase words only
- exactly 5 letters
- unique entries

So this is not an official Wordle answer list. It is a cleaned local dictionary source that works well for a reproducible experiment.

`targets.txt` was created by taking a reproducible sample from that word list for benchmarking.

## Setup from a fresh clone

If you are setting this up from scratch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

That will:

- create the virtual environment
- activate it
- install the project and `pytest`

After you close the terminal, you will need to activate the virtual environment again before running commands:

```bash
source .venv/bin/activate
```

## How to run it

### Default benchmark

This is the normal starting point:

```bash
python main.py benchmark --mode default
```

This runs:

- the random baseline
- the frequency baseline
- the default GA configuration

It still uses the full benchmark runner. `default` just means the smallest config set.

### Small smoke run

Useful for a quick check:

```bash
python main.py benchmark --mode default --limit 20 --output results/smoke.csv
```

### Isolation benchmark

This is where you start comparing GA parameter families:

```bash
python main.py benchmark --mode isolation --solver ga
```

This changes one parameter family at a time while keeping the rest at the default settings.

### Full grid benchmark

This is the biggest run:

```bash
python main.py benchmark --mode full-grid --solver ga --output results/full_grid.csv
```

This runs the full supported GA configuration grid.

### Console script version

After installing with `pip install -e ".[dev]"`, you can also run:

```bash
wordle-ga benchmark --mode default
```

That is just a cleaner alias for the same benchmark CLI.

## Main flags

### `--mode`

Controls how many configs get generated.

- `default`: run the default GA config plus the two baselines
- `isolation`: run multiple GA configs, varying one parameter family at a time
- `full-grid`: run the full GA combination grid

### `--solver`

Controls which solver family gets included.

- `all`: run baselines and GA
- `random`: run only the random solver
- `frequency`: run only the frequency solver
- `ga`: run only the genetic algorithm

### `--limit`

Limits how many target words from `targets.txt` are used.

Example:

```bash
python main.py benchmark --mode default --limit 10
```

### `--output`

Lets you choose where the CSV results should be written.

Example:

```bash
python main.py benchmark --mode default --output results/my_run.csv
```

### `--seed`

Sets the random seed so runs are reproducible.

Example:

```bash
python main.py benchmark --mode default --seed 123
```

### `--valid-words`

Overrides the default dictionary file.

Example:

```bash
python main.py benchmark --mode default --valid-words words.txt
```

### `--targets`

Overrides the default benchmark target file.

Example:

```bash
python main.py benchmark --mode default --targets targets.txt
```

## Output

Each benchmark run writes:

- a CSV file with one row per config/target pair
- a summary text file with aggregate results

The summary includes:

- success rate
- average guesses
- average generations
- average runtime
- best config
- worst config

## Running tests

If you want to make sure everything still works:

```bash
.venv/bin/pytest
```

## Practical starting point

If you just want the normal workflow, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python main.py benchmark --mode default
```

Then move on to:

```bash
python main.py benchmark --mode isolation --solver ga
```
