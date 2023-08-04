"""Measure performance of the kernel module and compare to prior runs.

This project is split into two parts: the inner loop is an optimized "kernel"
written in C++, while the rest of the program is written in Python for ease of
development and readability. As such, one goal is to get the kernel to run
quickly and reliably independent of the rest of the code. This is a simple tool
to help with that.

This script runs this project's inner loop a few times and measures how long it
takes. It keeps a history of the last few runs in order to test the performance
impact of a code change. It displays the results as text on the command line,
and optionally as a chart visualization.
"""

import argparse
from datetime import datetime
import pathlib
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import kernel
import phenotype_program

# Hyperparameters for evolution. These are set to match the formal experiments
# for now, but this is arbitrary since the main purpose of this script is to
# compare relative performance across runs.
NUM_SPECIES = 50
NUM_TRIALS = 5
NUM_ORGANISMS = 50
NUM_GENERATIONS = 200
POPULATION_SIZE = NUM_SPECIES * NUM_TRIALS * NUM_ORGANISMS
NUM_LIFETIMES = POPULATION_SIZE * NUM_GENERATIONS

# Repeat each benchmark NUM_SAMPLES times to reduce noise.
NUM_SAMPLES = 3

# Remember the last HISTORY_SIZE benchmarks for comparison.
HISTORY_SIZE = 5

# For testing, just make NUM_SPECIES copies of the same PhenotypeProgram.
CLADE = phenotype_program.TestClade(NUM_SPECIES)


def sample_performance():
    """Measure the performance of running this project's inner loop once."""
    # Don't count the one-time initialization in our measurement.
    random.seed(42)
    simulator = kernel.Simulator(NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
    goal = kernel.FitnessGoal.STILL_LIFE

    start = time.perf_counter()
    # TODO: Should this just call evolve()? Or should we keep it like this to
    # include the cost of get_fitness_scores() in the benchmark?
    simulator.populate(CLADE.serialize())
    for _ in range(NUM_GENERATIONS - 1):
        simulator.simulate(goal)
        simulator.get_fitness_scores()
        simulator.propagate()
    simulator.simulate(goal)
    fitness = simulator.get_fitness_scores()
    elapsed = time.perf_counter() - start
    return fitness, elapsed


def collect_samples():
    """Collect NUM_SAMPLES performance samples and sanity check results."""
    # Performance doesn't count if the results are wrong, so we keep track of
    # what the "right" fitness scores should be in a file on disk and assert we
    # get the same results every time. Note that this file should just be
    # deleted manually if any changes to the C++ code intentionally change the
    # expected fitness results.
    fitness_file = pathlib.Path('output/benchmark/fitness.npy')
    fitness_file.parent.mkdir(parents=True, exist_ok=True)
    if fitness_file.exists():
        expected_fitness = np.load(fitness_file)
    else:
        expected_fitness = None

    # Actually collect some new performance samples.
    print(f'Collecting {NUM_SAMPLES} samples of {NUM_LIFETIMES} lifetimes:')
    performance_samples = []
    for sample in range(NUM_SAMPLES):
        # To avoid measurement noise, this script does not show a progress bar
        # as simulations run. This can take a while, so indicate to the user
        # when each sample collection begins and ends (on the same line).
        pre_msg = f' - Sample {sample}: Running simulations...'
        # Print to the screen without a newline, so we can overwrite it later.
        print(pre_msg, end='', flush=True)

        # Actually run the benchmark.
        fitness, elapsed = sample_performance()
        warning = ''
        # If there was no fitness history save to disk, use the first run as a
        # baseline to compare the rest agains.
        if expected_fitness is None:
            expected_fitness = fitness
        elif not np.array_equal(expected_fitness, fitness):
            # If running the same simulations produces different results, then
            # something is wrong and benchmark results may not be meaningful.
            warning = ' (WARNING: inconsistent fitness)'

        # Compute performance in terms of thousands of lifetimes per second.
        klps = NUM_LIFETIMES / elapsed / 1000
        performance_samples.append(klps)
        # Show results by overwriting the message from before, adding
        # whitespace to the end if necessary to erase pre_msg from the screen.
        post_msg = f' - Sample {sample}: {elapsed:.2f}s {klps:.2f}klps{warning}'
        padding = max(0, len(pre_msg) - len(post_msg) - len(warning)) * ' '
        print(f'\r{post_msg}{warning}{padding}')

    # Save the last set of fitness scores as the new baseline.
    np.save(fitness_file, expected_fitness)
    return performance_samples


def update_history(performance_samples):
    """Combine a new set of samples with historical data."""
    # This script tracks performance over the last HISTORY_SIZE runs in a file
    # on disk. Try loading that historical data if its available.
    history_file = pathlib.Path('output/benchmark/history.csv')
    history_file.parent.mkdir(parents=True, exist_ok=True)
    next_sample_id = 0
    if history_file.exists():
        performance_history = pd.read_csv(history_file)
        next_sample_id = performance_history['SampleId'].max() + 1
    else:
        performance_history = pd.DataFrame()

    # Add the new data points to the history, truncate, and save it to disk.
    new_data = pd.DataFrame({
        'Timestamp': datetime.now().timestamp(),
        'SampleId': next_sample_id,
        'klps': performance_samples
    })
    performance_history = pd.concat((performance_history, new_data))
    performance_history = performance_history.tail(HISTORY_SIZE * NUM_SAMPLES)
    performance_history.to_csv(history_file, index=False)
    return performance_history


def render_history_table(performance_history):
    """Display benchmark history as text on the command line."""
    print('\nComparing to previous results:')
    prev = None
    # Print one row for each run of this script recorded in the history.
    for sample_id in performance_history['SampleId'].unique():
        # Lookup all the samples from a previous run and find their average.
        data = performance_history[
            performance_history['SampleId'] == sample_id]
        timestamp = data['Timestamp'].values[0]
        mean = data['klps'].mean()
        std = data['klps'].std()

        # Compare the average performance of this one to the previous one.
        from_last = ''
        if prev is not None:
            percent = 100 * (prev - mean) / mean
            from_last = f', {percent:+0.2f}% from last'

        # Add one line to the data table summarizing this run.
        print(f' - {datetime.fromtimestamp(timestamp):%b %d %I:%M%p}: '
              f'{mean:0.2f} ±{std:0.2f} klps{from_last}')
        prev = mean


def render_history_chart(performance_history):
    """Display benchmark history as a Seaborn chart."""
    def fmt(timestamps):
        return [
            f'{datetime.fromtimestamp(timestamp):%b %d\n%I:%M%p}'
            for timestamp in timestamps
        ]

    sns.set_theme()
    ax = sns.lineplot(
        data=performance_history, x='SampleId', y='klps',
        estimator='median')
    ax.set_ylim(ymin=0, ymax=1.1*performance_history['klps'].max())
    ax.set_xticks(
        ticks=performance_history['SampleId'].unique(),
        labels=fmt(performance_history['Timestamp'].unique()))
    ax.set(xlabel=None)
    plt.tight_layout()
    plt.show()


def main():
    """Run a test program, measure its speed, and compare to past runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--chart', action='store_true',
        help='Display a chart of past performance in a popup window')
    args = parser.parse_args()

    # Actually run the benchmark.
    performance_samples = collect_samples()
    performance_history = update_history(performance_samples)

    # Print results in a table, and maybe popup a chart visualization.
    render_history_table(performance_history)
    if args.chart:
        render_history_chart(performance_history)


if __name__ == '__main__':
    main()
