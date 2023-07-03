"""Measure performance of the kernel module and compare to prior runs.

This project is split into two parts: the inner loop is an optimized "kernel"
written in C++, while the rest of the program is written in Python for ease of
development and readability. As such, one goal is to get the kernel to run
quickly and reliably independent of the rest of the code. This is a simple tool
to help with that.

This script runs this project's inner loop a few times and measures how long it
takes. It displays the results as text on the command line, and optionally as a
chart visualization.
"""

from datetime import datetime
import os.path
import sys
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import kernel
import phenotype_program

NUM_SPECIES = 50
NUM_TRIALS = 5
NUM_ORGANISMS = 50
NUM_GENERATIONS = 200
# The total number of organisms to be run in one batch by the kernel.
POPULATION_SIZE = NUM_SPECIES * NUM_TRIALS * NUM_ORGANISMS
NUM_LIFETIMES = POPULATION_SIZE * NUM_GENERATIONS

SAMPLE_SIZE = 3
NUM_SAMPLES = 5


# For testing, just make NUM_SPECIES copies of the same PhenotypeProgram.
CLADE = phenotype_program.Clade(NUM_SPECIES, testing=True)


def sample_performance():
    """Measure the performance of running this project's inner loop once."""
    # Don't count the one-time initialization in our measurement.
    random.seed(42)
    simulator = kernel.Simulator(NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
    goal = kernel.FitnessGoal.STILL_LIFE

    start = time.perf_counter()
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
    """Collect SAMPLE_SIZE performance samples and sanity check them."""

    # Performance doesn't count if the results are wrong, so we keep track of
    # what the "right" fitness scores should be in a file on disk and assert we
    # get the same results every time.
    fitness_file = 'output/benchmark/fitness.npy'
    if os.path.exists(fitness_file):
        expected_fitness = np.load(fitness_file)
    else:
        expected_fitness = None

    # Actually collect some new performance samples.
    print(f'Collection {SAMPLE_SIZE} samples of {NUM_LIFETIMES} lifetimes:')
    performance_samples = []
    for sample in range(SAMPLE_SIZE):
        fitness, elapsed = sample_performance()
        if expected_fitness is None:
            expected_fitness = fitness
        elif not np.array_equal(expected_fitness, fitness):
            print('Warning: inconsistent fitness! Results may be invalid.')
        klps = NUM_LIFETIMES / elapsed / 1000
        performance_samples.append(klps)
        print(f' - Sample {sample}: {elapsed:.2f}s {klps:.2f}klps')
    np.save(fitness_file, expected_fitness)
    return performance_samples


def update_history(performance_samples):
    """Combine a new set of samples with historical data."""
    # This script tracks performance over the last NUM_SAMPLES runs in a file
    # on disk. Try loading that historical data if its available.
    history_file = 'output/benchmark/history.csv'
    next_sample_id = 0
    if os.path.exists(history_file):
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
    performance_history = performance_history.tail(NUM_SAMPLES * SAMPLE_SIZE)
    performance_history.to_csv(history_file, index=False)
    return performance_history


def render_history_table(performance_history):
    """Display benchmark history as text on the command line."""
    print('\nComparing to previous results:')
    prev = None
    for sample_id in performance_history['SampleId'].unique():
        data = performance_history[
            performance_history['SampleId'] == sample_id]
        timestamp = data['Timestamp'].values[0]
        mean = data['klps'].mean()
        std = data['klps'].std()
        from_last = ''
        if prev is not None:
            percent = 100 * (prev - mean) / mean
            from_last = f', {percent:+0.2f}% from last'
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
    performance_samples = collect_samples()
    performance_history = update_history(performance_samples)

    render_history_table(performance_history)
    if 'chart' in sys.argv:
        render_history_chart(performance_history)


if __name__ == '__main__':
    main()
