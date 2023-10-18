"""Render charts and capture sample videos from experiment runs.

This script looks for experiment results in the output directory and generates
data visualizations for each one found. By default, this script skips over any
experiments that already have visualizations generated since the last run. To
force a rebuild of visualizations from all experiment runs, use the --rebuild
argument on the command line.

Nromally, the run_experiments script will run this script automatically to
visualize result data as it gets generated. The only reason to run this script
independently is to iterate on how experiment data gets visualized without
re-running the experiments (which can be quite time consuming).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments import (
    NUM_TRIALS, NUM_SPECIES_GENERATIONS, NUM_ORGANISM_GENERATIONS,
    experiment_list)
from gif_files import save_simulation_data_as_image
from kernel import simulate_organism, FitnessGoal, Simulator


# Used to transform experiment data from a wide-format (one column for each
# generation) to long-format (one row for each generation). This is useful for
# rendering labeled charts using Pandas and Seaborn.
SPECIES_INDEX_COLUMNS = (
    np.repeat(np.arange(NUM_TRIALS), NUM_SPECIES_GENERATIONS),
    np.tile(np.arange(NUM_SPECIES_GENERATIONS), NUM_TRIALS))

# Used to transform experiment data from a wide-format (one column for each
# generation) to long-format (one row for each generation). This is useful for
# rendering labeled charts using Pandas and Seaborn.
ORGANISM_INDEX_COLUMNS = (
    np.repeat(np.arange(NUM_TRIALS), NUM_ORGANISM_GENERATIONS),
    np.tile(np.arange(NUM_ORGANISM_GENERATIONS), NUM_TRIALS))


def render_random_populations(species_list):
    """Record simulation videos from random, unevolved populations.

    This returns NUM_ORGANISMS simulations for each species in species_list.
    """
    # Simulate just one trial with the given set of species.
    simulator = Simulator(len(species_list), 1, NUM_ORGANISMS)
    simulator.populate(np.array([
        species_data.phenotype_program.serialize()
        for species_data in species_list]))
    # TODO: It would be slightly nicer use a no-op fitness goal, since the
    # fitness function doesn't actually contribute to the result.
    simulations = simulator.simulate_and_record(FitnessGoal.STILL_LIFE)
    # Get rid of the pointless trial index, since we only ran one trial.
    return simulations[:, 0]


def visualize_species_data(species_data, species_path):
    """Summarize the results for a single evolved species."""
    # Save a chart of organism fitness across all trials of this species.
    organism_fitness_by_trial = pd.DataFrame(
        np.column_stack(
            (*ORGANISM_INDEX_COLUMNS,
             species_data.all_trial_organism_fitness.flatten())),
        columns=['Trial', 'Generation', 'Fitness'])
    fig = sns.relplot(
        data=organism_fitness_by_trial, kind='line',
        x='Generation', y='Fitness', hue='Trial')
    fig.set(title='Organism Fitness by Trial')
    fig.savefig(species_path.joinpath('organism_fitness.svg'))
    plt.close()

    # Save this species' PhenotypeProgram in human and Python readable formats.
    np.save(species_path.joinpath('phenotype_program.npy'),
            species_data.phenotype_program.serialize())
    with open(species_path.joinpath('phenotype_program.txt'), 'w') as file:
        file.write(str(species_data.phenotype_program))

    # Save the genotype and video for the best organism of this species.
    best_organism = species_data.best_organism
    np.save(
        species_path.joinpath('best_organism_genotype.npy'),
        best_organism.genotype)
    save_simulation_data_as_image(
        simulate_organism(
            species_data.phenotype_program.serialize(),
            best_organism.genotype),
        species_path.joinpath(
            f'best_organism_f{best_organism.fitness}.gif'))


def recursive_delete_directory(path):
    """Delete all files and subdirectories under path, then delete path."""
    for file in path.glob('*'):
        if file.is_dir():
            recursive_delete_directory(file)
        else:
            file.unlink()
    path.rmdir()


def visualize_experiment_data(experiment):
    """Summarize the results for a single experiment."""
    # Go through all files in the output directory for this experiment and all
    # of its subdirectories and delete any generated output (that is,
    # everything but the raw experiment data)
    experiment_files = [
        experiment.results_path.resolve(),
        experiment.state_path.resolve()]
    for file in experiment.path.glob('*'):
        if file.is_dir():
            recursive_delete_directory(file)
        elif file.resolve() not in experiment_files:
            file.unlink()

    experiment_data = experiment.get_results()

    # Generate a chart summarizing species fitness across trials
    species_fitness_by_trial = pd.DataFrame(
        np.column_stack(
            (*SPECIES_INDEX_COLUMNS,
             experiment_data.all_trial_species_fitness.flatten())),
        columns=['Trial', 'Generation', 'Fitness'])
    fig = sns.relplot(
        data=species_fitness_by_trial, kind='line',
        x='Generation', y='Fitness', hue='Trial')
    fig.set(title='Species Fitness by Trial')
    fig.savefig(experiment.path.joinpath('species_fitness.svg'))
    plt.close()

    # Summarize experiment results for the best species evolved in each trial,
    # including a sample initial population.
    random_populations = render_random_populations(
        experiment_data.best_species_per_trial)
    best_fitness = 0
    best_simulation = None
    for trial in range(experiment.trial + 1):
        # Generate output for this species
        species_data = experiment_data.best_species_per_trial[trial]
        species_path = experiment.path.joinpath(
            f'best_species_from_trial_{trial:d}'
            f'_f{species_data.fitness}')
        species_path.mkdir()
        visualize_species_data(species_data, species_path)

        # Save the best simulation for each trial, and track the best overall.
        best_organism = species_data.best_organism
        simulation = simulate_organism(
            species_data.phenotype_program.serialize(),
            best_organism.genotype)
        fitness = best_organism.fitness
        save_simulation_data_as_image(
            simulation,
            experiment.path.joinpath(
                f'best_organism_from_trial_{trial}_f{fitness}.gif'))
        if fitness > best_fitness:
            best_fitness = fitness
            best_simulation = simulation

        # Save a sample random population to visualize this species.
        population_path = species_path.joinpath('random_initial_population')
        population_path.mkdir()
        for index, video in enumerate(random_populations[trial]):
            save_simulation_data_as_image(
                video, population_path.joinpath(f'sample_{index:02d}.gif'))

    # Save the overall best simulation found for this experiment.
    save_simulation_data_as_image(
        best_simulation,
        experiment.path.parent.joinpath(
            f'{experiment.name}_f{best_fitness}.gif'))


def visualize_results():
    """Look for new experiment data and generate visualizations for it."""
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rebuild', action='store_true',
        help='Regenerate all outputs, instead of just the outdated ones')
    args = parser.parse_args()

    # For all the experiments run by this project...
    for experiment in experiment_list:
        # If this experiment has been started, there should be state data saved
        # to the filesystem. Check the last modified time.
        if experiment.state_path.exists():
            data_update_time = experiment.state_path.stat().st_mtime
        else:
            data_update_time = -1

        # Check for one of the experiment data visualizations and find its last
        # modified time. This assumes all visualizations for an experiment are
        # generated at once, and can't be out of sync with each other.
        summary_path = experiment.state_path.with_name('species_fitness.svg')
        if summary_path.exists():
            summary_update_time = summary_path.stat().st_mtime
        else:
            summary_update_time = -1

        # If this experiment has data that hasn't been visualized yet, or the
        # user requested a full rebuild, then generate visualizations for this
        # experiment.
        if data_update_time > summary_update_time or args.rebuild:
            experiment.path.mkdir(exist_ok=True)
            visualize_experiment_data(experiment)
    # TODO: If all experiments have at least one trial done, generate charts
    # comparing performance across experiments.


if __name__ == '__main__':
    visualize_results()
