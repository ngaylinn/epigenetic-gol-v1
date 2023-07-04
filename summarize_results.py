import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import experiments
import gif_files
import kernel


SPECIES_INDEX_COLUMNS = (
    np.repeat(
        np.arange(experiments.NUM_TRIALS),
        experiments.NUM_SPECIES_GENERATIONS),
    np.tile(
        np.arange(experiments.NUM_SPECIES_GENERATIONS),
        experiments.NUM_TRIALS))

ORGANISM_INDEX_COLUMNS = (
    np.repeat(
        np.arange(experiments.NUM_TRIALS),
        experiments.NUM_ORGANISM_GENERATIONS),
    np.tile(
        np.arange(experiments.NUM_ORGANISM_GENERATIONS),
        experiments.NUM_TRIALS))


def get_random_population(species_list):
    num_species = len(species_list)
    simulator = kernel.Simulator(num_species, 1, experiments.NUM_ORGANISMS)
    simulator.populate(np.array([
        species_data.phenotype_program.serialize()
        for species_data in species_list]))
    # TODO: NONE fitness goal? Fitness doesn't matter.
    videos = simulator.simulate_and_record(kernel.FitnessGoal.STILL_LIFE)
    # Get rid of the pointless trial index, since we only ran one trial.
    return videos[:, 0]


def summarize_species_data(species_data, species_path):
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

    # Save PhenotypeProgram in human and Python readable formats
    np.save(species_path.joinpath('phenotype_program.npy'),
            species_data.phenotype_program.serialize())
    with open(species_path.joinpath('phenotype_program.txt'), 'w') as file:
        file.write(str(species_data.phenotype_program))

    # Output the samples of the best organisms at multiple points in species
    # evolution across all trials.
    for generation, samples_by_trial in species_data.sample_organisms.items():
        sample_path = species_path.joinpath(
            f'best_organisms_from_gen{generation:03d}')
        sample_path.mkdir()
        for trial, organism_data in enumerate(samples_by_trial):
            np.save(
                sample_path.joinpath(f'genotype_from_trial{trial:d}.npy'),
                organism_data.genotype)
            gif_files.save_image(
                kernel.simulate_organism(
                    species_data.phenotype_program.serialize(),
                    organism_data.genotype),
                sample_path.joinpath(
                    f'simulation_from_trial{trial:d}'
                    f'_f{organism_data.fitness_scores[-1]}.gif'))


def recursive_delete_directory(path):
    for file in path.glob('*'):
        if file.is_dir():
            recursive_delete_directory(file)
        else:
            file.unlink()
    path.rmdir()


def summarize_experiment_data(experiment_data, experiment_path):
    # Go through all files in the output directory for this experiment and all
    # of its subdirectories and delete any generated output (that is,
    # everything but the raw experiment data in state.pickle)
    for file in experiment_path.glob('*'):
        if file.is_dir():
            recursive_delete_directory(file)
        elif file.name != 'state.pickle':
            file.unlink()

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
    fig.savefig(experiment_path.joinpath('species_fitness.svg'))
    plt.close()

    # Summarize experiment results for the best species evolved in each trial,
    # including a sample initial population.
    random_populations = get_random_population(
        experiment_data.best_species_per_trial)
    for trial in range(experiment_data.trial + 1):
        species_data = experiment_data.best_species_per_trial[trial]
        species_path = experiment_path.joinpath(
            f'best_species_from_trial_{trial:d}'
            f'_f{species_data.fitness_scores[-1]}')
        species_path.mkdir()
        summarize_species_data(species_data, species_path)
        population_path = species_path.joinpath('random_initial_population')
        population_path.mkdir()
        for index, video in enumerate(random_populations[trial]):
            gif_files.save_image(
                video, population_path.joinpath(f'sample_{index:02d}.gif'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rebuild', action='store_true',
        help='Regenerate all outputs, instead of just the outdated ones')
    args = parser.parse_args()
    for experiment in experiments.experiment_list:
        if experiment.state_path.exists():
            data_update_time = experiment.state_path.stat().st_mtime
        else:
            data_update_time = -1
        summary_path = experiment.state_path.with_name('species_fitness.svg')
        if summary_path.exists():
            summary_update_time = summary_path.stat().st_mtime
        else:
            summary_update_time = -1
        # If the summary data is out of date or the user requested a full
        # rebuild from the command line.
        if data_update_time > summary_update_time or args.rebuild:
            experiment_path = experiment.state_path.parent
            experiment_path.mkdir(exist_ok=True)
            summarize_experiment_data(experiment, experiment_path)
    # TODO: If all experiments have at least one trial done, generate charts
    # comparing performance between experiments.


if __name__ == '__main__':
    main()
