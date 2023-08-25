import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import experiments
import gif_files
from kernel import simulate_organism, FitnessGoal, Simulator


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
    simulator = Simulator(num_species, 1, experiments.NUM_ORGANISMS)
    simulator.populate(np.array([
        species_data.phenotype_program.serialize()
        for species_data in species_list]))
    # TODO: Maybe use a no-op fitness goal, since it doesn't matter here?
    videos = simulator.simulate_and_record(FitnessGoal.STILL_LIFE)
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

    # Save the genotype and video for the best organism from this species.
    best_organism = species_data.best_organism
    np.save(
        species_path.joinpath('best_organism_genotype.npy'),
        best_organism.genotype)
    gif_files.save_image(
        simulate_organism(
            species_data.phenotype_program.serialize(),
            best_organism.genotype),
        species_path.joinpath(
            f'best_organism_f{best_organism.fitness}.gif'))


def recursive_delete_directory(path):
    for file in path.glob('*'):
        if file.is_dir():
            recursive_delete_directory(file)
        else:
            file.unlink()
    path.rmdir()


def summarize_experiment_data(experiment):
    # Go through all files in the output directory for this experiment and all
    # of its subdirectories and delete any generated output (that is,
    # everything but the raw experiment data in state.pickle)
    for file in experiment.path.glob('*'):
        if file.is_dir():
            recursive_delete_directory(file)
        # TODO: use path objects in experiment.
        elif file.name != 'state.pickle' and file.name != 'results.pickle':
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
    random_populations = get_random_population(
        experiment_data.best_species_per_trial)
    best_fitness = 0
    best_video = None
    for trial in range(experiment.trial + 1):
        # Generate output for this species
        species_data = experiment_data.best_species_per_trial[trial]
        species_path = experiment.path.joinpath(
            f'best_species_from_trial_{trial:d}'
            f'_f{species_data.fitness}')
        species_path.mkdir()
        summarize_species_data(species_data, species_path)

        # Save the best video for each trial, and track best overall
        best_organism = species_data.best_organism
        video = simulate_organism(
            species_data.phenotype_program.serialize(),
            best_organism.genotype)
        fitness = best_organism.fitness
        gif_files.save_image(
            video,
            experiment.path.joinpath(
                f'best_organism_from_trial_{trial}_f{fitness}.gif'))
        # TODO: compare organisms directly, or remove comparison support in the
        # OrganismData class.
        if fitness > best_fitness:
            best_fitness = fitness
            best_video = video

        # Save a sample random population to visualize this species.
        population_path = species_path.joinpath('random_initial_population')
        population_path.mkdir()
        for index, video in enumerate(random_populations[trial]):
            gif_files.save_image(
                video, population_path.joinpath(f'sample_{index:02d}.gif'))

    # Save the overall best organism found for this experiment.
    gif_files.save_image(
        best_video,
        experiment.path.parent.joinpath(
            f'{experiment.name}_f{best_fitness}.gif'))


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
            experiment.path.mkdir(exist_ok=True)
            summarize_experiment_data(experiment)
    # TODO: If all experiments have at least one trial done, generate charts
    # comparing performance between experiments.


if __name__ == '__main__':
    main()
