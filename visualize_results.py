"""Render charts and capture sample Videos from experiment runs.

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

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evolution import (
    Clade, NUM_TRIALS, NUM_ORGANISMS, NUM_SPECIES_GENERATIONS,
    NUM_ORGANISM_GENERATIONS)
from experiments import experiment_list, control_list
from phenotype_program import Constraints
import gif_files
from kernel import (
    simulate_organism, FitnessGoal, PhenotypeProgramDType, Simulator)


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


def render_random_populations(programs):
    """Record simulation Videos from random, unevolved populations.

    This returns NUM_ORGANISMS simulations for each PhenotypeProgram in
    programs.
    """
    # Simulate just one trial with the given set of species.
    simulator = Simulator(len(programs), 1, NUM_ORGANISMS)
    simulator.populate(programs)
    # TODO: It would be slightly nicer use a no-op FitnessGoal, since the
    # fitness function doesn't actually contribute to the result.
    simulations = simulator.simulate_and_record(FitnessGoal.STILL_LIFE)
    # Get rid of the pointless trial index, since we only ran one trial.
    return simulations[:, 0]


def visualize_random_populations(videos, title, filename):
    """Export a collage image of random, unevolved populations.

    This chooses a set of 50 images from the first Frames in videos and
    composes them into a single image.
    """
    fig = plt.figure(title, figsize=(20, 10))
    fig.suptitle(title)
    if len(videos) > 50:
        videos = np.random.choice(videos, [50], replace=False)
    for index, video in enumerate(videos):
        axis = fig.add_subplot(5, 10, index + 1)
        axis.tick_params(bottom=False, left=False, labelbottom=False,
                         labelleft=False)
        axis.grid(False)
        gif_files.add_simulation_data_to_figure(video[0], fig, axis)
    fig.savefig(filename)
    plt.close()


def visualize_species_data(species_data, species_path, name):
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
    fig.set(title=f'Organism Fitness by Trial ({name})')
    fig.savefig(species_path.joinpath('organism_fitness.png'))
    plt.close()

    # Save this species' PhenotypeProgram in human and Python readable formats.
    np.save(species_path.joinpath('phenotype_program.npy'),
            species_data.phenotype_program.serialize())
    with open(species_path.joinpath('phenotype_program.txt'), 'w') as file:
        file.write(str(species_data.phenotype_program))

    # Save the Genotype and simulation for the best organism of this species.
    best_organism = species_data.best_organism
    np.save(
        species_path.joinpath('best_organism_genotype.npy'),
        best_organism.genotype)
    gif_files.save_simulation_data_as_image(
        simulate_organism(
            species_data.phenotype_program.serialize(),
            best_organism.genotype),
        species_path.joinpath(
            f'best_organism_f{best_organism.fitness}.gif'))


def visualize_control_data(control):
    # Summarize this species
    species_data = control.load_from_filesystem()
    phenotype_program = species_data.phenotype_program.serialize()
    visualize_species_data(species_data, control.path, control.name)

    random_populations = render_random_populations([phenotype_program])
    visualize_random_populations(
        random_populations[0],
        f'Random Initial Population ({control.name})',
        control.path.joinpath('random_initial_population.png'))

    # Copy the best organism to the experiment directory.
    best_organism = species_data.best_organism
    best_simulation = simulate_organism(phenotype_program, best_organism.genotype)
    gif_files.save_simulation_data_as_image(
        best_simulation,
        control.path.parent.joinpath(
            f'{control.name}_f{best_organism.fitness}.gif'))


def recursive_delete_directory(path):
    """Delete all files and subdirectories under path, then delete path."""
    for file in path.glob('*'):
        if file.is_dir():
            recursive_delete_directory(file)
        else:
            file.unlink()
    path.rmdir()


def delete_visualizations(experiment):
    """Delete generated dataviz files associated with the given experiment."""
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

    # For each experiment, a gif of the best simulation from that experiment
    # is stored adjacent to the experiment directory. Clear those out, too.
    for file in experiment.path.parent.glob(f'{experiment.name}*gif'):
        file.unlink()


def visualize_experiment_data(experiment):
    """Summarize the results for a single experiment."""
    # Generate a chart summarizing species fitness across trials
    experiment_data = experiment.get_results()
    species_fitness_by_trial = pd.DataFrame(
        np.column_stack(
            (*SPECIES_INDEX_COLUMNS,
             experiment_data.all_trial_species_fitness.flatten())),
        columns=['Trial', 'Generation', 'Fitness'])
    fig = sns.relplot(
        data=species_fitness_by_trial, kind='line',
        x='Generation', y='Fitness', hue='Trial')
    fig.set(title=f'Species Fitness by Trial ({experiment.name})')
    fig.savefig(experiment.path.joinpath('species_fitness.png'))
    plt.close()

    # Summarize experiment results for the best species evolved in each trial,
    # including a sample initial population.
    random_populations = render_random_populations(
        np.fromiter(
            (species_data.phenotype_program.serialize()
             for species_data in experiment_data.best_species_per_trial),
            dtype=PhenotypeProgramDType))
    best_fitness = 0
    best_simulation = None
    for trial in range(experiment.trial + 1):
        # Generate output for this species
        species_data = experiment_data.best_species_per_trial[trial]
        species_path = experiment.path.joinpath(
            f'best_species_from_trial_{trial:d}'
            f'_f{species_data.fitness}')
        species_path.mkdir()
        visualize_species_data(species_data, species_path, experiment.name)

        # Save the best simulation for each trial, and track the best overall.
        best_organism = species_data.best_organism
        simulation = simulate_organism(
            species_data.phenotype_program.serialize(),
            best_organism.genotype)
        fitness = best_organism.fitness
        gif_files.save_simulation_data_as_image(
            simulation,
            experiment.path.joinpath(
                f'best_organism_from_trial_{trial}_f{fitness}.gif'))
        if fitness > best_fitness:
            best_fitness = fitness
            best_simulation = simulation

        # Visualize a random population of organisms generated for this species
        # by putting the first Frame from all NUM_ORGANISMS Videos into a
        # single image.
        visualize_random_populations(
            random_populations[trial],
            f'Random Initial Population ({experiment.name}, Trial {trial:d})',
            species_path.joinpath('random_initial_population.png'))

    gif_files.save_simulation_data_as_image(
        best_simulation,
        experiment.path.parent.joinpath(
            f'{experiment.name}_f{best_fitness}.gif'))


def visualize_cross_experiment_comparisons():
    # Combine data from all experiments into one table.
    all_experiment_data = pd.DataFrame()
    for experiment in experiment_list:
        this_experiment_data = experiment.get_results()
        all_experiment_data = pd.concat((all_experiment_data, pd.DataFrame({
            'FitnessGoal': experiment.fitness_goal.name,
            'FitnessScore': this_experiment_data.all_trial_species_fitness[:, -1],
            'Bias': experiment.constraints.allow_bias,
            'Composition': experiment.constraints.allow_composition,
            'StampTransforms': experiment.constraints.allow_stamp_transforms,
        })))
    # Normalize fitness scores across all FitnessGoals.
    per_goal_max_scores = {
        goal: all_experiment_data.where(
            all_experiment_data['FitnessGoal'] == goal)['FitnessScore'].max()
        for goal in all_experiment_data['FitnessGoal'].unique()
    }
    max_scores = all_experiment_data['FitnessGoal'].map(per_goal_max_scores)
    all_experiment_data['FitnessScore'] /= max_scores

    plt.figure().suptitle('Impact of Bias on species fitness')
    ax = sns.boxplot(
        data=all_experiment_data, x='FitnessGoal', y='FitnessScore',
        hue='Bias', width=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_ylabel('FitnessScore (normalized)')
    plt.tight_layout()
    plt.savefig('output/experiments/configuration_bias.png')
    plt.close()

    plt.figure().suptitle('Impact of Composition on species fitness')
    ax = sns.boxplot(
        data=all_experiment_data, x='FitnessGoal', y='FitnessScore',
        hue='Composition', width=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_ylabel('FitnessScore (normalized)')
    plt.tight_layout()
    plt.savefig('output/experiments/configuration_composition.png')
    plt.close()

    plt.figure().suptitle('Impact of Stamp Transforms on species fitness')
    ax = sns.boxplot(
        data=all_experiment_data, x='FitnessGoal', y='FitnessScore',
        hue='StampTransforms', width=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_ylabel('FitnessScore (normalized)')
    plt.tight_layout()
    plt.savefig('output/experiments/configuration_stamp_transforms.png')
    plt.close()


def visualize_experiment_vs_control():
    best_experiment_species_by_goal = {}
    all_experiment_data = pd.DataFrame()
    for experiment in experiment_list:
        goal = experiment.fitness_goal.name
        this_experiment_data = experiment.get_results()
        species_data = this_experiment_data.best_species_per_trial[-1]
        prev_best, _, _ = best_experiment_species_by_goal.get(
            goal, (None, None, None))
        if (prev_best is None or
            prev_best.best_organism.fitness <
            species_data.best_organism.fitness):
            best_experiment_species_by_goal[goal] = (
                species_data, str(experiment.constraints),
                this_experiment_data)
        all_experiment_data = pd.concat((all_experiment_data, pd.DataFrame({
            'FitnessGoal': goal,
            'FitnessScore': species_data.all_trial_organism_fitness[:, -1],
            'Configuration': str(experiment.constraints),
            'Kind': 'Experiment',
            'Best': False,
        })))

    best_control_species_by_goal = {}
    for control in control_list:
        species_data = control.load_from_filesystem()
        goal = control.fitness_goal.name
        prev_best, _ = best_control_species_by_goal.get(goal, (None, None))
        if (prev_best is None or
            prev_best.best_organism.fitness <
            species_data.best_organism.fitness):
            best_control_species_by_goal[goal] = (
                species_data, control.config_str)
        all_experiment_data = pd.concat((all_experiment_data, pd.DataFrame({
            'FitnessGoal': goal,
            'FitnessScore': species_data.all_trial_organism_fitness[:, -1],
            'Configuration': control.config_str,
            'Kind': 'Control',
            'Best': False,
        })))
    goals = all_experiment_data['FitnessGoal'].unique()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    axes[0, 0].set(ylabel='First step')
    axes[1, 0].set(ylabel='Last step')
    for index, goal in enumerate(goals):
        best_expt_species, config, _ = best_experiment_species_by_goal[goal]
        best_expt_organism = best_expt_species.best_organism
        best_expt_simulation = simulate_organism(
            best_expt_species.phenotype_program.serialize(),
            best_expt_organism.genotype)

        axes[0, index].xaxis.set_label_position('top')
        axes[0, index].set(xlabel=config)
        axes[0, index].set_title(goal)

        axes[0, index].imshow(
            best_expt_simulation[0], **gif_files.FORMAT_OPTIONS)
        axes[0, index].grid(False)
        axes[0, index].tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False)
        axes[1, index].imshow(
            best_expt_simulation[-1], **gif_files.FORMAT_OPTIONS)
        axes[1, index].grid(False)
        axes[1, index].tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.suptitle('Best experimental phenoytpes summary')
    plt.tight_layout()
    plt.savefig(f'output/experiments/best_expt_phenotypes.png')
    plt.close()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    axes[0, 0].set(ylabel='First step')
    axes[1, 0].set(ylabel='Last step')
    for index, goal in enumerate(goals):
        best_ctrl_species, config = best_control_species_by_goal[goal]
        best_ctrl_organism = best_ctrl_species.best_organism
        best_ctrl_simulation = simulate_organism(
            best_ctrl_species.phenotype_program.serialize(),
            best_ctrl_organism.genotype)

        axes[0, index].xaxis.set_label_position('top')
        axes[0, index].set(xlabel=config)
        axes[0, index].set_title(goal)

        axes[0, index].imshow(
            best_ctrl_simulation[0], **gif_files.FORMAT_OPTIONS)
        axes[0, index].grid(False)
        axes[0, index].tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False)
        axes[1, index].imshow(
            best_ctrl_simulation[-1], **gif_files.FORMAT_OPTIONS)
        axes[1, index].grid(False)
        axes[1, index].tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.suptitle('Best control phenoytpes summary')
    plt.tight_layout()
    plt.savefig(f'output/experiments/best_ctrl_phenotypes.png')
    plt.close()

    for goal in goals:
        experiment_data = all_experiment_data.where(
            all_experiment_data['FitnessGoal'] == goal)
        plt.figure().suptitle(f'Best organism fitness per configuration ({goal})')
        ax = sns.boxplot(data=experiment_data, x='Configuration', y='FitnessScore')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
        plt.tight_layout()
        plt.savefig(f'output/experiments/best_organism_fitness_{goal}.png')
        plt.close()

        expt_species_data, config, expt_data = best_experiment_species_by_goal[goal]
        expt_organism_fitness_data = pd.DataFrame(
            np.column_stack(
                (*ORGANISM_INDEX_COLUMNS,
                 expt_species_data.all_trial_organism_fitness.flatten())),
            columns=['Trial', 'Generation', 'Fitness'])
        expt_organism_fitness_data['Configuration'] = config
        expt_organism_fitness_data['FitnessFor'] = 'Organism'

        expt_species_fitness_data = pd.DataFrame(
            np.column_stack(
                (*SPECIES_INDEX_COLUMNS,
                 expt_data.all_trial_species_fitness.flatten())),
            columns=['Trial', 'Generation', 'Fitness'])
        expt_species_fitness_data['Configuration'] = config
        expt_species_fitness_data['FitnessFor'] = 'Species'

        ctrl_species_data, config = best_control_species_by_goal[goal]
        ctrl_organism_fitness_data = pd.DataFrame(
            np.column_stack(
                (*ORGANISM_INDEX_COLUMNS,
                 ctrl_species_data.all_trial_organism_fitness.flatten())),
            columns=['Trial', 'Generation', 'Fitness'])
        ctrl_organism_fitness_data['Configuration'] = config
        ctrl_organism_fitness_data['FitnessFor'] = 'Organism'

        species_max_fitness = expt_species_fitness_data['Fitness'].max()
        organism_max_fitness = max(expt_organism_fitness_data['Fitness'].max(),
                                   ctrl_organism_fitness_data['Fitness'].max())
        expt_organism_fitness_data['Fitness'] /= organism_max_fitness
        expt_species_fitness_data['Fitness'] /= species_max_fitness
        ctrl_organism_fitness_data['Fitness'] /= organism_max_fitness

        combined_organism_fitness_data = pd.concat(
            (expt_species_fitness_data, expt_organism_fitness_data,
             ctrl_organism_fitness_data))

        facet_grid = sns.relplot(
            data=combined_organism_fitness_data, kind='line',
            col='FitnessFor', x='Generation', y='Fitness', hue='Configuration')
        facet_grid.figure.suptitle(f'Best species fitness curves ({goal})',
                                   y=1.05)
        facet_grid.savefig(f'output/experiments/{goal}_evolvability.png')
        plt.tight_layout()
        plt.close()

    for goal in goals:
        # Find and mark the experiment with the best organism for this goal.
        best_index = all_experiment_data.where(
            (all_experiment_data['Kind'] == 'Experiment') &
            (all_experiment_data['FitnessGoal'] == goal)
        )['FitnessScore'].argmax()
        all_experiment_data.iloc[
            best_index, all_experiment_data.columns.get_loc('Best')] = True

        # Do the same for the controls.
        best_index = all_experiment_data.where(
            (all_experiment_data['Kind'] == 'Control') &
            (all_experiment_data['FitnessGoal'] == goal)
        )['FitnessScore'].argmax()
        all_experiment_data.iloc[
            best_index, all_experiment_data.columns.get_loc('Best')] = True

    # Normalize fitness scores across all FitnessGoals.
    per_goal_max_scores = {
        goal: all_experiment_data.where(
            all_experiment_data['FitnessGoal'] == goal)['FitnessScore'].max()
        for goal in all_experiment_data['FitnessGoal'].unique()
    }
    max_scores = all_experiment_data['FitnessGoal'].map(per_goal_max_scores)
    all_experiment_data['FitnessScore'] /= max_scores

    plt.figure().suptitle(f'Best organism fitness per FitnessGoal')
    experiment_data = all_experiment_data.where(all_experiment_data['Best'])
    ax = sns.barplot(
        data=experiment_data, x='FitnessGoal', y='FitnessScore',
        hue='Kind', width=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_ylabel('Best fitness score (normalized)')
    plt.tight_layout()
    plt.savefig('output/experiments/best_organism_fitness_all_goals.png')
    plt.close()



def visualize_species_range():
    filename = 'output/random_initial_population.png'
    if not Path(filename).exists():
        programs = Clade.make_random_species(50)
        videos = render_random_populations(
            np.fromiter(
                (program.serialize() for program in programs),
                dtype=PhenotypeProgramDType))
        title = 'Random Initial Population from Random Species'
        visualize_random_populations(videos, title, filename)


def visualize_results():
    """Look for new experiment data and generate visualizations for it."""
    sns.set_style('darkgrid')

    # Parse command line arguments.
    parser = ArgumentParser()
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
        summary_path = experiment.state_path.with_name('species_fitness.png')
        if summary_path.exists():
            summary_update_time = summary_path.stat().st_mtime
        else:
            summary_update_time = -1

        # If this experiment has data that hasn't been visualized yet, or the
        # user requested a full rebuild, delete any existing visualizations and
        # then generate new ones for this experiment.
        if data_update_time > summary_update_time or args.rebuild:
            experiment.path.mkdir(exist_ok=True)
            delete_visualizations(experiment)
            visualize_experiment_data(experiment)

    # If we haven't yet run at least one trial for each experiment, stop here
    # and don't bother summarizing cross-experiment comparisons.
    if all(experiment.has_started() for experiment in experiment_list):
        visualize_cross_experiment_comparisons()

    # Now visualize all the controls.
    for control in control_list:
        if control.results_path.exists():
            data_update_time = control.results_path.stat().st_mtime
        else:
            data_update_time = -1

        summary_path = control.results_path.with_name('organism_fitness.png')
        if summary_path.exists():
            summary_update_time = summary_path.stat().st_mtime
        else:
            summary_update_time = -1

        if data_update_time > summary_update_time or args.rebuild:
            visualize_control_data(control)

    # If all the experiments and controls have run, compare them.
    if (all(experiment.has_finished() for experiment in experiment_list) and
        all(control.has_finished() for control in control_list)):
        visualize_experiment_vs_control()

    # Visualize the range of species for a random batch of species.
    visualize_species_range()


if __name__ == '__main__':
    visualize_results()
