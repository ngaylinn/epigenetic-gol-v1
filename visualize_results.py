"""Render charts and capture sample Videos from experiment runs.

This script looks for experiment results in the output directory and generates
data visualizations for each one found. By default, this script skips over any
experiments that already have visualizations generated since the last run. To
force a rebuild of visualizations from all experiment runs, use the --rebuild
argument on the command line.

Normally, the run_experiments script will run this script automatically to
visualize result data as it gets generated. The only reason to run this script
independently is to iterate on how experiment data gets visualized without
re-running the experiments (which can be quite time consuming).
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, false_discovery_control
import seaborn as sns

from evolution import (
    Clade, NUM_TRIALS, NUM_ORGANISMS, NUM_SPECIES_GENERATIONS,
    NUM_ORGANISM_GENERATIONS)
from experiments import experiment_list, control_list
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


def render_image_grid(images, rows, cols):
    """Layout GOL simulation images into a 2D grid of rows and cols."""
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    for index in range(rows * cols):
        axis = axes[divmod(index, cols)]
        gif_files.add_simulation_data_to_figure(images[index], fig, axis)
    return fig, axes


def visualize_species_data(species_data, species_path, name):
    """Summarize the results for a single evolved species."""
    # Save a chart of organism fitness across all trials of this species.
    organism_fitness_by_trial = pd.DataFrame(
        np.column_stack(
            (*ORGANISM_INDEX_COLUMNS,
             species_data.best_organism_fitness_history().flatten())),
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
    """Summarize the results for a single control."""
    # Summarize this species
    species_data = control.load_from_filesystem()
    phenotype_program = species_data.phenotype_program.serialize()
    visualize_species_data(species_data, control.path, control.name)

    random_populations = render_random_populations([phenotype_program])
    # 5x10 view showing the full population.
    fig, axes = render_image_grid(random_populations[0, :, 0], rows=5, cols=10)
    # 2x4 view for paper
    # fig, axes = render_image_grid(random_populations[0, :, 0], rows=2, cols=4)
    fig.suptitle(f'Random Initial Population ({control.name})')
    plt.tight_layout()
    fig.savefig(control.path.joinpath('random_initial_population.png'))
    plt.close()

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
             np.array(experiment_data.all_trial_species_fitness).flatten())),
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
    best_fitness = -1
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
        # by putting the first Frame from a sampling of Videos into a single
        # image.
        # 5x10 view showing the full population.
        fig, _ = render_image_grid(
            random_populations[trial, :, 0], rows=5, cols=10)
        # 2x4 view for paper
        # fig, _ = render_image_grid(
        #     random_populations[trial, :, 0], rows=2, cols=4)
        plt.tight_layout()
        fig.savefig(species_path.joinpath('random_initial_population.png'))
        plt.close()

    gif_files.save_simulation_data_as_image(
        best_simulation,
        experiment.path.parent.joinpath(
            f'{experiment.name}_f{best_fitness}.gif'))


def visualize_cross_experiment_comparisons():
    """Compare experiment performance across configurations for each goal."""
    # Combine data from all experiments into one table.
    all_experiment_data = pd.DataFrame()
    for experiment in experiment_list:
        this_experiment_data = experiment.get_results()
        all_experiment_data = pd.concat((all_experiment_data, pd.DataFrame({
            'FitnessGoal': experiment.fitness_goal.name,
            'FitnessScore': [trial_fitness[-1] for trial_fitness in
                this_experiment_data.all_trial_species_fitness],
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


def visualize_best_phenotypes(goals, species_by_goal):
    """Compare best phenotypes for each goal."""
    # Simulate the best phenotype for each goal, then record just the state in
    # the first and last steps.
    images = [None] * 2 * len(goals)
    for index, goal in enumerate(goals):
        best_species, config, _ = species_by_goal[goal]
        best_organism = best_species.best_organism
        best_simulation = simulate_organism(
            best_species.phenotype_program.serialize(),
            best_organism.genotype)
        images[index] = best_simulation[0]
        images[index + len(goals)] = best_simulation[-1]

    # Render the first and last images for each goal, which is a good way to
    # see how well the phenotype did at a glance.
    _, axes = render_image_grid(images, rows=2, cols=8)
    axes[0, 0].set(ylabel='First step')
    axes[1, 0].set(ylabel='Last step')
    for index, goal in enumerate(goals):
        _, config, _ = species_by_goal[goal]
        axes[0, index].xaxis.set_label_position('top')
        axes[0, index].set(xlabel=config)
        axes[0, index].set_title(goal)
    plt.tight_layout()


def visualize_evolvability(goals,
                           first_gen_best_organism_fitness_by_goal,
                           best_expt_species_by_goal,
                           best_ctrl_species_by_goal):
    """Compare expt & ctrl performance across species generations."""
    # Produce a separate set of visualization for each FitnessGoal.
    for goal in goals:
        # Visualize species fitness growth over generations.
        expt_species_data, config, expt_data = best_expt_species_by_goal[goal]
        expt_species_fitness_data = pd.DataFrame(
            np.column_stack(
                (*SPECIES_INDEX_COLUMNS,
                 np.array(expt_data.all_trial_species_fitness).flatten())),
            columns=['Trial', 'Generation', 'Fitness'])
        facet_grid = sns.relplot(
            data=expt_species_fitness_data, kind='line',
            x='Generation', y='Fitness')
        facet_grid.figure.set_size_inches(4, 4)
        plt.suptitle(f'Outer loop best fitness ({goal}, all trials)')
        plt.tight_layout()
        plt.savefig(f'output/experiments/{goal}/species_fitness.png')
        plt.close()

        # Organize and merge all fitness data into one big DataFrame
        expt_species_data, config, expt_data = best_expt_species_by_goal[goal]
        best_organism_fitness = expt_species_data.all_trial_organism_fitness.max(
            axis=1)
        expt_organism_fitness_data = pd.DataFrame(
            np.column_stack(
                (*ORGANISM_INDEX_COLUMNS, best_organism_fitness.flatten())),
            columns=['Trial', 'Generation', 'Fitness'])
        expt_organism_fitness_data['Configuration'] = 'Expt (evolved GP map)'

        first_organism_fitness_data = pd.DataFrame(
            np.column_stack(
                (*ORGANISM_INDEX_COLUMNS,
                 first_gen_best_organism_fitness_by_goal[goal].flatten())),
            columns=['Trial', 'Generation', 'Fitness'])
        first_organism_fitness_data['Configuration'] = 'Expt (unevolved GP map)'

        ctrl_species_data, config, _ = best_ctrl_species_by_goal[goal]
        ctrl_organism_fitness_data = pd.DataFrame(
            np.column_stack(
                (*ORGANISM_INDEX_COLUMNS,
                 ctrl_species_data.best_organism_fitness_history().flatten())),
            columns=['Trial', 'Generation', 'Fitness'])
        ctrl_organism_fitness_data['Configuration'] = config

        combined_organism_fitness_data = pd.concat(
            (expt_organism_fitness_data,
             first_organism_fitness_data,
             ctrl_organism_fitness_data))

        # Compute p-values comparing the organism fitness trajectory curves.
        eu = first_organism_fitness_data.where(
            first_organism_fitness_data['Generation'] == 149
        )['Fitness'].dropna()
        ee = expt_organism_fitness_data.where(
            expt_organism_fitness_data['Generation'] == 149
        )['Fitness'].dropna()
        c = ctrl_organism_fitness_data.where(
            ctrl_organism_fitness_data['Generation'] == 149
        )['Fitness'].dropna()

        print(f'Organsim fitness, experiment v. control ({goal}):')
        print('kruskal-walace', kruskal(eu, ee, c).pvalue)
        print('evolved experiment v. control', mannwhitneyu(ee, c).pvalue)
        print('unevolved experiment v. control', mannwhitneyu(eu, c).pvalue)
        print('evolved experiment v. unevolved experiment',
              mannwhitneyu(ee, eu).pvalue)
        print()

        # Render a plot of organism fitness trajectories for the control and
        # experiment (before and after evolving species).
        facet_grid = sns.relplot(
            data=combined_organism_fitness_data, kind='line', legend='brief',
            x='Generation', y='Fitness', hue='Configuration')
        facet_grid.legend.remove()
        facet_grid.figure.set_size_inches(4, 4)
        plt.tight_layout()
        ax = facet_grid.axes[0, 0]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.7])

        plt.legend(loc='best', bbox_to_anchor=(1.1, -0.2), ncol=2,
                   frameon=False)
        plt.suptitle(f'Inner loop best fitness ({goal}, all trials)')
        plt.savefig(f'output/experiments/{goal}/organism_fitness.png')
        plt.close()


def visualize_per_goal_fitness(goals, experiment_data):
    """Compare best phenotype fitness for each goal, experiment vs. control."""
    # Each FitnessGoal has its own score scale, so to compare between goals we
    # must first normalize so the max score for each goal is 1.0.
    per_goal_max_scores = {
        goal: experiment_data.where(
            experiment_data['FitnessGoal'] == goal)['FitnessScore'].max()
        for goal in goals
    }
    max_scores = experiment_data['FitnessGoal'].map(per_goal_max_scores)
    experiment_data['FitnessScore'] /= max_scores

    # Calculate expt vs. ctrl p-values for each of 8 fitness goals
    per_goal_p_value = {}
    for goal in goals:
        data_for_goal = experiment_data.where(
            experiment_data['FitnessGoal'] == goal)
        expt_data_points = data_for_goal.where(
            experiment_data['Kind'] == 'Experiment'
        )['FitnessScore'].dropna()
        ctrl_data_points = data_for_goal.where(
            experiment_data['Kind'] == 'Control'
        )['FitnessScore'].dropna()
        per_goal_p_value[goal] = mannwhitneyu(
            expt_data_points, ctrl_data_points).pvalue

    # Adjust the p values to account for the false positive rate associated
    # with repeated statistical tests.
    corrected_p_values = false_discovery_control(
        list(per_goal_p_value.values()))
    for goal, new_value in zip(per_goal_p_value.keys(), corrected_p_values):
        per_goal_p_value[goal] = new_value

    # Render a chart with one column for each FitnessGoal, comparing experiment
    # to control.
    plt.figure().suptitle(f'Final phenotype fitness per FitnessGoal')
    ax = sns.boxplot(
        data=experiment_data, x='FitnessGoal', y='FitnessScore',
        hue='Kind', width=0.3, showfliers=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_ylabel('Fitness score (normalized)')

    # Display p-values above each column, staggered so they don't overlap with
    # one another.
    for index, goal in enumerate(goals):
        p_value = per_goal_p_value[goal]
        x = index - 0.5  # Ideally, this should account for label width.
        y = 1.175 - 0.1 * (index % 2)
        plt.text(x, y, f'(p = {p_value:.2g})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('output/experiments/organism_fitness_all_goals.png')
    plt.close()


def visualize_experiment_vs_control():
    """Generate all visualizations comparing experiment vs. control."""
    best_expt_species_by_goal = {}
    first_gen_best_organism_fitness_by_goal = {}
    all_experiment_data = pd.DataFrame()
    # Pre-process all the experiment data, capturing the best species for each
    # goal and building up a master table of all the experiment data.
    for experiment in experiment_list:
        goal = experiment.fitness_goal.name
        this_experiment_data = experiment.get_results()
        # Pick out the species from the trial with the most fit organisms.
        species_data = this_experiment_data.best_species_per_trial[
            np.argmax([species.best_organism.fitness
                       for species in
                       this_experiment_data.best_species_per_trial])]
        # If we're processing multiple experiments with the same fitness goal,
        # only keep track of the best one.
        prev_best, _, _ = best_expt_species_by_goal.get(
            goal, (None, None, None))
        if (prev_best is None or
            prev_best.best_organism.fitness <
            species_data.best_organism.fitness):
            best_expt_species_by_goal[goal] = (
                species_data, str(experiment.constraints),
                this_experiment_data)
        prev_best, _ = first_gen_best_organism_fitness_by_goal.get(
            goal, (None, None))
        first_gen_best_organism_fitness_by_goal[goal] = (
            this_experiment_data.first_gen_organism_fitness_scores)

        # Go through the organism scores for the first generation to find the
        # ones associated with the best performing species trial.
        all_species_first_gen_scores = (
            this_experiment_data.first_gen_organism_fitness_scores)
        assert len(all_species_first_gen_scores) == NUM_TRIALS
        prev_best = -1
        for first_gen_scores in all_species_first_gen_scores:
            assert first_gen_scores.shape == (
                NUM_TRIALS, NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS)
            this_species_best_fitness = first_gen_scores[:, :, -1].max()
            if prev_best < this_species_best_fitness:
                first_gen_best_organism_fitness_by_goal[goal] = (
                    first_gen_scores.max(axis=1))

        all_experiment_data = pd.concat((all_experiment_data, pd.DataFrame({
            'FitnessGoal': goal,
            'FitnessScore': this_experiment_data.organism_fitness_scores,
            'Configuration': str(experiment.constraints),
            'Kind': 'Experiment',
        })))

    # Do equivalent processing for all the control data.
    best_ctrl_species_by_goal = {}
    for control in control_list:
        species_data = control.load_from_filesystem()
        goal = control.fitness_goal.name
        prev_best, _, _ = best_ctrl_species_by_goal.get(
            goal, (None, None, None))
        if (prev_best is None or
            prev_best.best_organism.fitness <
            species_data.best_organism.fitness):
            best_ctrl_species_by_goal[goal] = (
                species_data, control.config_str, None)
        best_organism_fitness = species_data.all_trial_organism_fitness.max(
            axis=1)
        all_experiment_data = pd.concat((all_experiment_data, pd.DataFrame({
            'FitnessGoal': goal,
            'FitnessScore': best_organism_fitness[:, -1],
            'Configuration': control.config_str,
            'Kind': 'Control',
        })))
    goals = all_experiment_data['FitnessGoal'].unique()

    # Render a summary of the best phenotypes from experiment and control.
    visualize_best_phenotypes(goals, best_expt_species_by_goal)
    plt.savefig(f'output/experiments/best_expt_phenotypes.png')
    plt.close()
    visualize_best_phenotypes(goals, best_ctrl_species_by_goal)
    plt.savefig(f'output/experiments/best_ctrl_phenotypes.png')
    plt.close()

    # Visualize how evolvability improves as the species evolve.
    visualize_evolvability(goals,
                           first_gen_best_organism_fitness_by_goal,
                           best_expt_species_by_goal,
                           best_ctrl_species_by_goal)

    # Visualize best phenotype fitness of experiment vs control for all goals.
    visualize_per_goal_fitness(goals, all_experiment_data)


def visualize_species_range():
    """Render sample phenotypes from 50 unevolved species."""
    filename = 'output/random_initial_population.png'
    if not Path(filename).exists():
        programs = Clade.make_random_species(50)
        videos = render_random_populations(
            np.fromiter(
                (program.serialize() for program in programs),
                dtype=PhenotypeProgramDType))
        # 5x10 view showing all samples.
        fig, axes = render_image_grid(videos[:, 0], rows=5, cols=10)
        # 2x4 view for paper
        # fig, axes = render_image_grid(videos[:, 0], rows=2, cols=4)
        plt.tight_layout()
        fig.savefig(filename)
        plt.close()


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
    # Note: Disabled. Didn't end up using this visualization for the paper, and
    # did not re-generate data for all configuration variants.
    # if all(experiment.has_started() for experiment in experiment_list):
    #     visualize_cross_experiment_comparisons()

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
