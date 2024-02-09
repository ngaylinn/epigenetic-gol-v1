"""Define all experiment data collected by this project.

This project is an experimental evolutionary algorithm. To better understand
how it works and how various design decisions impact performance, the project
tests various algorithm variants on a suite of different FitnessGoals. Because
luck is a major factor in any evolutionary algorithm, it repeats each
experiment NUM_TRIALS times to estimate average performance.

The run_experiments script uses this module to define the set of experiments to
run, generate result data, and manage that data on a filesystem. This module
uses the evolution module to actually run the experiment and gather data, then
it post-processes that data to record just the information needed by the
visualize_results script.
"""

import functools
import pathlib
import pickle
import time

import numpy as np

from evolution import (
    NUM_TRIALS, NUM_ORGANISMS,
    NUM_SPECIES_GENERATIONS, NUM_ORGANISM_GENERATIONS,
    compute_species_fitness, Clade, ControlClade)
from kernel import FitnessDType, FitnessGoal
from phenotype_program import Constraints

NUM_SPECIES = 50

POPULATION_SIZE = NUM_SPECIES * NUM_TRIALS * NUM_ORGANISMS

LIFETIMES_PER_TRIAL = (
    NUM_SPECIES_GENERATIONS * NUM_ORGANISM_GENERATIONS * POPULATION_SIZE)


class OrganismData:
    """A summary of a single evolved organism.

    Attributes
    ----------
    fitness: FitnessDType
        The fitness score for this organism
    genotype: numpy array of GenotypeDType
        The Genotype of this organism, useful for reproducing the phenotype on
        demand.
    """
    def __init__(self, organism_fitness, genotype):
        self.fitness = organism_fitness
        self.genotype = genotype


# Sortable by fitness
@functools.total_ordering
class SpeciesData:
    """A summary of an evolved species of organisms.

    Attributes
    ----------
    fitness: FitnessDType
        The fitness score for this species
    phenotype_program: PhenotypeProgram
        The PhenotypeProgram for this species, useful for reproducing the
        phenotype of an organism of this species.
    all_trial_organism_fitness: numpy array of FitnessDType
        A record of the best fitness score from the full population of
        organisms across all trials and generations, useful for visualizing
        the "evolvability" of organisms of each species (which determines
        species fitness, like phenotype does for an organism).
    best_organism: OrganismData
        A summary of the best organism found for this species.
    """
    def __init__(self, species_fitness, organism_fitness,
                 phenotype_program, best_organism):
        self.fitness = species_fitness
        self.phenotype_program = phenotype_program

        # Record the fitness history for the full population of organisms in
        # this species.
        assert organism_fitness.shape == (
            NUM_TRIALS, NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS)
        self.all_trial_organism_fitness = organism_fitness

        # The best organism can vary significantly across trials, but not
        # always, and the focus of this experiment is on the species. For that
        # reason, only capture one best organism per species.
        self.best_organism = best_organism

    def best_organism_fitness_history(self):
        """ Return the fitness history of the best organism in each trial."""
        result = self.all_trial_organism_fitness.max(axis=1)
        assert result.shape == (NUM_TRIALS, NUM_ORGANISM_GENERATIONS)
        return result

    def __eq__(self, other):
        if other is None:
            return False
        return self.fitness == other.fitness

    def __lt__(self, other):
        if other is None:
            return False
        return self.fitness < other.fitness


class ExperimentData:
    """A summary of the species evolved in an experiment with multiple trials.

    Attributes
    ----------
    best_species_per_trial: list of SpeciesData
        A summary of the best species from each trial.
    all_trial_species_fitness: numpy array of FitnessDType
        A record of the best fitness score from the full population of
        species across all trials and generations, useful for visualizing
        the "evolvability" of species in this experiment (and the effectiveness
        of the epigenetic algorithm overall).
    first_gen_organism_fitness_scores: numpy array of FitnessDType
        The full fitness history of all organisms in the first species
        generation (unevolved).
    organism_fitness_scores: numpy array of FitnessDType
        The final fitness scores for all organisms of all species in this
        experiment.
    """
    def __init__(self, results_path):
        # Try to load partial results from the filesystem, or create an empty
        # ExperimentData object.
        self.results_path = results_path
        if self.load_from_filesystem():
            return
        self.best_species_per_trial = []
        self.all_trial_species_fitness = []
        self.first_gen_organism_fitness_scores = []
        self.organism_fitness_scores = []

    def log_trial(self, species_trial, clade):
        """Record the results of one experiment trial to the filesystem.

        This method processes the raw experiment data collected in clade to
        capture just the data needed by the visualize_results script. Once the
        data is processed, it gets written to the filesystem.
        """
        # For reference, document the shape of the arrays manipulated here.
        assert clade.species_fitness_history.shape == (
            NUM_SPECIES, NUM_SPECIES_GENERATIONS)
        assert clade.organism_fitness_history.shape == (
            NUM_SPECIES, NUM_TRIALS,
            NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS)
        assert clade.genotypes.shape == (
            NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)

        # Find the species with the best fitness in the final generation, then
        # find the fittest organism evolved for that species across all
        # organism trials.
        species_index = clade.species_fitness_history[:, -1].argmax()
        organism_fitness_by_trial = clade.organism_fitness_history[
            species_index, :, :, -1]
        assert organism_fitness_by_trial.shape == (NUM_TRIALS, NUM_ORGANISMS)
        organism_trial = organism_fitness_by_trial.max(axis=1).argmax()
        organism_index = organism_fitness_by_trial[organism_trial].argmax()

        # Record basic information about the best organism produced by the best
        # species in this species trial.
        organism_data = OrganismData(
            clade.organism_fitness_history[
                species_index, organism_trial, organism_index, -1],
            clade.genotypes[species_index, organism_trial, organism_index])

        # Record the full fitness history for organisms in the first generation
        # of species (unevolved).
        first_gen_species_index = clade.species_fitness_history[:, 0].argmax()
        self.first_gen_organism_fitness_scores.append(
            clade.first_organism_fitness_history[
                first_gen_species_index, :, :, :])

        # Record the final organism fitness for each trial and generation of
        # species evolved.
        self.organism_fitness_scores.extend(
            clade.organism_fitness_history[:, :, :, -1].flatten())

        # Record basic information about the best species in this species
        # trial, including the best organism of that species.
        species_data = SpeciesData(
            clade.species_fitness_history[species_index, -1],
            clade.organism_fitness_history[species_index, :, :, :],
            clade.programs[species_index],
            organism_data)

        # Record best species data from each species trial on filesystem.
        self.best_species_per_trial.append(species_data)
        self.best_species_per_trial.sort()
        self.all_trial_species_fitness.append(
            clade.species_fitness_history[species_index])
        self.save_to_filesystem()

    def load_from_filesystem(self):
        """Attempt to load ExperimentData from the filesystem."""
        try:
            with open(self.results_path, 'rb') as file:
                data = pickle.load(file)
            (self.best_species_per_trial,
             self.all_trial_species_fitness,
             self.first_gen_organism_fitness_scores,
             self.organism_fitness_scores) = data
            return True
        except Exception:
            return False

    def save_to_filesystem(self):
        """Attempt to save ExperimentData to the filesystem."""
        data = (self.best_species_per_trial,
                self.all_trial_species_fitness,
                self.first_gen_organism_fitness_scores,
                self.organism_fitness_scores)
        with open(self.results_path, 'wb') as file:
            pickle.dump(data, file)


# Sortable by number of completed trials.
@functools.total_ordering
class Experiment:
    """Handle for managing result data objects for an Experiment.

    This is the primary interface between this module and the rest of the
    project. Each experiment is associated with a directory path where
    experiment state and results will be recorded. Each experiment takes a set
    of Constraints that determine how species evolution will work, a
    FitnessGoal that each species will adapt to, and a name string to identify
    the experiment. If experiment state is found in the path directory, then
    these experiment metadata will be restored from that state and the
    arguments will be ignored. Otherwise, a new Experiment is created with
    those metadata to be saved to the filesystem once the first trial is run.
    """
    def __init__(self, path, name=None, fitness_goal=None,
                 constraints=None):
        assert path.is_dir()
        self.path = path
        self.results_path = path.joinpath('results.pickle')
        self.state_path = path.joinpath('state.pickle')
        # Either load metadata and state from the filesystem, or initialize
        # them for a new Experiment.
        if self.load_from_filesystem():
            return
        self.name = name
        self.fitness_goal = fitness_goal
        self.constraints = constraints
        self.trial = -1
        self.elapsed = 0
        self.lifetimes = 0
        self.experiment_data = None

    def has_started(self):
        """Returns True iff at least one trial has been run."""
        return self.trial > -1

    def has_finished(self):
        """Returns True iff all trials have completed."""
        return self.trial + 1 >= NUM_TRIALS

    def get_results(self):
        """Load results from the filesystem."""
        return ExperimentData(self.results_path)

    def run_trial(self):
        """Run one trial, and save results to the filesystem."""
        self.trial += 1

        # Record how long it takes to run this trial in order to estimate how
        # long it will take to run future trials.
        start = time.perf_counter()

        # Actually evolve species and capture data for this trial to the
        # filesystem. Note this either starts a new ExperimentData object or
        # appends to the one already present on the filesystem.
        # Seed the RNG with the trial number, so each trial will have a
        # different but repeatable pseudorandom sequence.
        clade = Clade(
            NUM_SPECIES, constraints=self.constraints, seed=self.trial)
        clade.evolve_species(self.fitness_goal)
        experiment_data = self.get_results()
        experiment_data.log_trial(self.trial, clade)

        # Only hold experiment results in memory for the duration of the trial.
        # This prevents the program's memory footprint from growing as it
        # processes long batch runs.
        del experiment_data
        del clade

        # Update experiment metadata and save it to the filesystem.
        self.elapsed += time.perf_counter() - start
        self.lifetimes += LIFETIMES_PER_TRIAL
        self.save_to_filesystem()

    @property
    def klps(self):
        """Experiment execution speed, in thousands of lifetimes per second."""
        return self.lifetimes / self.elapsed / 1000

    @property
    def average_trial_duration(self):
        """Trial runtime in seconds, averaged across all trials run so far."""
        if self.trial == -1:
            return None
        return self.elapsed / (self.trial + 1)

    @property
    def remaining_trials(self):
        """Number of trials left to run."""
        return NUM_TRIALS - self.trial - 1

    def save_to_filesystem(self):
        """Attempt to save metadata and state to the filesystem."""
        data = (self.name,
                self.fitness_goal,
                self.constraints,
                self.trial,
                self.elapsed,
                self.lifetimes)
        with open(self.state_path, 'wb') as file:
            pickle.dump(data, file)

    def load_from_filesystem(self):
        """Attempt to load metadata and state from the filesystem."""
        try:
            with open(self.state_path, 'rb') as file:
                data = pickle.load(file)
            (self.name,
             self.fitness_goal,
             self.constraints,
             self.trial,
             self.elapsed,
             self.lifetimes) = data
            return True
        except Exception:
            return False

    def __eq__(self, other):
        if other is None:
            return False
        return self.trial == other.trial

    def __lt__(self, other):
        if other is None:
            return False
        return self.trial < other.trial


class Control:
    def __init__(self, path, name=None, fitness_goal=None, use_tiling=False,
                 config_str=None):
        self.path = path
        self.results_path = path.joinpath('results.pickle')
        self.name = name
        self.fitness_goal = fitness_goal
        self.use_tiling = use_tiling
        self.config_str = config_str
        self.organism_fitness_scores = []

    def has_started(self):
        return not self.has_finished()

    def has_finished(self):
        return self.results_path.exists()

    def run(self):
        """Run this control and save the results.

        While experiments must evolve NUM_SPECIES different species NUM_TRIALS
        times, the control condition uses just two hard-coded species. That
        means running controls is much simpler and faster than running
        experiments. There's no need to break down the work into smaller chunks
        and save / restore partial progress like above. Just run the control
        and collect the results in one shot.
        """
        # Evolve some organisms for the fixed control species.
        clade = ControlClade(self.use_tiling)
        clade.evolve_organisms(self.fitness_goal)

        # Find the best organism across all trials and record its metadata.
        organism_fitness_by_trial = clade.organism_fitness_history[0, :, :, -1]
        assert organism_fitness_by_trial.shape == (NUM_TRIALS, NUM_ORGANISMS)
        organism_trial = organism_fitness_by_trial.max(axis=1).argmax()
        organism_index = organism_fitness_by_trial[organism_trial].argmax()
        organism_data = OrganismData(
            clade.organism_fitness_history[
                0, organism_trial, organism_index, -1],
            clade.genotypes[0, organism_trial, organism_index])

        # Record metadata for this one species.
        species_fitness = compute_species_fitness(1, clade.organism_fitness_history)
        species_data = SpeciesData(
            species_fitness, clade.organism_fitness_history[0],
            clade.programs[0], organism_data)
        self.save_to_filesystem(species_data)

    def save_to_filesystem(self, species_data):
        """Record results from run to the filesystem."""
        try:
            # Save that metadata.
            with open(self.results_path, 'wb') as file:
                pickle.dump(species_data, file)
            return True
        except Exception:
            return False

    def load_from_filesystem(self):
        """Record results from run to the filesystem."""
        try:
            # Save that metadata.
            with open(self.results_path, 'rb') as file:
                species_data = pickle.load(file)
            return species_data
        except Exception:
            return None


def build_experiment_list():
    """Construct the list of all experiments to run for this project.

    This project compares how many variations of the core algorithm performs on
    a variety of FitnessGoals. Each experiment can take a long time to run
    (nearly two hours on the original development machine), and the cross
    product of all possible Constraints and FitnessGoals is quite large. For
    that reason, it's helpful to sometimes run just a few of the full set of
    experiments. This is easily done by commenting and uncommenting lines
    below, to change up which subsets of the full list of experiments will be
    considered by the run_experiments and visualize_results scripts. Since all
    result data is saved to the filesystem keyed by experiment name, changing
    the set of experiments to run won't clobber results already computed for
    other experiments.

    There's no need to call this function directly, just use the
    experiment_list variable below.
    """
    goals = [
        FitnessGoal.ENTROPY,
        FitnessGoal.EXPLODE,
        FitnessGoal.LEFT_TO_RIGHT,
        FitnessGoal.RING,
        FitnessGoal.STILL_LIFE,
        FitnessGoal.SYMMETRY,
        FitnessGoal.THREE_CYCLE,
        FitnessGoal.TWO_CYCLE,
    ]
    # goals = FitnessGoal.__members__.values()
    bias_options = [
        # False,
        True,
    ]
    composition_options = [
        # False,
        True,
    ]
    stamp_options = [
        # False,
        True,
    ]
    experiment_list = []
    for goal in goals:
        for allow_bias in bias_options:
            for allow_composition in composition_options:
                for allow_stamp_transforms in stamp_options:
                    constraints = Constraints(
                        allow_bias, allow_composition, allow_stamp_transforms)
                    path = pathlib.Path(
                        f'output/experiments/{goal.name}/{constraints}')
                    path.mkdir(parents=True, exist_ok=True)
                    name = f'{goal.name}_{constraints}'
                    experiment_list.append(
                        Experiment(path, name, goal, constraints))
    return experiment_list


# The master list of all experiments to run for this project.
experiment_list = build_experiment_list()


def build_control_list():
    """Like build_experiment_list, but for the controls."""
    goals = [
        FitnessGoal.ENTROPY,
        FitnessGoal.EXPLODE,
        FitnessGoal.LEFT_TO_RIGHT,
        FitnessGoal.RING,
        FitnessGoal.STILL_LIFE,
        FitnessGoal.SYMMETRY,
        FitnessGoal.THREE_CYCLE,
        FitnessGoal.TWO_CYCLE,
    ]
    # goals = FitnessGoal.__members__.values()
    tiling_options = [
        False,
        True,
    ]
    control_list = []
    for goal in goals:
        for use_tiling in tiling_options:
            config_str = f'CONTROL_{"TILE" if use_tiling else "PLACE"}'
            path = pathlib.Path(
                f'output/experiments/{goal.name}/{config_str}')
            path.mkdir(parents=True, exist_ok=True)
            name = f'{goal.name}_{config_str}'
            control_list.append(
                Control(path, name, goal, use_tiling, config_str))
    return control_list

# The master list of all controls to run for this project.
control_list = build_control_list()
