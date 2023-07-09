import functools
import pathlib
import pickle
import random

import numpy as np
import time
import tqdm

from kernel import FitnessDType, FitnessGoal, Simulator
from phenotype_program import Clade, Constraints

NUM_SPECIES = 50
NUM_TRIALS = 5
NUM_ORGANISMS = 50
POPULATION_SIZE = NUM_SPECIES * NUM_TRIALS * NUM_ORGANISMS

NUM_SPECIES_GENERATIONS = 150
NUM_ORGANISM_GENERATIONS = 150

LIFETIMES_PER_TRIAL = (
    NUM_SPECIES_GENERATIONS * NUM_ORGANISM_GENERATIONS * POPULATION_SIZE)

# Weight for computing species fitness. The first generation of organisms has
# their fitness discounted by 50% while the last generation gets full credit.
# By shaping these values to match the fitness_summary data, we can compute
# fitness for all species simultaneously.
WEIGHTS = np.full(
    (NUM_SPECIES, NUM_ORGANISM_GENERATIONS),
    np.linspace(0.5, 1.0, num=NUM_ORGANISM_GENERATIONS))


def evolve_organisms(simulator, fitness_goal, clade):
    fitness_scores = np.empty(
        (NUM_SPECIES, NUM_TRIALS,
         NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS),
        dtype=np.uint32)
    simulator.populate(clade.serialize())
    for generation in range(NUM_ORGANISM_GENERATIONS):
        simulator.simulate(fitness_goal)
        fitness_scores[:, :, :, generation] = simulator.get_fitness_scores()
        if generation + 1 < NUM_ORGANISM_GENERATIONS:
            simulator.propagate()
    return fitness_scores


def compute_species_fitness(organism_fitness):
    assert organism_fitness.shape == (
        NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS,
        NUM_ORGANISM_GENERATIONS)

    result = (
        # Consider just the median score across trials
        np.median(
            # Consider just the best-performing organism score for each trial
            np.max(organism_fitness, axis=2),
            axis=1,
        # Weight the scores so earlier generations are worth less.
        ) * WEIGHTS
    # Sum up the weighted per-generation median score for each species.
    ).sum(axis=1)
    assert result.shape == (NUM_SPECIES,)
    return result.astype(FitnessDType)


def evolve_species(trial, clade, fitness_goal):
    simulator = Simulator(NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
    simulator.seed(trial)
    species_fitness = np.empty(
        (NUM_SPECIES, NUM_SPECIES_GENERATIONS), dtype=np.uint32)
    progress_bar = tqdm.tqdm(
        total=NUM_SPECIES_GENERATIONS,
        mininterval=1,
        bar_format='Gen {n_fmt} of {total_fmt} | {bar} | {elapsed}')

    for generation in range(NUM_SPECIES_GENERATIONS):
        organism_fitness = evolve_organisms(simulator, fitness_goal, clade)
        assert organism_fitness.shape == (
            NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS)

        genotypes = simulator.get_genotypes()
        assert genotypes.shape == (NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)

        this_gen_fitness = compute_species_fitness(organism_fitness)
        species_fitness[:, generation] = this_gen_fitness

        if generation + 1 < NUM_SPECIES_GENERATIONS:
            clade.propagate(genotypes, this_gen_fitness)
        progress_bar.update()

    species_index = this_gen_fitness.argmax()
    organism_index = organism_fitness[
        species_index, trial, :, -1].argmax()
    fitness = organism_fitness[
        species_index, trial, organism_index]
    genotype = genotypes[
        species_index, trial, organism_index]
    best_organism = OrganismData(fitness, genotype)

    # Record the fitness history for each species, along with sample
    # organisms from the last generation representing the best and median
    # trials.
    best_species_index = species_fitness[:, -1].argmax()
    return SpeciesData(
        species_fitness[best_species_index],
        organism_fitness[best_species_index],
        clade[best_species_index],
        best_organism)


@functools.total_ordering
class OrganismData:
    def __init__(self, organism_fitness, genotype):
        self.fitness_scores = organism_fitness
        assert self.fitness_scores.shape == (NUM_ORGANISM_GENERATIONS,)
        self.genotype = genotype

    def __eq__(self, other):
        if other is None:
            return False
        return self.fitness_scores[-1] == other.fitness_scores[-1]

    def __lt__(self, other):
        if other is None:
            return False
        return self.fitness_scores[-1] < other.fitness_scores[-1]


@functools.total_ordering
class SpeciesData:
    def __init__(self, species_fitness, organism_fitness,
                 phenotype_program, best_organism):
        assert organism_fitness.shape == (
            NUM_TRIALS, NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS)
        assert species_fitness.shape == (NUM_SPECIES_GENERATIONS,)

        self.fitness_scores = species_fitness
        self.phenotype_program = phenotype_program

        self.all_trial_organism_fitness = organism_fitness.max(axis=1)
        assert self.all_trial_organism_fitness.shape == (
            NUM_TRIALS, NUM_ORGANISM_GENERATIONS)

        # The best organism can vary significantly across trials, but not
        # always, and the focus of this experiment is on the species. For that
        # reason, only capture one best organism per species.
        self.best_organism = best_organism

    def __eq__(self, other):
        if other is None:
            return False
        return self.fitness_scores[-1] == other.fitness_scores[-1]

    def __lt__(self, other):
        if other is None:
            return False
        return self.fitness_scores[-1] < other.fitness_scores[-1]


class ExperimentData:
    def __init__(self, experiment_path):
        self.results_path = experiment_path.joinpath('results.pickle')
        if self.load_from_disk():
            return
        self.best_species_per_trial = []
        self.all_trial_species_fitness = np.zeros(
            (NUM_TRIALS, NUM_SPECIES_GENERATIONS), dtype=FitnessDType)

    def log_trial(self, trial, species_data):
        print('Log trial')
        self.best_species_per_trial.append(species_data)
        self.best_species_per_trial.sort()
        self.all_trial_species_fitness[trial] = (
            species_data.fitness_scores)
        self.save_to_disk()

    def load_from_disk(self):
        try:
            with open(self.results_path, 'rb') as file:
                data = pickle.load(file)
            (self.best_species_per_trial,
             self.all_trial_species_fitness) = data
            return True
        except Exception:
            return False

    def save_to_disk(self):
        print('Saving results to disk')
        data = (self.best_species_per_trial,
                self.all_trial_species_fitness)
        with open(self.results_path, 'wb') as file:
            pickle.dump(data, file)
        return True


@functools.total_ordering
class Experiment:
    def __init__(self, path, name=None, fitness_goal=None,
                 constraints=None):
        self.path = path
        self.state_path = path.joinpath('state.pickle')
        if self.load_from_disk():
            return
        self.name = name
        self.fitness_goal = fitness_goal
        self.constraints = constraints
        self.trial = -1
        self.elapsed = 0
        self.lifetimes = 0
        self.experiment_data = None

    def has_started(self):
        return self.trial > -1

    def has_finished(self):
        return self.trial + 1 >= NUM_TRIALS

    def get_data(self):
        return ExperimentData(self.path)

    def run_trial(self):
        self.trial += 1
        # Make sure we get the same pseudorandom numbers if we re-run the same
        # trial, but different numbers for different trials.
        random.seed(self.trial)
        start = time.perf_counter()
        clade = Clade(NUM_SPECIES, self.constraints)
        # Load partial results from disk into memory, if they are available.
        experiment_data = self.get_data()
        experiment_data.log_trial(
            self.trial, evolve_species(self.trial, clade, self.fitness_goal))
        # Only hold experiment results in memory for the duration of the trial.
        # This prevents the program's memory footprint from growing as it
        # processes long batch runs.
        del experiment_data
        self.elapsed += time.perf_counter() - start
        self.lifetimes += LIFETIMES_PER_TRIAL
        self.save_to_disk()

    @property
    def klps(self):
        return self.lifetimes / self.elapsed / 1000

    @property
    def average_trial_duration(self):
        if self.trial == -1:
            return None
        return self.elapsed / (self.trial + 1)

    @property
    def remaining_trials(self):
        return NUM_TRIALS - self.trial - 1

    def save_to_disk(self):
        data = (self.name,
                self.fitness_goal,
                self.constraints,
                self.trial,
                self.elapsed,
                self.lifetimes)
        with open(self.state_path, 'wb') as file:
            pickle.dump(data, file)
        return True

    def load_from_disk(self):
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


def build_experiment_list():
    goals = [
        # FitnessGoal.EXPLODE,
        # FitnessGoal.GLIDERS,
        # FitnessGoal.LEFT_TO_RIGHT,
        FitnessGoal.STILL_LIFE,
        # FitnessGoal.SYMMETRY,
        # FitnessGoal.THREE_CYCLE,
        # FitnessGoal.TWO_CYCLE,
    ]
    bias_options = [
        # False,
        True,
    ]
    composition_options = [
        False,
        # True,
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
                    constraint_str = (
                        f'B{allow_bias:b}'
                        f'C{allow_composition:b}'
                        f'S{allow_stamp_transforms:b}')
                    path = pathlib.Path(
                        f'output/experiments/{goal.name}/{constraint_str}')
                    path.mkdir(parents=True, exist_ok=True)
                    name = f'{goal.name}_{constraint_str}'
                    experiment_list.append(
                        Experiment(path, name, goal, constraints))
    return experiment_list


experiment_list = build_experiment_list()
