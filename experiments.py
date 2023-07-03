import functools
import pathlib
import pickle
import random

import numpy as np
import time
import tqdm

import kernel
import phenotype_program

# TODO: Merge SPECIES and ORGANISMS constants now that we know they should have
# the same values.
NUM_SPECIES = 50
NUM_TRIALS = 5
NUM_ORGANISMS = 50
POPULATION_SIZE = NUM_SPECIES * NUM_TRIALS * NUM_ORGANISMS

NUM_SPECIES_GENERATIONS = 150
NUM_ORGANISM_GENERATIONS = 150

# Capture the best organisms evolved for a species will every N generations
SAMPLE_FREQUENCY = 10

LIFETIMES_PER_TRIAL = (
    NUM_SPECIES_GENERATIONS * NUM_ORGANISM_GENERATIONS * POPULATION_SIZE)

# Weight for computing species fitness. The first generation of organisms has
# their fitness discounted by 50% while the last generation gets full credit.
# By shaping these values to match the fitness_summary data, we can compute
# fitness for all species simultaneously.
WEIGHTS = np.full(
    (NUM_SPECIES, NUM_ORGANISM_GENERATIONS),
    np.linspace(0.5, 1.0, num=NUM_ORGANISM_GENERATIONS))


# TODO: Eliminate this function by having simulate return all-generation
# fitness.
def evolve_organisms(simulator, fitness_goal, clade):
    fitness_scores = np.empty(
        (NUM_SPECIES, NUM_TRIALS,
         NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS),
        dtype=np.uint32)
    simulator.populate(clade.serialize())
    for generation in range(NUM_ORGANISM_GENERATIONS):
        simulator.simulate(fitness_goal)
        fitness_scores[
            :, :, :, generation] = simulator.get_fitness_scores()
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
            # Consider just the max organism score for each trial
            np.max(organism_fitness, axis=2),
            axis=1,
        # Weight the scores so earlier generations are worth less.
        ) * WEIGHTS
    # Sum up the weighted per-generation median score for each species.
    ).sum(axis=1)
    assert result.shape == (NUM_SPECIES,)
    # TODO: Replace all instances of uint32 with kernel.Fitness?
    return result.astype(np.uint32)


def evolve_species(trial, simulator, fitness_goal, constraints):
    clade = phenotype_program.Clade(NUM_SPECIES, constraints)
    species_fitness = np.empty(
        (NUM_SPECIES, NUM_SPECIES_GENERATIONS), dtype=np.uint32)
    progress_bar = tqdm.tqdm(
        total=NUM_SPECIES_GENERATIONS,
        mininterval=1,
        bar_format='Gen {n_fmt} of {total_fmt} | {bar} | {elapsed}')

    sample_organisms_by_generation = {}
    for generation in range(NUM_SPECIES_GENERATIONS):
        organism_fitness = evolve_organisms(simulator, fitness_goal, clade)
        assert organism_fitness.shape == (
            NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS)

        genotypes = simulator.get_genotypes()
        assert genotypes.shape == (NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)

        this_gen_fitness = compute_species_fitness(organism_fitness)
        species_fitness[:, generation] = this_gen_fitness
        species_index = this_gen_fitness.argmax()

        if (generation + 1) % SAMPLE_FREQUENCY == 0:
            samples = []
            for trial in range(NUM_TRIALS):
                organism_index = organism_fitness[
                    species_index, trial, :, -1].argmax()
                fitness = organism_fitness[
                    species_index, trial, organism_index]
                genotype = genotypes[
                    species_index, trial, organism_index]
                samples.append(OrganismData(fitness, genotype))
            sample_organisms_by_generation[generation] = samples

        if generation + 1 < NUM_SPECIES_GENERATIONS:
            clade.propagate(genotypes, this_gen_fitness)
        progress_bar.update()

    # Record the fitness history for each species, along with sample
    # organisms from the last generation representing the best and median
    # trials.
    best_species_index = species_fitness[:, -1].argmax()
    return SpeciesData(
        species_fitness[best_species_index],
        organism_fitness[best_species_index],
        clade[best_species_index],
        sample_organisms_by_generation)


@functools.total_ordering
class OrganismData:
    def __init__(self, organism_fitness, genotype):
        self.fitness_scores = organism_fitness
        assert self.fitness_scores.shape == (NUM_ORGANISM_GENERATIONS,)
        self.genotype = genotype

    def __eq__(self, other):
        return self.fitness_scores[-1] == other.fitness_scores[-1]

    def __lt__(self, other):
        return self.fitness_scores[-1] < other.fitness_scores[-1]


@functools.total_ordering
class SpeciesData:
    def __init__(self, species_fitness, organism_fitness,
                 phenotype_program, samples):
        assert organism_fitness.shape == (
            NUM_TRIALS, NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS)
        assert species_fitness.shape == (NUM_SPECIES_GENERATIONS,)

        self.fitness_scores = species_fitness
        self.phenotype_program = phenotype_program

        self.all_trial_organism_fitness = organism_fitness.max(axis=1)
        assert self.all_trial_organism_fitness.shape == (
            NUM_TRIALS, NUM_ORGANISM_GENERATIONS)

        # This is a dict where the keys are the generation at which organisms
        # were sampled and the value is a list of length NUM_TRIALS containing
        # the best organism of that generation for each trial.
        self.sample_organisms = samples

    def __eq__(self, other):
        return self.fitness_scores[-1] == other.fitness_scores[-1]

    def __lt__(self, other):
        return self.fitness_scores[-1] < other.fitness_scores[-1]


class ExperimentData:
    def __init__(self, state_path, name=None, fitness_goal=None,
                 constraints=None):
        self.state_path = state_path
        if self.load_from_disk():
            return
        self.name = name
        self.fitness_goal = fitness_goal
        self.constraints = constraints
        self.trial = -1
        self.elapsed = 0
        self.lifetimes = 0
        self.best_species_per_trial = []
        # TODO: replace uint32 with Fitness?
        self.all_trial_species_fitness = np.zeros(
            (NUM_TRIALS, NUM_SPECIES_GENERATIONS), dtype=np.uint32)

    def has_started(self):
        return self.trial > -1

    def has_finished(self):
        return self.trial + 1 >= NUM_TRIALS

    def run_trial(self, simulator):
        self.trial += 1
        # Make sure we get the same pseudorandom numbers if we re-run the same
        # trial, but different numbers for different trials.
        random.seed(self.trial)
        simulator.seed(self.trial)
        start = time.perf_counter()
        species_data = evolve_species(
            self.trial, simulator, self.fitness_goal, self.constraints)
        self.best_species_per_trial.append(species_data)
        self.best_species_per_trial.sort()
        self.all_trial_species_fitness[self.trial] = (
            species_data.fitness_scores)
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
        try:
            data = (
                self.name,
                self.fitness_goal,
                self.constraints,
                self.trial,
                self.elapsed,
                self.lifetimes,
                self.best_species_per_trial,
                self.all_trial_species_fitness)
            with open(self.state_path, 'wb') as file:
                pickle.dump(data, file)
            return True
        except Exception:
            raise
            return False

    def load_from_disk(self):
        try:
            with open(self.state_path, 'rb') as file:
                data = pickle.load(file)
            (self.name,
             self.fitness_goal,
             self.constraints,
             self.trial,
             self.elapsed,
             self.lifetimes,
             self.best_species_per_trial,
             self.all_trial_species_fitness) = data
            return True
        except Exception:
            return False


def build_experiment_list():
    goals = [kernel.FitnessGoal.TWO_CYCLE]
    experiment_list = []
    for goal in goals:
        for allow_bias in (False, True):
            for allow_composition in (False, True):
                for allow_stamp_transforms in (False, True):
                    constraints = phenotype_program.Constraints(
                        allow_bias, allow_composition, allow_stamp_transforms)
                    name = (
                        f'{goal.name}_'
                        f'B{allow_bias:b}'
                        f'C{allow_composition:b}'
                        f'S{allow_stamp_transforms:b}')
                    path = pathlib.Path(f'output/experiments/{name}/')
                    path.mkdir(exist_ok=True)
                    experiment_list.append(
                        ExperimentData(
                            path.joinpath('state.pickle'),
                            name, goal, constraints))
    return experiment_list


experiment_list = build_experiment_list()
