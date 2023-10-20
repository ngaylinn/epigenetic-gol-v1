"""Implement this project's nested evolutionary algorithm.

This module is used to evolve species of organisms to solve a FitnessGoal. The
Clade class is the primary interface between this module and the rest of the
project. It represents a population of species and organisms that is initially
random but then evolved to find species of greater fitness. The fitness of each
species is determined by evolving a population of organisms of that species,
which is run as a batch job on a CUDA GPU using the kernel module.
"""

from copy import deepcopy
import itertools
import random

import numpy as np
import tqdm

from phenotype_program import (
    BiasMode, Constraints, PhenotypeProgram, TransformMode)
from kernel import select, FitnessDType, PhenotypeProgramDType, Simulator

# Hyperparameters for the evolutionary process.
NUM_SPECIES = 50
NUM_TRIALS = 5
NUM_ORGANISMS = 50
NUM_SPECIES_GENERATIONS = 150
NUM_ORGANISM_GENERATIONS = 150

# Weight for computing species fitness. The first generation of organisms has
# their fitness discounted by 50% while the last generation gets full credit.
# All the others lie on a smooth gradient in between. By shaping these values
# to match the fitness_summary data, we can compute fitness for all species
# simultaneously.
WEIGHTS = np.full(
    (NUM_SPECIES, NUM_ORGANISM_GENERATIONS),
    np.linspace(0.5, 1.0, num=NUM_ORGANISM_GENERATIONS))


def compute_species_fitness(organism_fitness):
    """Evaluate the fitness of a species from the fitness of its organisms.

    This project evolves species for "evolvability," using a proxy metric
    called the "weighted median integral." Each species gets NUM_TRIALS
    attempts to evolve organisms, and this metric looks at how the best
    organism fitness score from the median trial improves over generations.
    Looking at the median trial eliminates outlier results. The fitness scores
    for each generation are weighted (so early generations matter less than
    later ones) and then summed. This encourages species whose organisms
    achieve high fitness that grows over time.
    """
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


class Clade:
    """A population of species to evolve.

    One Clade is created for each experiment trial, representing a poluation of
    species with a common ancestry. Use the evolve_species method to run the
    evolutionary algorithm and generate results.

    Attributes
    ----------
    programs: list of PhenotypeProgram
        A list of NUM_SPECIES PhenotypePrograms designed by evolution.
    genotypes: numpy array of GenotypeDType
        The genotypes from the last generation of organisms for all species and
        organism trials.
    organism_fitness_history: numpy array of FitnessDType
        The fitness scores from the full population of organisms of the last
        species generation, across all organism trials and generations.
    species_fitness_history: numpy array of FitnessDType
        The fitness scores for the full population of species, across all
        species generations.
    """
    def __init__(self, constraints=Constraints(), seed=None):
        self.simulator = Simulator(NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
        if seed is not None:
            random.seed(seed)
            self.simulator.seed(seed)
        self.constraints = constraints
        # This project uses "innovation numbers" as a technique for aligning
        # PhenotypePrograms for cross breeding. As such, all species in a clade
        # share the same counter to keep track of which mutation occurred when.
        self.innovation_counter = itertools.count()
        self.programs = []
        self.genotypes = None
        self.organism_fitness_history = np.empty(
            (NUM_SPECIES, NUM_TRIALS,
             NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS),
            dtype=np.uint32)
        self.species_fitness_history = np.empty(
            (NUM_SPECIES, NUM_SPECIES_GENERATIONS), dtype=np.uint32)

    def evolve_organisms(self, fitness_goal, record=False):
        """Evolve a population of organisms of all species."""
        # Simulations captured from the last generation (if record is True)
        simulations = None
        # Set up the simulator with the latest generation of evolved species
        # and a randomly generated population of organisms for each.
        self.populate_simulator()
        # Evolve organisms on the GPU, capturing data from each generation.
        for generation in range(NUM_ORGANISM_GENERATIONS):
            last_generation = generation + 1 == NUM_ORGANISM_GENERATIONS
            # Record the last generation, only if requested.
            if record and last_generation:
                simulations = self.simulator.simulate_and_record(fitness_goal)
            else:
                self.simulator.simulate(fitness_goal)
            fitness_scores = self.simulator.get_fitness_scores()
            assert fitness_scores.shape == (
                NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
            self.organism_fitness_history[:, :, :, generation] = fitness_scores
            if last_generation:
                # Keep track of organism genotypes from the last generation.
                self.genotypes = self.simulator.get_genotypes()
                assert self.genotypes.shape == (
                    NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
            else:
                self.simulator.propagate()
        return simulations

    def evolve_species(self, fitness_goal):
        """Evolve a population of species."""
        # This operation can take a long time, so show a progress bar on the
        # command line.
        progress_bar = tqdm.tqdm(
            total=NUM_SPECIES_GENERATIONS,
            mininterval=1,
            bar_format='Gen {n_fmt} of {total_fmt} | {bar} | {elapsed}')

        # Start off with a random population of species, then evolve.
        self.randomize_species()
        for generation in range(NUM_SPECIES_GENERATIONS):
            # Evolve a population of organisms for this species and use the
            # results to determine the fitness for each species.
            self.evolve_organisms(fitness_goal)
            this_gen_fitness = compute_species_fitness(
                self.organism_fitness_history)
            self.species_fitness_history[:, generation] = this_gen_fitness

            # Breed new species, unless this is the last generation.
            if generation + 1 < NUM_SPECIES_GENERATIONS:
                self.propagate_species(this_gen_fitness)
            progress_bar.update()
        progress_bar.close()

    def randomize_species(self):
        """Generate a random population of species to evolve.

        This works by starting with a minimal program (just a Stamp with no
        bias or transforms), then systematically generating a population of
        mutants from that individual to cover a diverse range.
        """
        minimal_program = PhenotypeProgram()
        minimal_program.add_draw(self.innovation_counter)
        self.programs = [minimal_program]
        self.programs.extend(
            [deepcopy(minimal_program) for _ in range(1, NUM_SPECIES)])
        for program in self.programs:
            program.randomize(self.innovation_counter, self.constraints)

    def propagate_species(self, fitness_scores):
        """Make a new generation of species derived from the previous one.

        This randomly selects parents and mates from the species population,
        with probability determined by fitness. Each parent will randomly
        either reproduce asexually, or perform crossover with their mate. The
        genotypes from the last generation of organisms is also considered for
        use as bias (see PhenotypeProgram for details).

        This method returns parent and mate selection, which is used by the
        trace_species_evolution script.
        """
        parent_selections = select(fitness_scores, random.getrandbits(32))
        mate_selections = select(fitness_scores, random.getrandbits(32))
        self.programs = [
            self.programs[parent_index].make_offspring(
                self.programs[mate_index],
                self.innovation_counter,
                # Combine genotypes from all trials and organisms.
                this_species_genotypes.flatten(),
                self.constraints)
            for this_species_genotypes, parent_index, mate_index in
            zip(self.genotypes, parent_selections, mate_selections)
        ]
        return parent_selections, mate_selections

    def populate_simulator(self):
        """Send PhenotypeProgram data to the GPU for simulation."""
        programs = np.zeros(NUM_SPECIES, dtype=PhenotypeProgramDType)
        for index, program in enumerate(self.programs):
            program.serialize(programs[index])
        self.simulator.populate(programs)


class TestClade(Clade):
    """A clade with default PhenotypePrograms, for testing."""
    def populate(self):
        test_program = PhenotypeProgram()
        draw_op = test_program.add_draw(self.innovation_counter)
        transform = draw_op.add_global_transform(self.innovation_counter)
        transform.type = TransformMode.TRANSLATE
        transform.args[0].bias_mode = BiasMode.FIXED_VALUE
        transform.args[0].bias = 28
        transform.args[1].bias_mode = BiasMode.FIXED_VALUE
        transform.args[1].bias = 28
        self.programs = [deepcopy(test_program) for _ in range(NUM_SPECIES)]

