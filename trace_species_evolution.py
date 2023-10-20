"""Trace a few generations of species evolution.

This script duplicates Clade.evolve_species, but with added instrumentation to
capture detailed information about fitness, selection, and phenotypes for each
generation. This is useful for iterating on the implementation of FitnessGoals
and debugging the evolutionary process.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from evolution import NUM_SPECIES, NUM_TRIALS, Clade, compute_species_fitness
import gif_files
from kernel import Cell, FitnessGoal, WORLD_SIZE
from phenotype_program import Constraints, CROSSOVER_RATE

# This script only looks at the first few generations of species.
NUM_SPECIES_GENERATIONS = 5


def main():
    goal = FitnessGoal.ENTROPY
    constraints = Constraints(
        allow_bias=True,
        allow_stamp_transforms=True,
        allow_composition=False)
    clade = Clade(constraints, seed=42)
    clade.randomize_species()
    selections = None
    for generation in range(NUM_SPECIES_GENERATIONS):
        print(f'Running generation {generation}...')
        path = f'output/trace/gen{generation:03d}'
        os.makedirs(path, exist_ok=True)

        # Record parents for this generation (not available on the first)
        if selections is not None:
            np.savetxt(f'{path}/parent_selections.csv', selections, fmt='%d')

        # Save a summary of all of the Clade's PhenotypePrograms
        with open(f'{path}/phenotype_programs.txt', mode='w') as file:
            for species_index, program in enumerate(clade.programs):
                file.write(f'{species_index}: {program}\n')

        # Evolve some organisms
        simulations = clade.evolve_organisms(goal, record=True)
        organism_fitness = clade.organism_fitness_history
        species_fitness = compute_species_fitness(organism_fitness)

        # Breed the next generation and record selections. This happens even
        # for the last generation, even though the children go unusued, just to
        # see which species would have reproduced more.
        parents, mates = clade.propagate_species(species_fitness)
        parent_counts = dict(zip(*np.unique(parents, return_counts=True)))
        mate_counts = dict(zip(*np.unique(mates, return_counts=True)))
        num_children = [
            parent_counts.get(index, 0) +
            CROSSOVER_RATE * mate_counts.get(index, 0)
            for index in range(NUM_SPECIES)
        ]
        selections = np.array(list(zip(parents, mates)))

        # Generate a population chart
        fig = plt.figure(
            f'Generation {generation} Species Summary',
            figsize=(20, 10))
        best_fitness = -1
        best_simulation = None
        for species_index, program in enumerate(clade.programs):
            # Find the simulation for the best organism of the median trial
            # Sort trial indices...
            trial_index = np.argsort(
                # By the total fitness of all organisms in the last generation
                np.sum(organism_fitness[species_index, :, :, -1], axis=1))[
                    # Then take the median trial index.
                    NUM_TRIALS // 2]
            organism_index = np.argmax(
                organism_fitness[species_index, trial_index, :, -1])
            fitness = organism_fitness[
                species_index, trial_index, organism_index, -1]
            simulation = (
                simulations[species_index][trial_index][organism_index])
            gif_files.save_simulation_data_as_image(
                simulation, f'{path}/best_for_species{species_index:02d}.gif')
            if fitness > best_fitness:
                best_fitness = fitness
                best_simulation = simulation

            axis = fig.add_subplot(5, 10, species_index + 1)
            axis.set_title(species_index)
            axis.grid(False)
            axis.spines[:].set_visible(True)
            axis.set_xlabel(
                f'S:{species_fitness[species_index]:,} | O:{fitness:,}')
            axis.tick_params(bottom=False, left=False,
                             labelbottom=False, labelleft=False)
            gif_files.add_simulation_data_to_figure(simulation[0], fig, axis)

            if num_children[species_index] > 0:
                plt.setp(axis.spines.values(), color='#ff0000')
                plt.setp(axis.spines.values(),
                         linewidth=num_children[species_index])
        plt.tight_layout()
        fig.savefig(f'{path}/summary.png')
        plt.close()

        gif_files.save_simulation_data_as_image(
            best_simulation, f'{path}/best_organism.gif')


if __name__ == '__main__':
    main()
