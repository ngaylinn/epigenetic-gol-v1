"""Trace a few generations of species evolution.

This script duplicates Clade.evolve_species, but with added instrumentation to
capture detailed information about fitness, selection, and phenotypes for each
generation. This is useful for iterating on the implementation of FitnessGoals
and debugging the evolutionary process.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from evolution import Clade, compute_species_fitness
from experiments import NUM_SPECIES, NUM_TRIALS
import gif_files
from kernel import Cell, FitnessGoal, WORLD_SIZE
from phenotype_program import Constraints, CROSSOVER_RATE

# This script only looks at the first few generations of species.
NUM_SPECIES_GENERATIONS = 5


def main():
    goal = FitnessGoal.RING
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


def glider_fitness(video):
    before = np.pad(
        video[-4 - 1],
        ((1, 0), (1, 0)),
        constant_values=Cell.DEAD
    )[:WORLD_SIZE, :WORLD_SIZE]
    after = video[-1]
    static = np.logical_and(
        video[-1] == video[-2],
        np.logical_and(
            video[-2] == video[-3],
            video[-3] == video[-4]))
    gif_files.display_simulation_data(static * 255)
    repeating = np.count_nonzero(
        np.logical_and(
            np.logical_not(static),
            np.logical_and(
                before == int(Cell.ALIVE),
                after == int(Cell.ALIVE))))
    gif_files.display_simulation_data(
        np.logical_and(
            before == int(Cell.ALIVE),
            after == int(Cell.ALIVE)) * 255)
    not_repeating = np.count_nonzero(
        np.logical_and(
            after == int(Cell.ALIVE),
            np.logical_or(
                np.logical_not(before == int(Cell.ALIVE)),
                static)))

    result = (100 * repeating) / (1 + not_repeating)
    print(f'(100 * {repeating}) / (1 + {not_repeating}) == {result}')


def debug_cycling(cycle_length, video):
    num_iterations = 4
    last_cycle = video[-cycle_length:]
    cycling = np.ones((WORLD_SIZE, WORLD_SIZE))
    not_cycling = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(num_iterations):
        this_cycle = video[100-(1 + i)*cycle_length:100-i*cycle_length]
        repeating = np.ones((WORLD_SIZE, WORLD_SIZE))
        always_on = np.ones((WORLD_SIZE, WORLD_SIZE))
        always_off = np.ones((WORLD_SIZE, WORLD_SIZE))
        for frame in range(cycle_length):
            always_on = np.logical_and(
                always_on, this_cycle[frame] == Cell.ALIVE)
            always_off = np.logical_and(
                always_off, this_cycle[frame] == Cell.DEAD)
            repeating = np.logical_and(
                repeating, this_cycle[frame] == last_cycle[frame])
        cycling = np.logical_and(
            cycling,
            np.logical_and(
                np.logical_not(always_off),
                np.logical_and(
                    np.logical_not(always_on),
                    repeating)))
        not_cycling = np.logical_or(
            not_cycling,
            np.logical_and(
                np.logical_not(always_off),
                np.logical_not(repeating)))
    print(cycling.shape)
    print(f'Cycling: {np.count_nonzero(cycling)}')
    gif_files.display_simulation_data(cycling * 255)
    print(f'Not Cycling: {np.count_nonzero(not_cycling)}')
    gif_files.display_simulation_data(not_cycling * 255)
    # prev_cycle = video[-2 * cycle_length:-cycle_length]
    # cycling = np.count_nonzero(last_cycle == prev_cycle)
    # not_cycling = np.count_nonzero(last_cycle != prev_cycle)
    # print(f'Cycling: {cycling}')
    # print(f'Not cycling: {not_cycling}')
    # print(f'Fitness: {(100 * cycling) / (1 + not_cycling)}')
    # overlay = last_cycle // cycle_length + prev_cycle // cycle_length
    # gif_files.display_simulation_data(overlay)


if __name__ == '__main__':
    # o22 = gif_files.load_simulation_data_from_image('output/trace/gen000/best_for_species22.gif')
    # o29 = gif_files.load_simulation_data_from_image('output/trace/gen000/best_for_species29.gif')
    # glider_fitness(o22)
    # glider_fitness(o29)
    main()
    # debug_cycling(
    #     3,
    #     gif_files.load_simulation_data_from_image('output/trace/gen000/best_organism.gif'))
