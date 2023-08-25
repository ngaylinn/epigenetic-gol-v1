import os

import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm

from experiments import (
    NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS,
    NUM_SPECIES_GENERATIONS, NUM_ORGANISM_GENERATIONS)
import gif_files
from kernel import Cell, FitnessGoal, Simulator, WORLD_SIZE
from phenotype_program import Clade, Constraints, CROSSOVER_RATE


def evolve_organisms(simulator, fitness_goal, clade):
    fitness_scores = np.empty(
        (NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS, NUM_ORGANISM_GENERATIONS),
        dtype=np.uint32)
    simulator.populate(clade.serialize())
    videos = None
    for generation in range(NUM_ORGANISM_GENERATIONS):
        if generation + 1 < NUM_ORGANISM_GENERATIONS:
            simulator.simulate(fitness_goal)
            simulator.propagate()
        else:
            videos = simulator.simulate_and_record(fitness_goal)
        fitness_scores[:, :, :, generation] = simulator.get_fitness_scores()
    return fitness_scores, videos


def main():
    random.seed(42)
    goal = FitnessGoal.GLIDERS
    constraints = Constraints(
        allow_bias=True,
        allow_stamp_transforms=True,
        allow_composition=False)
    clade = Clade(NUM_SPECIES, constraints)
    simulator = Simulator(NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
    selections = None
    progress_bar = tqdm.tqdm(
        total=NUM_SPECIES_GENERATIONS,
        mininterval=1,
        bar_format='Gen {n_fmt} of {total_fmt} | {bar} | {elapsed}')
    for generation in range(NUM_SPECIES_GENERATIONS):
        path = f'output/trace/gen{generation:03d}'
        os.makedirs(path, exist_ok=True)

        # Record parents for this generation (not available on the first)
        if selections is not None:
            np.savetxt(f'{path}/parent_selections.csv', selections, fmt='%d')

        # Save a summary of all of the Clade's PhenotypePrograms
        with open(f'{path}/phenotype_programs.txt', mode='w') as file:
            for species_index, species in enumerate(clade):
                file.write(f'{species_index}: {species}\n')

        # Evolve some organisms
        organism_fitness, videos = evolve_organisms(simulator, goal, clade)
        genotypes = simulator.get_genotypes()
        species_fitness = experiments.compute_species_fitness(organism_fitness)

        # Breed the next generation and record selections.
        if generation + 1 < NUM_SPECIES_GENERATIONS:
            parents, mates = clade.propagate(genotypes, species_fitness)
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
        best_video = None
        for species_index, species in enumerate(clade):
            # Find the video for the best organism of the median trial
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
            video = videos[species_index][trial_index][organism_index]
            gif_files.save_simulation_data_as_image(
                video, f'{path}/best_for_species{species_index:02d}.gif')
            if fitness > best_fitness:
                best_fitness = fitness
                best_video = video

            axis = fig.add_subplot(5, 10, species_index + 1)
            axis.set_title(species_index)
            axis.grid(False)
            axis.spines[:].set_visible(True)
            axis.set_xlabel(
                f'{species_fitness[species_index]}, {fitness}')
            axis.tick_params(bottom=False, left=False,
                             labelbottom=False, labelleft=False)
            gif_files.add_simulation_data_to_figure(video[0], fig, axis)

            if num_children[species_index] > 0:
                plt.setp(axis.spines.values(), color='#ff0000')
                plt.setp(axis.spines.values(),
                         linewidth=num_children[species_index])
        plt.tight_layout()
        fig.savefig(f'{path}/summary.png')
        plt.clf()

        gif_files.save_simulation_data_as_image(
            best_video, f'{path}/best_organism.gif')

        progress_bar.update()
    progress_bar.close()


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


def debug_cycling(num_frames, video):
    prev_a, prev_b, last_a, last_b = video[-4:]
    a_same = last_a == prev_a
    b_same = last_b == prev_b
    static = last_a == last_b
    alive = last_b == int(Cell.ALIVE)
    cycling = np.count_nonzero(
        np.logical_and(
            a_same, np.logical_and(
                b_same, np.logical_not(static))))
    not_cycling = np.count_nonzero(
        np.logical_and(alive, static))
    print(f'Live cells: {np.count_nonzero(alive)}')
    print(f'Static cells: {np.count_nonzero(static)}')
    print(f'Cycling: {cycling}')
    print(f'Not cycling: {not_cycling}')
    print(f'Fitness: {(100 * cycling) / (1 + not_cycling)}')

    last_cycle = video[-num_frames:]
    prev_cycle = video[-2 * num_frames:-num_frames]
    cycling = np.count_nonzero(last_cycle == prev_cycle)
    not_cycling = np.count_nonzero(last_cycle != prev_cycle)
    print(f'Cycling: {cycling}')
    print(f'Not cycling: {not_cycling}')
    print(f'Fitness: {(100 * cycling) / (1 + not_cycling)}')
    overlay = last_cycle // num_frames + prev_cycle // num_frames
    gif_files.display_simulation_data(overlay)


if __name__ == '__main__':
    # o22 = gif_files.load_simulation_data_from_image('output/trace/gen000/best_for_species22.gif')
    # o29 = gif_files.load_simulation_data_from_image('output/trace/gen000/best_for_species29.gif')
    # glider_fitness(o22)
    # glider_fitness(o29)
    main()
    # debug_cycling(
    #     2, gif_files.load_simulation_data_from_image('output/trace/gen000/best_for_species40.gif'))
