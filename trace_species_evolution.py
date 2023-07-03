import os

import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm

import experiments
import gif_files
import kernel
import phenotype_program


def evolve_organisms(simulator, fitness_goal, clade):
    fitness_scores = np.empty(
        (experiments.NUM_SPECIES, experiments.NUM_TRIALS,
         experiments.NUM_ORGANISMS, experiments.NUM_ORGANISM_GENERATIONS),
        dtype=np.uint32)
    simulator.populate(clade.serialize())
    videos = None
    for generation in range(experiments.NUM_ORGANISM_GENERATIONS):
        if generation + 1 < experiments.NUM_ORGANISM_GENERATIONS:
            simulator.simulate(fitness_goal)
            simulator.propagate()
        else:
            videos = simulator.simulate_and_record(fitness_goal)
        fitness_scores[
            :, :, :, generation] = simulator.get_fitness_scores()
    return fitness_scores, videos


def main():
    random.seed(42)
    goal = kernel.FitnessGoal.LEFT_TO_RIGHT
    constraints = phenotype_program.Constraints(
        allow_bias=True,
        allow_stamp_transforms=True,
        allow_composition=False)
    clade = phenotype_program.Clade(experiments.NUM_SPECIES, constraints)
    simulator = kernel.Simulator(
        experiments.NUM_SPECIES,
        experiments.NUM_TRIALS,
        experiments.NUM_ORGANISMS)
    selections = None
    progress_bar = tqdm.tqdm(
        total=experiments.NUM_SPECIES_GENERATIONS,
        mininterval=1,
        bar_format=('Evolving species: {bar} '
                    '| Gen {n_fmt} of {total_fmt} '
                    '| {elapsed}'))
    for generation in range(experiments.NUM_SPECIES_GENERATIONS):
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
        if generation + 1 < experiments.NUM_SPECIES_GENERATIONS:
            parents, mates = clade.propagate(genotypes, species_fitness)
            parent_counts = dict(zip(*np.unique(parents, return_counts=True)))
            mate_counts = dict(zip(*np.unique(mates, return_counts=True)))
            num_children = [
                parent_counts.get(index, 0) +
                phenotype_program.CROSSOVER_RATE * mate_counts.get(index, 0)
                for index in range(experiments.NUM_SPECIES)
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
                    experiments.NUM_TRIALS // 2]
            organism_index = np.argmax(
                organism_fitness[species_index, trial_index, :, -1])
            fitness = organism_fitness[
                species_index, trial_index, organism_index, -1]
            video = videos[species_index][trial_index][organism_index]
            gif_files.save_image(
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
            gif_files.add_image_to_figure(video[0], fig, axis)

            if num_children[species_index] > 0:
                plt.setp(axis.spines.values(), color='#ff0000')
                plt.setp(axis.spines.values(),
                         linewidth=num_children[species_index])
        plt.tight_layout()
        fig.savefig(f'{path}/summary.png')
        plt.clf()

        gif_files.save_image(best_video, f'{path}/best_organism.gif')

        progress_bar.update()
    progress_bar.close()

def glider_fitness(video):
    before = np.pad(
        video[-4 - 1],
        ((1, 0), (1, 0)),
        constant_values=kernel.DEAD
    )[:kernel.WORLD_SIZE, :kernel.WORLD_SIZE]
    after = video[-1]
    static = np.logical_and(
        video[-1] == video[-2],
        np.logical_and(
            video[-2] == video[-3],
            video[-3] == video[-4]))
    gif_files.display_image(static * 255)
    repeating = np.count_nonzero(
        np.logical_and(
            np.logical_not(static),
            np.logical_and(
                before == int(kernel.ALIVE),
                after == int(kernel.ALIVE))))
    gif_files.display_image(
        np.logical_and(
            before == int(kernel.ALIVE),
            after == int(kernel.ALIVE)) * 255)
    not_repeating = np.count_nonzero(
        np.logical_and(
            after == int(kernel.ALIVE),
            np.logical_or(
                np.logical_not(before == int(kernel.ALIVE)),
                static)))

    result = (100 * repeating) / (1 + not_repeating)
    print(f'(100 * {repeating}) / (1 + {not_repeating}) == {result}')

if __name__ == '__main__':
    # o22 = gif_files.load_image('output/trace/gen000/best_for_species22.gif')
    # o29 = gif_files.load_image('output/trace/gen000/best_for_species29.gif')
    # glider_fitness(o22)
    # glider_fitness(o29)
    main()
