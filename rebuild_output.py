import time
import random

import gif_files
import kernel

NUM_SPECIES = 32
NUM_TRIALS = 5
NUM_ORGANISMS = 32
NUM_GENERATIONS = 200


def main():
    start = time.perf_counter()
    random.seed(42)
    simulator = kernel.Simulator(NUM_SPECIES, NUM_TRIALS, NUM_ORGANISMS)
    goal = kernel.FitnessGoal.STILL_LIFE
    simulator.populate()
    for _ in range(NUM_GENERATIONS - 1):
        simulator.simulate(goal)
        simulator.propagate()
    simulator.simulate(goal, record=True)
    fitness = simulator.get_fitness_scores()
    elapsed = time.perf_counter() - start
    # Don't count the time to copy the videos over in the benchmark.
    videos = simulator.get_videos()
    lifetimes = NUM_SPECIES * NUM_TRIALS * NUM_ORGANISMS * NUM_GENERATIONS
    # For comparison, the Python based prototype got about 5 klps at best.
    print(f'Finished {lifetimes:,} lifetimes in {elapsed:.2f}s '
          f'({lifetimes / elapsed / 1000:0,.2f} klps)')

    samples = []
    for trial_index, trial_fitness in enumerate(fitness):
        for species_index, species_fitness in enumerate(trial_fitness):
            for organism_index, organism_fitness in enumerate(species_fitness):
                if len(samples) < 5 or organism_fitness > max(samples)[0]:
                    samples.append((organism_fitness, trial_index,
                                    species_index, organism_index))
                    samples.sort()
                    samples = samples[-5:]
    for index, record in enumerate(samples):
        organism_fitness, trial_index, species_index, organism_index = record
        print(f"best{index}: {organism_fitness}")
        video = videos[trial_index][species_index][organism_index]
        gif_files.save_image(video, f'output/best{index}.gif')


if __name__ == '__main__':
    main()
