/*
 * A class to manage GPU operations when evolving populations of organisms.
 *
 * The Simulator class is the main interface that the Python part of this
 * project uses to invoke the C++ code. It is responsible for setting up the
 * GPU to run long batch jobs, with a focus on minimizing memory transfers.
 */

#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include "environment.h"
#include "phenotype_program.h"

namespace epigenetic_gol_kernel {

/*
 * A class for running Game of Life simulations.
 *
 * This class manages all the memory allocated on the GPU and coordinates CUDA
 * kernel launches that operate on those data. It is used to simulate many
 * different "species" which each have their own way of interpreting Genotype
 * data. For each species, this class will simulate and evolve num_organisms
 * individual organisms num_trials times. This means all operations on the
 * simulator operate on num_species * num_trials * num_organisms individual
 * Game of Life simulations simultaneously. This broad parallelism is essential
 * for getting good performance.
 *
 * This class is designed to be reusable. Assuming population sizes stay the
 * same, the only reason to make a new instance is to manually reset the full
 * state of the GPU. Although it's pretty cheep to instantiate a Simulator
 * object, that should still be avoided inside of a loop, for performance
 * reasons.
 */
class Simulator {
    protected:
        // The device-side memory allocations are stored in a sub-class in
        // order to prevent CUDA-specific details leaking into this .h file,
        // which needs to be standard C++ for pybind11 to process it.
        class DeviceAllocations;
        DeviceAllocations* d;

    public:
        const unsigned int num_species;
        const unsigned int num_trials;
        const unsigned int num_organisms;

        // The total number of individuals being simulated, which is
        // num_species * num_trials * num_organisms.
        const unsigned int size;

        Simulator(
            unsigned int num_species,
            unsigned int num_trials,
            unsigned int num_organisms);
        ~Simulator();

        // -------------------------------------------------------------------
        // Methods for managing simulation control flow
        // -------------------------------------------------------------------

        // Generate a randomized population.
        void populate(const PhenotypeProgram* h_programs);

        // Generate a new population from the previous generation.
        void propagate();

        // Simulate a population for one lifetime.
        void simulate(const FitnessGoal& goal);

        // Same as simulate, but records a video of the full lifetime of the
        // full population. Note, this is much slower than simulate.
        Video* simulate_and_record(const FitnessGoal& goal);

        // Evolve a population of organisms. This is equivalent to calling
        // populate then propagate and simulate repeatedly for num_generations.
        // This method is preferred unless you need to observe intermediate
        // states while the population evolves.
        const Fitness* evolve(
                const PhenotypeProgram* h_programs,
                const FitnessGoal& goal,
                const int num_generations);

        // -------------------------------------------------------------------
        // Methods to retrieve simulation results computed by simulate()
        // -------------------------------------------------------------------

        // Lookup Fitness scores for the full population as computed in the
        // last call to simulate or simulate_and_record. If no simulation has
        // been run previously, results of this function call are undefined.
        const Fitness* get_fitness_scores() const;

        // Lookup Genotypes for the full population as computed in the last
        // call to populate or propagate. If no organisms have been generated,
        // results of this function call are undefined.
        const Genotype* get_genotypes() const;

        // -------------------------------------------------------------------
        // Methods to manage RNG state
        // -------------------------------------------------------------------

        // Initialize RNG state. Two simulators initialized with the same seed
        // value and performing the same operations are guaranteed to produce
        // the same pseudrandom results.
        void seed(const unsigned int seed_value);
};


} // namespace epigenetic_gol_kernel

#endif
