#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <vector>

#include "environment.h"
#include "phenotype_program.h"

namespace epigenetic_gol_kernel {

/*
 * A class for running Game of Life simulations.
 *
 * This class manages all the memory allocated on the GPU and coordinating CUDA
 * kernel launches that operate on those data. It is used to simulate many
 * different "species" which each have their own way of interpreting genotype
 * data. For each species, this class will simulate and evolve num_organisms
 * individuals num_trials times. This means all operations on the simulator
 * simultaneously target num_species * num_trials * num_organisms individual
 * Game of Life simulations. This broad parallelism is essential for getting
 * good performance.
 *
 * This class is meant to be constructed once per project. Assuming population
 * sizes stay the same, the same Simulator can be reused over and over.
 */
class Simulator {
    protected:
        // The device-side memory allocations are stored in a sub-class in
        // order to prevent CUDA-specific details leaking into this .h file,
        // which needs to be standard C++ for pybind11 to use it.
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

        // Generate a new population from the previous generation (use the
        // get_genotypes method to access the new population).
        void propagate();

        // Simulate a population for one lifetime (use the get_fitness_scores
        // method to see how the population did).
        void simulate(const FitnessGoal& goal);

        // Same as simulate, but records a video of the full lifetime of the
        // full population.
        Video* simulate_and_record(const FitnessGoal& goal);

        // Evolve a population of organisms. This is equivalent to calling
        // populate then propagate and simulate repeatedly for num_generations.
        // This method is preferred unless you need observe intermediate states
        // while the population evolves.
        void evolve(
                const PhenotypeProgram* h_programs,
                const FitnessGoal& goal,
                const int num_generations);

        // -------------------------------------------------------------------
        // Methods to retrieve simulation results computed by simulate()
        // -------------------------------------------------------------------

        const Fitness* get_fitness_scores() const;
        const Genotype* get_genotypes() const;

        // -------------------------------------------------------------------
        // Methods to manage RNG state
        // -------------------------------------------------------------------

        void seed(const unsigned int seed_value);
};


// TODO: Find a better home for this function.
const Frame* preview_phenotype(const PhenotypeProgram& h_program);


} // namespace epigenetic_gol_kernel

#endif
