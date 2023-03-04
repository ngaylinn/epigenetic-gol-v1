#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <vector>

#include <curand_kernel.h>

#include "environment.h"
#include "genotype.cuh"
#include "interpreter.cuh"

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
        Interpreter** interpreters;
        curandState* rngs;
        Genotype* genotypes;
        unsigned int* parent_selections;
        unsigned int* mate_selections;
        Fitness* fitness_scores;
        Video* videos;

    public:
        const unsigned int num_species;
        const unsigned int num_trials;
        const unsigned int num_organisms;

        // The total number of individuals being simulated, which is
        // num_species * num_trials * num_organisms.
        const unsigned int size;

        Simulator(
            unsigned int num_species, unsigned int num_trials,
            unsigned int num_organisms);
        ~Simulator();

        // -------------------------------------------------------------------
        // Methods for managing simulation control flow
        // -------------------------------------------------------------------

        // Generate a randomized population.
        void populate(Interpreter* h_interpreters = nullptr);
        // Generate a new population from the previous generation.
        void propagate();
        // Note, simulate always updates fitness_scores but for performance
        // reasons it will only record videos if requested.
        void simulate(FitnessGoal goal, bool record=false);

        // -------------------------------------------------------------------
        // Methods to retrieve simulation results computed by simulate()
        // -------------------------------------------------------------------

        const Fitness* get_fitness_scores() const;
        const Video* get_videos() const;
        const Genotype* get_genotypes() const;

        // -------------------------------------------------------------------
        // Methods to manage RNG state
        // -------------------------------------------------------------------

        const std::vector<unsigned char> get_state() const;
        void restore_state(std::vector<unsigned char> data);
        void reset_state();
};

// An alternate implementation of Simulator for testing with methods for data
// injection.
class TestSimulator : public Simulator {
    public:
        using Simulator::Simulator;

        // Alternative to populate() that forces all simulations to use the
        // given phenotype.
        void simulate_phenotype(
                const Frame* h_phenotype, FitnessGoal goal, bool record=false);
};

} // namespace epigenetic_gol_kernel

#endif
