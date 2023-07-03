#include "environment.h"
#include "phenotype_program.h"
#include "simulator.h"

#include <vector>

#include "cuda_utils.cuh"
#include "gol_simulation.h"
#include "reproduction.h"
#include "selection.h"

namespace epigenetic_gol_kernel {

class Simulator::DeviceAllocations {
    friend class Simulator;
    private:
        DeviceData<curandState> rngs;
        DeviceData<PhenotypeProgram> programs;
        DeviceData<Genotype> curr_gen_genotypes;
        DeviceData<Genotype> next_gen_genotypes;
        DeviceData<unsigned int> parent_selections;
        DeviceData<unsigned int> mate_selections;
        DeviceData<Fitness> fitness_scores;

        DeviceAllocations(int num_species, int size)
            : programs(num_species),
              rngs(size),
              curr_gen_genotypes(size),
              next_gen_genotypes(size),
              parent_selections(size),
              mate_selections(size),
              fitness_scores(size) {}
};

// TODO: Consider running simulations in parallel with breeding
// PhenotypePrograms in python, if that makes sense.
Simulator::Simulator(
        unsigned int num_species,
        unsigned int num_trials,
        unsigned int num_organisms)
    : num_species(num_species),
      num_trials(num_trials),
      num_organisms(num_organisms),
      size(num_species * num_trials * num_organisms) {
    d = new DeviceAllocations(num_species, size);
    seed(42);
}

Simulator::~Simulator() {
    delete d;
}


// ---------------------------------------------------------------------------
// Methods for managing simulation control flow
// ---------------------------------------------------------------------------

void Simulator::populate(const PhenotypeProgram* h_programs) {
    d->programs.copy_from_host(h_programs);
    randomize_population(size, d->curr_gen_genotypes, d->rngs);
}

void Simulator::propagate() {
    select_from_population(size, num_organisms, d->fitness_scores,
            d->parent_selections, d->mate_selections, d->rngs);

    breed_population(
            size, d->parent_selections, d->mate_selections,
            d->curr_gen_genotypes, d->next_gen_genotypes, d->rngs);

    // After generationg next_gen_genotypes from curr_gen_genotypes, swap the
    // two pointers so that the the new data is "current" and the old data may
    // be overwritten to compute the next generation.
    d->curr_gen_genotypes.swap(d->next_gen_genotypes);
}

void Simulator::simulate(const FitnessGoal& goal) {
    simulate_population<false>(
            size, num_species, goal, d->programs,
            d->curr_gen_genotypes, d->fitness_scores, nullptr);
}

Video* Simulator::simulate_and_record(const FitnessGoal& goal) {
    DeviceData<Video> videos(size);
    simulate_population<true>(
            size, num_species, goal, d->programs,
            d->curr_gen_genotypes, d->fitness_scores, videos);
    return videos.copy_to_host();
}

void Simulator::evolve(
        const PhenotypeProgram* h_programs,
        const FitnessGoal& goal,
        const int num_generations) {
    populate(h_programs);
    for (int i = 0; i < num_generations - 1; i++) {
        simulate(goal);
        propagate();
    }
    simulate(goal);
}


// ---------------------------------------------------------------------------
// Methods to retrieve simulation results computed by simulate()
// ---------------------------------------------------------------------------

const Fitness* Simulator::get_fitness_scores() const {
    return d->fitness_scores.copy_to_host();
}

const Genotype* Simulator::get_genotypes() const {
    return d->curr_gen_genotypes.copy_to_host();
}


// ---------------------------------------------------------------------------
// Methods to manage RNG state
// ---------------------------------------------------------------------------

void Simulator::seed(const unsigned int seed_value) {
    seed_rngs(d->rngs, size, seed_value);
}

} // namespace epigenetic_gol_kernel


// ---------------------------------------------------------------------------
// A pure C++ demo for use with NVidia's profiling and debug tools.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    using namespace epigenetic_gol_kernel;
    Simulator simulator(32, 5, 32);
    FitnessGoal goal = FitnessGoal::STILL_LIFE;
    PhenotypeProgram programs[32];
    for (int i = 0; i < 32; i++) {
        // TODO: Test this works as intended.
        programs[i].draw_ops[0].compose_mode = ComposeMode::OR;
        programs[i].draw_ops[0].global_transforms[0].type =
            TransformType::TILE;
    }
    simulator.populate(programs);
    for (int i = 0; i < 199; i++) {
        simulator.simulate(goal);
        simulator.propagate();
        // Include this GPU->host data transfer that the real project requires.
        simulator.get_fitness_scores();
    }
    simulator.simulate(goal);
    return 0;
}
