#include "environment.h"
#include "phenotype_program.h"
#include "simulator.h"

// TODO: Unused?
#include <vector>

#include "cuda_utils.cuh"
#include "gol_simulation.h"
#include "reproduction.h"
#include "selection.h"

namespace epigenetic_gol_kernel {

// A convenience class for allocating, initializing, and freeing all of
// the GPU-side data objects used by Simulation.
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
        // The Videos take up a ton of space on the GPU, and aren't necessary
        // most of the time, since Videos are typically only recorded on
        // demand. This was added to simplify the ENTROPY FitnessGoal, which
        // requires capturing full simulation Videos.
        DeviceData<Video> videos;

        DeviceAllocations(int num_species, int size)
            : programs(num_species),
              rngs(size),
              curr_gen_genotypes(size),
              next_gen_genotypes(size),
              parent_selections(size),
              mate_selections(size),
              fitness_scores(size),
              videos(size) {
        }
};

Simulator::Simulator(
        unsigned int num_species,
        unsigned int num_trials,
        unsigned int num_organisms)
    : num_species(num_species),
      num_trials(num_trials),
      num_organisms(num_organisms),
      size(num_species * num_trials * num_organisms) {
    d = new DeviceAllocations(num_species, size);
    // Normally the caller would seed the Simulator manually, but convenience
    // and safety, make sure the RNGs always get initialized to SOMETHING.
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

    // After generating next_gen_genotypes from curr_gen_genotypes, swap the
    // two pointers so that the the new data is considered current and the old
    // data may be overwritten to compute the next generation.
    d->curr_gen_genotypes.swap(d->next_gen_genotypes);
}

void Simulator::simulate(const FitnessGoal& goal) {
    // Passing in d->videos is only necessary for the ENTROPY FitnessGoal, but
    // it's simpler just to use it every time.
    simulate_population<false>(
            size, num_species, goal, d->programs,
            d->curr_gen_genotypes, d->fitness_scores, d->videos);
}

Video* Simulator::simulate_and_record(const FitnessGoal& goal) {
    simulate_population<true>(
            size, num_species, goal, d->programs,
            d->curr_gen_genotypes, d->fitness_scores, d->videos);
    return d->videos.copy_to_host();
}

const Fitness* Simulator::evolve(
        const PhenotypeProgram* h_programs,
        const FitnessGoal& goal,
        const int num_generations) {
    populate(h_programs);
    for (int i = 0; i < num_generations - 1; i++) {
        simulate(goal);
        propagate();
    }
    simulate(goal);
    return get_fitness_scores();
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
    Simulator simulator(50, 5, 50);
    FitnessGoal goal = FitnessGoal::STILL_LIFE;
    PhenotypeProgram programs[50];
    for (int i = 0; i < 50; i++) {
        programs[i].draw_ops[0].compose_mode = ComposeMode::OR;
        programs[i].draw_ops[0].global_transforms[0].type =
            TransformMode::TILE;
    }
    simulator.evolve(programs, goal, 200);
    return 0;
}
