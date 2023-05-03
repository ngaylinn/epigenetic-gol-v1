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
        CurandStates rngs;
        DeviceData<PhenotypeProgram> programs;
        DeviceData<Genotype> curr_gen_genotypes;
        DeviceData<Genotype> next_gen_genotypes;
        DeviceData<unsigned int> parent_selections;
        DeviceData<unsigned int> mate_selections;
        DeviceData<Fitness> fitness_scores;
        DeviceData<Video> videos;

        DeviceAllocations(int num_species, int size)
            : rngs(size),
              programs(num_species),
              curr_gen_genotypes(size),
              next_gen_genotypes(size),
              parent_selections(size),
              mate_selections(size),
              fitness_scores(size),
              videos(size) {
            rngs.reset();
        }
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
    // two pointers so that the the new data is "current" and the old data is
    // available for computing the next generation.
    d->curr_gen_genotypes.swap(d->next_gen_genotypes);
}

void Simulator::simulate(FitnessGoal goal, bool record) {
    simulate_population(
            size, num_species, goal, d->programs,
            d->curr_gen_genotypes, d->videos, d->fitness_scores, record);
}


// ---------------------------------------------------------------------------
// Methods to retrieve simulation results computed by simulate()
// ---------------------------------------------------------------------------

const Fitness* Simulator::get_fitness_scores() const {
    return d->fitness_scores.copy_to_host();
}

// TODO: Consider optimizing this to only copy the videos you want.
const Video* Simulator::get_videos() const {
    return d->videos.copy_to_host();
}

const Genotype* Simulator::get_genotypes() const {
    return d->curr_gen_genotypes.copy_to_host();
}


// ---------------------------------------------------------------------------
// Methods to manage RNG state
// ---------------------------------------------------------------------------

const std::vector<unsigned char> Simulator::get_state() const {
    return d->rngs.get_state();
}

void Simulator::restore_state(std::vector<unsigned char> state) {
    d->rngs.restore_state(state);
}

void Simulator::reset_state() {
    d->rngs.reset();
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
        programs[i].ops[0].type = OperationType::TILE;
        programs[i].ops[0].next_op_index = 1;
        programs[i].ops[1].type = OperationType::DRAW;
        programs[i].ops[1].next_op_index = STOP_INDEX;
    }
    simulator.populate(programs);
    for (int i = 0; i < 199; i++) {
        simulator.simulate(goal);
        simulator.propagate();
        // Include this GPU->host data transfer that the real project requires.
        simulator.get_fitness_scores();
    }
    simulator.simulate(goal, true);
    // Include this GPU->host data transfer that the real project requires.
    simulator.get_videos();
    return 0;
}
