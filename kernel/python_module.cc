#include <cstdio>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "environment.h"
#include "gol_simulation.h"
#include "phenotype_program.h"
#include "reproduction.h"
#include "selection.h"
#include "simulator.h"

namespace epigenetic_gol_kernel {

namespace py = pybind11;

void bind_environment(py::module_& m) {
    py::enum_<FitnessGoal>(m, "FitnessGoal")
        .value("EXPLODE", FitnessGoal::EXPLODE)
        .value("GLIDERS", FitnessGoal::GLIDERS)
        .value("LEFT_TO_RIGHT", FitnessGoal::LEFT_TO_RIGHT)
        .value("STILL_LIFE", FitnessGoal::STILL_LIFE)
        .value("SYMMETRY", FitnessGoal::SYMMETRY)
        .value("THREE_CYCLE", FitnessGoal::THREE_CYCLE)
        .value("TWO_CYCLE", FitnessGoal::TWO_CYCLE)
        .export_values();
    py::enum_<Cell>(m, "Cell")
        .value("ALIVE", Cell::ALIVE)
        .value("DEAD", Cell::DEAD)
        .export_values();
    m.attr("FitnessDType") = py::dtype::of<Fitness>();
    m.attr("WORLD_SIZE") = py::int_(WORLD_SIZE);
    m.attr("NUM_STEPS") = py::int_(NUM_STEPS);
}

void bind_gol_simulation(py::module_& m) {
    m.def("simulate_phenotype",
        [](const py::array_t<Cell> phenotype) {
            Frame* phenotype_data = (Frame*) &phenotype.unchecked<2>()(0, 0);
            return py::array(
                    py::dtype::of<Cell>(),
                    {NUM_STEPS, WORLD_SIZE, WORLD_SIZE},
                    simulate_phenotype(*phenotype_data));
        });
    m.def("render_phenotype",
        [](py::array_t<PhenotypeProgram> program) {
            const PhenotypeProgram& program_data = program.unchecked<0>()();
            return py::array(
                    py::dtype::of<Cell>(),
                    {WORLD_SIZE, WORLD_SIZE},
                    render_phenotype(program_data));
        });
    m.def("simulate_organism",
        [](py::array_t<PhenotypeProgram> program,
           py::array_t<Genotype> genotype) {
            const PhenotypeProgram& program_data = program.unchecked<0>()();
            const Genotype& genotype_data = genotype.unchecked<0>()();
            return py::array_t<Cell>(
                {NUM_STEPS, WORLD_SIZE, WORLD_SIZE},
                (Cell*) simulate_organism(program_data, genotype_data));
        });
}

void bind_phenotype_program(py::module_& m) {
    py::enum_<TransformMode>(m, "TransformMode")
        .value("NONE", TransformMode::NONE)
        .value("ALIGN", TransformMode::ALIGN)
        .value("ARRAY_1D", TransformMode::ARRAY_1D)
        .value("ARRAY_2D", TransformMode::ARRAY_2D)
        .value("COPY", TransformMode::COPY)
        .value("CROP", TransformMode::CROP)
        .value("DRAW", TransformMode::DRAW)
        .value("FLIP", TransformMode::FLIP)
        .value("MIRROR", TransformMode::MIRROR)
        .value("QUARTER", TransformMode::QUARTER)
        .value("ROTATE", TransformMode::ROTATE)
        .value("SCALE", TransformMode::SCALE)
        .value("TEST", TransformMode::TEST)
        .value("TILE", TransformMode::TILE)
        .value("TRANSLATE", TransformMode::TRANSLATE)
        .value("SIZE", TransformMode::SIZE);
    py::enum_<BiasMode>(m, "BiasMode")
        .value("NONE", BiasMode::NONE)
        .value("FIXED_VALUE", BiasMode::FIXED_VALUE)
        .value("SIZE", BiasMode::SIZE);
    py::enum_<ComposeMode>(m, "ComposeMode")
        .value("NONE", ComposeMode::NONE)
        .value("OR", ComposeMode::OR)
        .value("XOR", ComposeMode::XOR)
        .value("AND", ComposeMode::AND)
        .value("SIZE", ComposeMode::SIZE);
    m.attr("GenotypeDType") = py::dtype::of<Genotype>();
    m.attr("PhenotypeProgramDType") = py::dtype::of<PhenotypeProgram>();
    m.attr("STAMP_SIZE") = py::int_(STAMP_SIZE);
    m.attr("CELLS_PER_STAMP") = py::int_(CELLS_PER_STAMP);
    m.attr("NUM_GENES") = py::int_(NUM_GENES);
    m.attr("MAX_OPERATIONS") = py::int_(MAX_OPERATIONS);
    m.attr("MAX_ARGUMENTS") = py::int_(MAX_ARGUMENTS);
}

void bind_reproduction(py::module_& m) {
    m.def("breed_population",
        [](const py::array_t<Genotype> genotypes,
           std::vector<unsigned int> parent_selections,
           std::vector<unsigned int> mate_selections) {
            const Genotype* genotypes_data =
                &genotypes.unchecked<3>()(0, 0, 0);
            return py::array_t<Genotype>(
                {genotypes.shape(0),
                 genotypes.shape(1),
                 genotypes.shape(2),},
                breed_population(
                    genotypes_data, parent_selections, mate_selections));
        });
    m.attr("CROSSOVER_RATE") = py::float_(CROSSOVER_RATE);
    m.attr("MUTATION_RATE") = py::float_(MUTATION_RATE);
}

void bind_selection(py::module_& m) {
    m.def("select", &select);
}

void bind_simulator(py::module_& m) {
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<int, int, int>())
        .def("populate",
            [](Simulator& simulator,
               const py::array_t<PhenotypeProgram> programs) {
                const PhenotypeProgram* programs_data =
                    &programs.unchecked<1>()(0);
                simulator.populate(programs_data);
            })
        .def("propagate", &Simulator::propagate)
        .def("simulate", &Simulator::simulate)
        .def("simulate_and_record",
            [](Simulator& simulator, const FitnessGoal& goal) {
                return py::array(
                    py::dtype::of<Cell>(),
                    {simulator.num_species,
                     simulator.num_trials,
                     simulator.num_organisms,
                     NUM_STEPS, WORLD_SIZE, WORLD_SIZE},
                    (Cell*) simulator.simulate_and_record(goal));
            })
        .def("evolve",
            [](Simulator& simulator,
               const py::array_t<PhenotypeProgram> programs,
               const FitnessGoal& goal,
               const int num_generations) {
                const PhenotypeProgram* programs_data =
                    &programs.unchecked<1>()(0);
                return py::array(
                    py::dtype::of<Fitness>(),
                    {simulator.num_species,
                     simulator.num_trials,
                     simulator.num_organisms},
                    simulator.evolve(programs_data, goal, num_generations));
            })
        .def("get_fitness_scores",
            [](const Simulator& simulator) {
                return py::array(
                    py::dtype::of<Fitness>(),
                    {simulator.num_species,
                     simulator.num_trials,
                     simulator.num_organisms},
                    simulator.get_fitness_scores());
            })
        .def("get_genotypes",
            [](const Simulator& simulator) {
                return py::array_t<Genotype>(
                    {simulator.num_species,
                     simulator.num_trials,
                     simulator.num_organisms},
                    simulator.get_genotypes());
            })
        .def("seed", &Simulator::seed);
}

/*
 * Bindings for C++ types and functions to expose to Python.
 *
 * The primary interface with Python is the Simulator class, which is mirrored
 * more or less 1:1 in the Python interface. There are also a few data types,
 * constants, and miscellaneous functions that are important for working with
 * the Simulator class and testing it.
 */
PYBIND11_MODULE(kernel, m) {
    PYBIND11_NUMPY_DTYPE(Genotype, scalar_genes, stamp_genes);
    PYBIND11_NUMPY_DTYPE(ScalarArgument, gene_index, bias, bias_mode);
    PYBIND11_NUMPY_DTYPE(StampArgument, gene_index, bias, bias_mode);
    PYBIND11_NUMPY_DTYPE(TransformOperation, type, args);
    PYBIND11_NUMPY_DTYPE(DrawOperation, compose_mode, stamp,
                         global_transforms, stamp_transforms);
    PYBIND11_NUMPY_DTYPE(PhenotypeProgram, draw_ops);

    bind_environment(m);
    bind_gol_simulation(m);
    bind_phenotype_program(m);
    bind_reproduction(m);
    bind_selection(m);
    bind_simulator(m);
}

} // namespace epigenetic_gol_kernel

/*
<%
setup_pybind11(cfg)
%>
*/
