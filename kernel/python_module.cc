#include <cstdio>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "environment.h"
#include "simulator.h"

namespace epigenetic_gol_kernel {

namespace py = pybind11;

PYBIND11_MODULE(kernel, m) {
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<int, int, int>())
        .def("populate",
            [](Simulator& simulator) {
                // TODO: Support passing Interpreters from Python.
                simulator.populate();
            })
        .def("propagate", &Simulator::propagate)
        .def("simulate", &Simulator::simulate,
            py::arg("goal"), py::arg("record") = false)
        .def("get_fitness_scores",
            [](const Simulator& simulator) {
                return py::array(
                    py::dtype::of<Fitness>(),
                    {simulator.num_species,
                     simulator.num_trials,
                     simulator.num_organisms},
                    simulator.get_fitness_scores());
            })
        .def("get_videos",
            [](const Simulator& simulator) {
                return py::array(
                    py::dtype::of<Cell>(),
                    {simulator.num_species,
                     simulator.num_trials,
                     simulator.num_organisms,
                     NUM_STEPS, WORLD_SIZE, WORLD_SIZE},
                    (Cell*) simulator.get_videos());
            })
        .def("get_genotypes",
            [](const Simulator& simulator) {
                // TODO: Figure out how to return Genotype objects as a numpy
                // structured array.
            })
        .def("get_state", &Simulator::get_state)
        .def("restore_state", &Simulator::restore_state)
        .def("reset_state", &Simulator::reset_state)
        .def_readonly("num_species", &Simulator::num_species)
        .def_readonly("num_trials", &Simulator::num_trials)
        .def_readonly("num_organisms", &Simulator::num_organisms)
        .def_readonly("size", &Simulator::size);
    py::class_<TestSimulator, Simulator>(m, "TestSimulator")
        .def(py::init<int, int, int>())
        .def("simulate_phenotype",
            [](TestSimulator& simulator,
                py::array_t<Cell, py::array::c_style> h_phenotypes,
                FitnessGoal goal, bool record) {
               if (h_phenotypes.ndim() != 2 ||
                       h_phenotypes.shape()[0] != WORLD_SIZE ||
                       h_phenotypes.shape()[1] != WORLD_SIZE) {
                   fprintf(
                        stderr,
                        "Injected phenotype must have size %d x %d\n",
                        WORLD_SIZE, WORLD_SIZE);
                   return;
               }
               Frame* raw_data = (Frame*) &h_phenotypes.unchecked<2>()(0, 0);
               simulator.simulate_phenotype(raw_data, goal, record);
            },
            py::arg("h_phenotypes"),
            py::arg("goal"),
            py::arg("record") = false);
    py::enum_<FitnessGoal>(m, "FitnessGoal")
        .value("STILL_LIFE", FitnessGoal::STILL_LIFE)
        .value("TWO_CYCLE", FitnessGoal::TWO_CYCLE)
        .export_values();
    m.attr("WORLD_SIZE") = py::int_(WORLD_SIZE);
    m.attr("NUM_STEPS") = py::int_(NUM_STEPS);
    py::enum_<Cell>(m, "Cell")
        .value("ALIVE", Cell::ALIVE)
        .value("DEAD", Cell::DEAD)
        .export_values();
}

} // namespace epigenetic_gol_kernel

/*
<%
setup_pybind11(cfg)
%>
*/
