#ifndef __INTERPRETER_H__
#define __INTERPRETER_H__

/*
 * Classes representing a program for converting a Genotype into a Phenotype.
 *
 * The Interpreter is effectively a graph of function calls bound to arguments.
 * This would be natural to represent with a tree-like strucutre, but these
 * classes are designed to be allocated into a fixed block of memory. This
 * makes it much easier to pass Interpreter objects from Python to C++ and from
 * host to device.
 */

#include <vector>

#include "development.cuh"
#include "environment.h"
#include "genotype.cuh"

namespace epigenetic_gol_kernel {

// Fixed allocation sizes for Interpreter programs.
constexpr unsigned int MAX_OPERATIONS = 10;
constexpr unsigned int MAX_ARGUMENTS = 4;

// Program counter constants indicating where to start and stop.
constexpr unsigned int START_INDEX = 0;
constexpr unsigned int STOP_INDEX = MAX_OPERATIONS;

// An enum representing how to bias data read from the genotype. This may get
// more interesting in future versions.
enum class BiasMode {
    NONE,        // Use gene value from genotype unmodified.
    FIXED_VALUE, // Use a fixed value instead of reading from the genotype.
    SIZE
};

class BoundArguments;

// A class for binding a value from the genotype to an argument for an
// Operation function.
class Argument {
    friend class BoundArguments;
    private:
        // Which gene to read from.
        int gene_index = 0;

        // Restrict genes values baked into the interpreter to bias the results
        // read from the genotype.
        union {
            unsigned int scalar;
            Stamp stamp;
        } gene_bias;
        BiasMode bias_mode = BiasMode::NONE;

    public:
        // Methods for setting state. To make it easy to manage whole
        // Interpreters as blocks of pre-allocated memory that can be copied in
        // bulk, these are methods and not constructors.
        __device__ void init(int index);
        __device__ void init(
                int index, unsigned int scalar_bias,
                BiasMode mode=BiasMode::FIXED_VALUE);
        __device__ void init(
                int index, const Stamp& stamp_bias,
                BiasMode mode=BiasMode::FIXED_VALUE);
};

// A utility class for binding an array of Arguments with the gene data
// they read from to avoid having to pass Genotypes around.
class BoundArguments {
    private:
        const Argument* args;
        const Genotype& genotype;

    public:
        __device__ BoundArguments(
                const Argument* args, const Genotype& genotype)
            : args(args), genotype(genotype) {}

        // Get the value for this Argument from bound Genotype. Any argument
        // can be used to read scalar or stamp gene values, making it easy for
        // Operations to mix and match argument types.
        __device__ unsigned int scalar_value(int index) const;
        __device__ const Stamp& stamp_value(int index) const;
};

// A class representing one step in the development of a phenotype. It's
// basically a wrapper for a function call with arguments.
struct Operation {
    OperationType type = OperationType::SIZE;
    Argument args[MAX_ARGUMENTS] = {};
    unsigned int next_op_index = STOP_INDEX;
};

// A class representing a program for developing a genotype into a phenotype.
// It's basically just a collection of operations and a method to execute them.
class Interpreter {
    public:
        Operation ops[MAX_OPERATIONS] = {};

        __device__ Interpreter() {}
        __device__ Interpreter(const Interpreter& other);

        __device__ virtual void run(
                const Genotype& genotype, int row, int col, Cell& cell) const;
};

// A default Interpreter for tests that don't care which one they use.
class DefaultInterpreter : public Interpreter {
    public:
        __device__ DefaultInterpreter();
};

// An Interpreter for tests that just reproduces the provided phenotype,
// ignoring the Genotype all together.
class MockInterpreter : public Interpreter {
    private:
        const Frame* injected_phenotype;

    public:
        __device__ MockInterpreter(const Frame* injected_phenotype)
            : injected_phenotype(injected_phenotype) {}

        __device__ void run(
                const Genotype&, int row, int col, Cell& cell) const override;
};

} // namespace epigenetic_gol_kernel

#endif
