Project by Nate Gaylinn:
[github](https://github.com/ngaylinn),
[email](mailto:nate.gaylinn@gmail.com),
[blog](https://thinkingwithnate.wordpress.com/)

# Overview

This project is a demonstration of an experimental evolutionary algorithm design (tentatively named an "epigenetic algorithm"). It's inspired by a simple observation: a gene sequence alone does not describe an organism, because amino acids don't have any inherent *meaning* in nature. Instead, the cell must *interpret* the gene sequence in order to build and operate a body, and that interpretation process was evolved along with the gene sequence. This project mimics that structure, evolving programs to solve a fitness challenge while also evolving an interpreter for those programs. The result is an algorithm that can search over a large space of possible phenotypes by learning which regions of that search space are best for evolving solutions to the fitness challenge.

This project is based on an earlier [prototype](https://github.com/ngaylinn/epigenetic-gol-prototype). Since then, the goals of the project have become clearer. The design is simpler, in that it does fewer things, but more complicated, in that it does them in a more optimized way.

This algorithm is not yet particularly useful or scientifically informative. It's meant as an exploration of a wild idea, to justify [further exploration](#future-work)

## Game of Life

This demo uses Conway's Game of Life (GOL) as a playground for evolutionary
algorithms. For those not familiar with the GOL, you may want to check out this
[introduction ](https://conwaylife.com/wiki/Conway's_Game_of_Life) and try
playing with this [interactive demo](https://playgameoflife.com/).

The GOL was chosen for a few reasons. First and foremost, this project is meant
to evolve *programs* that generate solutions, not just solutions themselves.
This will be important for [future work](#future-work), where this algorithm
may be adapted to different sorts of programming tasks. As computer programs
go, a GOL simulation is about as simple as they come. It takes no input, it's
fully deterministic, has no dependencies, and aside from the game board itself,
it has no state or output.

Other reasons for using the GOL are that it's relatively well known and
produces nice visuals that make it clear how well the genetic algorithm is
performing. It's also easy to optimize with [parallel
execution](#inner-loop-design).

That said, doing cool things with the GOL is *not* a primary goal for this
project. If you're a GOL aficionado and would like to help make this work more
interesting and useful to your community, your input would be greatly
appreciated! Please [contact the author](mailto:nate.gaylinn@gmail.com) for
possible collaborations.

## Motivation

A major practical limitation of evolutionary algorithms is that they require an
expert programmer to optimize the algorithm design to fit their specific use
case. They carefully design a variety of phenotypes and an encoding scheme to
describe them with a gene sequence. They devise clever mutation and crossover
strategies to make the search process more efficient. With luck and a great
deal of trial and error, this sometimes produces useful results. But why should a
programmer to do this hard work by hand, when life does the same thing via
evolution?

The mechanisms of epigenetics serve to manage the interpretation and variation
of the gene sequence. This project explores one way of doing the same thing in
an artificial context. Rather than fine-tuning the performance of a single
genetic algorithm designed for a single fitness goal, this is more like
searching over many possible genetic algorithms to find which one gets the best
traction on the problem at hand.

If successful, this might one day lead to genetic algorithms that are more like
deep learning foundation models. A powerful generic model could be built by
specialists, then automatically fine-tuned to a specific narrow task on demand.
It might also produce results that are more open-ended, flexible, and life-like
than traditional genetic algorithms.

In biology, the notion that life "evolves for evolvability" is still controversial.
The hope is this line of research might produce evidence in support of that hypothesis.
Although not very [biologically realistic](#biological-realism), this project
suggests that "evolving for evolvability" is an effective strategy that arises naturally
from the sort of nested "program and interpreter" design seen in nature.

# Technical Overview

This project uses a nested evolutionary algorithm. The ["outer loop"](#outer-loop-design) of the project evolves species&mdash;subsets of possible organisms, each with a range of possible forms. A species is represented by a `PhenotypeProgram`, which is used to turn a `Genotype` into a phenotype (in this case, a GOL simulation). The ["inner loop"](#inner-loop-design) holds the set of species constant and evolves a population of organisms for each `PhenotypeProgram`. It uses a process of [development](#genetics-and-development) to make a GOL simulation from each organism's `Genotype`, then scores that simulation based on one of several different fitness challenges. The fitness of a species is determined by how effective the inner loop is at evolving fit organisms.

The outer loop is implemented in Python, mostly because the code is simpler, more readable, and doesn't have to be high performance. It's purpose is to randomize and breed `PhenotypeProgram`s, run the inner loop, and analyze the results. The inner loop is written in C++ and is optimized for execution on a CUDA-enabled GPU. It's main purpose is to run many GOL simulations in parallel, though also evaluates the fitness of those simulations, and handles randomizing and breeding of organism populations. The primary interface between these two halves of the project are the `Simulator` class (which manages operations on populations of organisms in GPU memory) and the `PhenotypeProgram` data structure (which the outer loop evolves and the inner loop uses for development).

## Nested Evolution

TODO: Diagram of nested evolution

The primary challenge with evolving a program to interpret gene sequences is *stability*. A traditional evolutionary algorithm requires a stable and relativel smooth [fitness landscape](https://en.wikipedia.org/wiki/Fitness_landscape#:~:text=In%20evolutionary%20biology%2C%20fitness%20landscapes,often%20referred%20to%20as%20fitness) in order to "get traction" on a problem, meaning it can tell when it's making progress by evolving in fruitful directions. Allowing the meaning of the gene sequence to change completely breaks this. Even a small mutation to a `PhenotypeProgram` completely reshapes the fitness landscape, making hill climbing impossible. To work around that problem, this project breaks evolution into two phases. The outer loop continuously evolves a popoulation of `PhenotypeProgram`s (species). The inner loop evolves a population of `Genotypes` (organisms), but *not* continously. In each generation of the outer loop, the inner loop starts over with a randomized organism population. While this makes nested evolution possible, it's not [biologically realistic](#biological-realism), and requires [special intervention](#genetics-and-development) for genotype-level innovations to persist as species evolve.

Normally, to evolve GOL simulations with interesting qualities, a programmer would use their intuition to hand-design an evolutionary algorithm, using what they know about GOL and a specific fitness challenge to make the search process efficient enough to work. A major [motivation](#motivation) for this project is to avoid doing that, to search more broadly with less human bias. Instead, this algorithm is optimized to search over the domain of monochrome bitmap images, with no awareness of the GOL or any specific fitness challenge. This project provides many fitness challenges, chosen so that different evolutionary strategies are needed. This shows that a epigenetic algorithm can find effective evolutionary strategies to a variety of problems, all on its own.

An epigenetic algorithm produces two outputs: the best `PhenotypeProgram` (species) and `Genotype` (organism) for a given fitness challenge. These are combined to make the organism's phenotype, which is a GOL simulation. The fitness of that simulation is determined by counting which cells are alive and dead at different frames, looking for patterns that the programmer found "interesting." The fitness of a species is derived from those organism fitness scores, but in an indirect way. Species are evolved for *evolvability*, not fitness. As such, species fitness is determined by the performance of the inner loop&mdash;how well fitness of the whole population improved over many generations. This is approximated using a metric called the "weighted median integral."

TODO: Mention "weighted median integral" in the code comments, link to it here.

Since the inner and outer loops of this project have different notions of fitness, it's best to think of them as two separate evoluationary searches. The inner loop searches for interesting GOL simulations within the search space defined by a particular `PhenotypeProgram`. The outer loop searches for subsets of the larger search space where interesting GOL simulations can be found. In other words, an epigenetic algorithm doesn't search for fit organisms, it searches for evolutionary algorithms that produce fit organisms.

## Genetics and Development

## Inner-Loop Design

## Outer-Loop Design

## Execution

# Biological Realism

# Future Work


