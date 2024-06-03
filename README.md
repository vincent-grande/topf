# Topological Point Features

This is the python package for topological point features (TOPF), enabling the construction of point-level features in point clouds stemming from algebraic topology and differential geometry.

## Installation
Although being a python package, TOPF requires an installation [Julia](https://julialang.org/downloads/) because it uses the wonderful package [`Ripserer.jl`](https://mtsch.github.io/Ripserer.jl/dev/). After having installed Julia and set up PATH variables, you can install TOPF simply by

    pip install topf

## Usage

Two Jupyter-Notebooks with example usage of TOPF with [basic examples](https://github.com/vincent-grande/topf/exxamples/topf_basic_examples.ipynb) and [3d examples](https://github.com/vincent-grande/topf/examples/topf_examples3d.ipynb) can be found in the examples folder.

## Citation

TOPF is based on the paper 'Node-Level Topological Representation Learning on Point Clouds', Vincent P. Grande and Michael T. Schaub, 2024:

    @article{Grande2024topf,
        title={Node-Level Topological Representation Learning on Point Clouds},
        author={Grande, Vincent P. and Schaub, Michael T.},
        year={2024}
    }