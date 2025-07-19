# Topological Point Features 🪴

This is the python package for topological point features (TOPF), enabling the construction of point-level features in point clouds stemming from algebraic topology and differential geometry as described in [Point-Level Topological Representation Learning on Point Clouds](https://arxiv.org/abs/2406.02300), which has been accepted at ICML 2025. 🪴

![Example of TOPF on three point clouds](https://github.com/vincent-grande/topf/blob/main/examples/teaserfigure.png?raw=True)

## Installation

Although being a python package, TOPF requires an installation [Julia](https://julialang.org/downloads/) because it uses the wonderful package [`Ripserer.jl`](https://mtsch.github.io/Ripserer.jl/dev/). After having installed Julia and set up PATH variables, you can install TOPF simply by running

    pip install topf

TOPF currently works under `macOS` and `Linux`. `Windows` is not supported.

## Usage

Two Jupyter-Notebooks with example usage of TOPF with [basic examples](https://github.com/vincent-grande/topf/blob/main/examples/topf_basic_examples.ipynb) and [3d examples](https://github.com/vincent-grande/topf/blob/main/examples/topf_examples3d.ipynb) can be found in the examples folder.

## Citation

TOPF is based on the paper 'Point-Level Topological Representation Learning on Point Clouds', Vincent P. Grande and Michael T. Schaub, Proceedings of the 42nd International Conference on Machine Learning 2025.
If you find TOPF useful, please consider citing the paper:

    @inproceedings{grande2025topf,
     title={Point-Level Topological Representation Learning on Point Clouds},
     author={Vincent P Grande and Michael T Schaub},
     booktitle={International Conference on Machine Learning},
     year={2025}
}

## Dependencies

TOPF depends on [`Julia`](https://julialang.org), the Julia package [`Ripserer.jl`](https://mtsch.github.io/Ripserer.jl/dev/), [`Python`](https://www.python.org) and the Python packages [`numpy`](https://numpy.org), [`gudhi`](https://gudhi.inria.fr), [`matplotlib`](https://matplotlib.org), [`scikit-learn`](https://scikit-learn.org/stable/), [`scipy`](https://scipy.org), [`pandas`](https://pandas.pydata.org), and [`plotly`](https://plotly.com). The idea of how to fix Z/3Z cycles with faulty lifts to real coefficients was inspired by [DreiMac](https://dreimac.scikit-tda.org/en/latest/)'s solution to the problem (for cocycles).

## Feedback

Any feedback, comments, or bug reports are welcome! Simply write an email to [Vincent](https://vincent-grande.github.io). 