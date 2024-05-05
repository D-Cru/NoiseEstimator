NoiseEstimator
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/d-cru/NoiseEstimator/workflows/CI/badge.svg)](https://github.com/d-cru/NoiseEstimator/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/d-cru/NoiseEstimator/branch/main/graph/badge.svg)](https://codecov.io/gh/d-cru/NoiseEstimator/branch/main)

# Noise estimation of ML datasets in chemistry

Estimate maximum performance bounds based on experimental errors for ML datasets

## Notebook contents

* `non_uniform_noise.ipynb`: Notebook estimating noise of varying levels, depending
  on the range of the data. (i.e. high noise for low values, low noise for high values)
* `general_noise.ipynb`: Notebook estimating noise for a synthetic dataset between [0, 1].
  This is to infer general trends about dataset size n, noise level, and 
* `test_noise.ipynb`: Notebook testing the `NoiseEstimator` defined in `noise.py`
* `matbench.ipynb`: Notebook estimating noise for the `matbench` dataset.
* `matbench_expt_gap.ipynb`: Notebook estimating noise for the `matbench_expt_gap` dataset.
* `rxnpredict_error.ipynb`: Notebook estimating noise for the `rxnpredict` datasets.
* `add_noise.ipynb`: Notebook adding noise to the `Caco2 Wang permeability` dataset from `tdc`.



## To-do's
TODO: Add more tests for the `NoiseEstimator` class
TODO: make 2 subclasses, 1 for regression, 1 for classification
TODO: add a method to better specify the types of metrics to use (i.e. allow custom metric functions) (via dictionary select which ones, etc.)
TODO: datasets: write json files instead of having a big csv file...
TODO: add tests for web application

### Copyright

Copyright (c) 2024, Daniel Crusius


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
