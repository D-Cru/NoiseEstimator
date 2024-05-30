NoiseEstimator
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/d-cru/NoiseEstimator/workflows/CI/badge.svg)](https://github.com/d-cru/NoiseEstimator/actions?query=workflow%3ACI)
[![Docker-WebApp](https://github.com/D-Cru/NoiseEstimator/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/D-Cru/NoiseEstimator/actions/workflows/docker-publish.yml)
[![Upload Python Package](https://github.com/D-Cru/NoiseEstimator/actions/workflows/python-publish.yml/badge.svg)](https://github.com/D-Cru/NoiseEstimator/actions/workflows/python-publish.yml)
<!-- [![codecov](https://codecov.io/gh/d-cru/NoiseEstimator/branch/main/graph/badge.svg)](https://codecov.io/gh/d-cru/NoiseEstimator/branch/main) -->
![PyPI - Version](https://img.shields.io/pypi/v/noiseestimator)

# Noise estimation of ML datasets in chemistry

This repository contains a collection of notebooks and scripts to estimate performance bounds for ML models based on experimental errors of the underlying datasets. This is particularly useful for ML applications in chemistry, materials science, and drug-discovery, where the experimental errors can be large and the datasets are often small. 
If ML models are trained and evaluated on datasets with experimental errors, there is an inherent limit to the performance that can be achieved. This limit is determined by the noise in the dataset, and can be estimated using the methodology presented here. Models that perform better than this limit are likely overfitting to the noise in the dataset, and may not generalize well to new data.

We first consider addition of noise to synthetic datasets, to understand general trends. We then apply the same methodology to application datasets from drug-discovery, chemistry and materials science domains and compare to state-of-the-art model performance.

For details on the methodology, see the accompanying preprint: [https://doi.org/10.26434/chemrxiv-2024-z0pz7](https://doi.org/10.26434/chemrxiv-2024-z0pz7)

To facilitate use of this methodology, we provide a Python package `NoiseEstimator` that can be used to estimate performance bounds based on experimental error in a dataset.

In addition, we provide a web application that allows users to explore the methodology interactively. Users can look at synthetic datasets, vary the noise level, and see how the noise affects the performance of ML models. It is also possible for users to 
explore the nine real-world datasets used in this study. Users can also upload their own datasets and estimate performance bounds of their dataset. Use of the web application does not require any software installation and is freely available at: [https://noiseestimator.bioch.ox.ac.uk](https://noiseestimator.bioch.ox.ac.uk)

Following is an overview of the notebooks and scripts in this repository. The notebooks contain all the code used to estimate noise for synthetic and real-world datasets in the pre-print and are provided here for reproducibility.

## General case for synthetic datasets

The following notebooks in the `notebooks` folder were used to estimate noise for synthetic datasets:

* `general_noise.ipynb`: Notebook estimating performance bounds for a synthetic dataset with range [0, 1].
  This is to infer general trends about performance bounds when varying dataset size N and the level of noise.
* `noise_drawing.ipynb`: Notebook to further illustrate the method of adding noise to synthetic datasets to estimate performance bounds. Shown are how different levels of noise affect correleation in a synthetic dataset. 
* `non_uniform_noise.ipynb`: Notebook estimating performance bounds for a synthetic dataset with noise of two levels, depending on the range of the data (i.e. high noise for low values, low noise for high values).


## Application to real-world datasets

The following datasets and accompanying notebooks in the `notebooks/` folder were used to estimate noise for real-world datasets:
* BACE: `d_BACE.ipynb`
* Caco2 Wang permeability: `d_caco2_wang.ipynb`
* CASF-2016: `d_casf2016-core.ipynb`
* Lipophilicity: `d_lipophilicity.ipynb`
* matbench_expt_gap: `d_matbench_expt_gap.ipynb`
* rxnpredict: `d_rxnpredict_Buchwald.ipynb`
* Rzepiela: `d_rzepiela.ipynb`
* AqSolDB: `d_solubility.ipynb`

An overview figure of how the derived performance bounds for these datasets compare to performance of state of the art ML models was generated with `all_datasets.ipynb`.

For reproducibility, we provide the labels for all dataset in `data/processed/`.

The produced figures are in the `reports/` folder. 

The subfolder `reports/leaderboards` contains a printout of the relevant leaderboards for some of the datasets.

The experimental error estimates were derived as follows:

* BACE and CASF 2016: The error estimates are based on a systematic study of duplicate values in the ChEMBL database. [1] This analysis is based on 2,540 systems with 7,667 measurements.

* Lipophilicity: The error estimate is based on the experimental standard deviation for the specific assay [2]. The standard deviation is based on a set of 22 compounds with literature reference values, to which the assay was compared to. The data for this is in `data/external/lipophilicity` and the standard deviation is computed in the `d_lipophilicity.ipynb` notebook.

* Caco-2: The error estimate is based on an inter-lab comparison study [3] of 10 compounds, measured in a total of 7 different laboratories. The error estimation includes all possible 169 pairwise values, which yields an experimental standard deviation of 0.41 log units.  Computation is in the `d_caco2_wang.ipynb` notebook.

* Rzepiela: The error estimate of the Pampa permeability dataset was reported by the authors for high and low permeability data points separately. 

* Solubility (AqSolDB): The AqSolDB dataset is a carefully curated dataset based on multiple previous datasets. We went back to the raw data and the error estimate is based on 9813 compounds with duplicate values. The code for this is in `scripts/aqsoldb-error-estimation.py` and requires raw data from https://doi.org/10.24433/CO.1992938.v1. 

* Buchwald-Hartwig HTE: The standard deviation is based on 64 duplicate value pairs from the original assay publication [4]: (see SI, Data S5, Experiment 9). For details, see `data/external/rxn_Buchwald-Hartwig/Buchwald-Hartwig HTE.xlsx`

* Matbench expt_gap: The curated matbench experimental band gap dataset is based on a matminer dataset, which contains duplicate values. A total of 462 compounds have duplicate values, which we used to estimate the experimental standard deviation in the notebook `d_matbench_expt_gap.ipynb`. 



## NoiseEstimator package

The `NoiseEstimator` package contains a class `NoiseEstimator` that can be used to estimate the noise in a dataset. The package is available on PyPI and can be installed using pip:

```bash
pip install NoiseEstimator
```

Necessary dependencies should be resolved automatically via `pip`. We recommend installation in a virtual or conda environment and use of Python 3.9 - 3.11. The conda environment definition used for development is in the file `noise.yml`, and can be setup via 

```bash
conda env create -f noise.yml
```

The code for the NoiseEstimator package is in the folder `noiseestimator/`.

## NoiseEstimator webapp

The source code for the web application built with streamlit and hosted at [https://noiseestimator.bioch.ox.ac.uk](https://noiseestimator.bioch.ox.ac.uk) can be found in app.py. Deployment of the application is via a Docker container defined in the `Dockerfile`. 


## Aspirational to-do's for additional functionality / more robustness
TODO: Add more tests for the `NoiseEstimator` class \
TODO: make 2 subclasses, 1 for regression, 1 for classification \
TODO: add a method to better specify the types of metrics to use \(i.e. allow custom metric functions) (via dictionary select which ones, etc.) \
TODO: datasets: write json files instead of having a big csv file... \
TODO: add tests for web application

### Copyright

Copyright (c) 2024, Daniel Crusius


### Acknowledgements

References: \
[1] C. Kramer, T. Kalliokoski, P. Gedeck and A. Vulpetti, J Med Chem, 2012, 55, 5165-5173 \
[2] M. C. Wenlock, T. Potter, P. Barton and R. P. Austin, J Biomol Screen, 2011, 16, 348-355 \
[3] J. B. Lee, A. Zgair, D. A. Taha, X. Zang, L. Kagan, T. H. Kim, M. G. Kim, H. Y. Yun, P. M. Fischer and P. Gershkovich, Eur J Pharm Biopharm, 2017, 114, 38-42 \
[4] J. B. Lee, A. Zgair, D. A. Taha, X. Zang, L. Kagan, T. H. Kim, M. G. Kim, H. Y. Yun, P. M. Fischer and P. Gershkovich, Eur J Pharm Biopharm, 2017, 114, 38-42
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
