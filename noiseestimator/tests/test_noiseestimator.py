"""
Unit and regression test for the noiseestimator package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

from noiseestimator.noiseestimator import NoiseEstimator
import numpy as np
import pandas as pd
import pytest
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, pearsonr, matthews_corrcoef, roc_auc_score
import seaborn as sns
import matplotlib

def test_noiseestimator_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "noiseestimator" in sys.modules


# Create fixtures for the tests
@pytest.fixture
def dataset():
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})


# Test the initialization of NoiseEstimator class
def test_noiseestimator_init(dataset):
    noise_levels = [0.1, 0.2, 0.3]
    predictor_noise_level = 0.05
    noise_type = 'gaussian'
    n_bootstrap = 1000
    classifier = False
    asym_bound = 2.5
    asym_noise_up = 0.1
    asym_noise_low = 0.2
    class_barrier = 3
    class_labels = [0, 1, 0]
    
    estimator = NoiseEstimator(dataset.B, noise_levels, predictor_noise_level, noise_type, n_bootstrap, classifier, asym_bound, asym_noise_up, asym_noise_low, class_barrier, class_labels)
    
    assert estimator.dataset.equals(dataset.B)
    assert estimator.noise_levels == noise_levels
    assert estimator.predictor_noise_level == predictor_noise_level
    assert estimator.noise_type == noise_type
    assert estimator.n_bootstrap == n_bootstrap
    assert estimator.classifier == classifier
    assert estimator.asym_bound == asym_bound
    assert estimator.asym_noise_up == asym_noise_up
    assert estimator.asym_noise_low == asym_noise_low
    # assert estimator.barrier == class_barrier
    # assert estimator.dataset_labels == class_labels

# Test the estimate method of NoiseEstimator class
def test_noiseestimator_estimate():
    dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    noise_levels = [0.1, 0.2, 0.3]
    estimator = NoiseEstimator(dataset.B, noise_levels)
    
    output_df = estimator.estimate()
    
    assert all(output_df.columns == ['original', 'noise_0.1', 'noise_0.2', 'noise_0.3'])
    assert output_df.shape == (3, 4)

# Test the estimate_multi_bootstrap method of NoiseEstimator class
def test_noiseestimator_estimate_multi_bootstrap():
    dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    noise_levels = [0.1, 0.2, 0.3]
    n_bootstrap = 1000
    estimator = NoiseEstimator(dataset.B, noise_levels, n_bootstrap=n_bootstrap)
    
    output_df = estimator.estimate_multi_bootstrap()
    
    assert all(output_df.columns == ['mae', 'mse', 'rmse', 'r2', 'pearsonr', 'noise'])
    assert output_df.shape[0] == len(noise_levels) * n_bootstrap

# Test the plot method of NoiseEstimator class
def test_noiseestimator_plot():
    dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    noise_levels = [0.1, 0.2, 0.3]
    estimator = NoiseEstimator(dataset.B, noise_levels)
    
    plot = estimator.plot()
    
    assert isinstance(plot, sns.axisgrid.FacetGrid)

# Test the plot_bootstrap method of NoiseEstimator class
def test_noiseestimator_plot_bootstrap():
    dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    noise_levels = [0.1, 0.2, 0.3]
    n_bootstrap = 1000
    estimator = NoiseEstimator(dataset.B, noise_levels, n_bootstrap=n_bootstrap)
    
    plot = estimator.plot_bootstrap()
    print(type(plot))
    
    assert isinstance(plot, matplotlib.axes._axes.Axes)

# Test the _estimate_gaussian method of NoiseEstimator class
def test_noiseestimator_estimate_gaussian():
    dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    noise_level = 0.1
    estimator = NoiseEstimator(dataset.B, [noise_level])
    
    output = estimator._estimate_gaussian(noise_level)
    
    assert output.shape == dataset.B.shape

# Test the _estimate_asymmetric method of NoiseEstimator class
def test_noiseestimator_estimate_asymmetric():
    dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    noise_level = 0.1
    asym_bound = 2.5
    asym_noise_up = 0.1
    asym_noise_low = 0.2
    estimator = NoiseEstimator(dataset.B, [noise_level], asym_bound=asym_bound, asym_noise_up=asym_noise_up, asym_noise_low=asym_noise_low)
    
    output = estimator._estimate_asymmetric(noise_level)
    
    assert output.shape == dataset.B.shape

# Test the _estimate_uniform method of NoiseEstimator class
def test_noiseestimator_estimate_uniform():
    dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    noise_level = 0.1
    estimator = NoiseEstimator(dataset.B, [noise_level])
    
    with pytest.raises(NotImplementedError):
        estimator._estimate_uniform(noise_level)