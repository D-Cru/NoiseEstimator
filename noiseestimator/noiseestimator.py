"""Provide the primary functions."""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, matthews_corrcoef
from scipy.stats import pearsonr

class NoiseEstimator():
    def __init__(self, dataset, noise_levels, predictor_noise_level=None, noise_type='gaussian', n_bootstrap=1000, classifier=False, asym_bound=None, asym_noise_up=None, asym_noise_low=None, class_barrier=None, class_labels=None):
        self.dataset = dataset
        self.noise_levels = noise_levels
        self.noise_type = noise_type
        self.n_bootstrap = n_bootstrap
        self.classifier = classifier
        if self.noise_levels is None:
            raise ValueError('Noise levels must be non-empty')
        elif type(self.noise_levels) == int or type(self.noise_levels) == float or type(self.noise_levels) == np.float64:
            self.noise_levels = [self.noise_levels]

        self.asym_bound = asym_bound
        self.asym_noise_up = asym_noise_up
        self.asym_noise_low = asym_noise_low

        self.predictor_noise_level = predictor_noise_level
        if predictor_noise_level is None:
            self.predictor_noise_level = 0

        if self.classifier:
            if class_barrier is None:
                self.barrier = np.median(self.dataset)
            else:
                self.barrier = class_barrier
            if class_labels is None:
                self.dataset_labels = np.where(self.dataset >= self.barrier, 1, 0)
            else:
                self.dataset_labels = class_labels
        
        self.noise_estimates = self.estimate()
        if self.classifier:
            self.noisy_labels = self.noise_estimates.applymap(lambda x: 1 if x >= self.barrier else 0)

        self.noise_bootstraps = self.estimate_multi_bootstrap(n_bootstrap=self.n_bootstrap)
            


    def estimate(self, noise_levels=None):
        output = [self.dataset]
        if self.noise_type == 'gaussian':
            estimator = self._estimate_gaussian
        elif self.noise_type == 'uniform':
            estimator = self._estimate_uniform
        elif self.noise_type == 'asymmetric':
            estimator = self._estimate_asymmetric

        else:
            raise ValueError('Unknown noise type')
        
        if noise_levels is None:
            noise_levels = self.noise_levels
        
        for noise_level in noise_levels:
            output.append(estimator(noise_level))

        output_df = pd.DataFrame(output).T
        output_df.columns = ['original'] + [f'noise_{i}' for i in (noise_levels)]
        return output_df


    def estimate_multi_bootstrap(self, noise_levels=None, n_bootstrap=1000):
        output = []
        if noise_levels is None:
            noise_levels = self.noise_levels
        
        for noise_level in noise_levels:
            for i in range(n_bootstrap):
                noisy_data = self.estimate(noise_levels=[noise_level])
                # reorder the index to match the original dataset
                noisy_data = noisy_data.reindex(self.dataset.index)
                # for a realistic bound, we need to consider predictor noise
                if self.predictor_noise_level > 0:
                    predictor_data = self.estimate(noise_levels=[self.predictor_noise_level]).iloc[:, 1]
                # if no predictor noise, use the original dataset for max bound
                else:
                    predictor_data = self.dataset
                if not self.classifier:
                    output.append(
                        [mean_absolute_error(predictor_data, noisy_data.iloc[:, 1]),
                        mean_squared_error(predictor_data, noisy_data.iloc[:, 1]),
                        np.sqrt(mean_squared_error(predictor_data, noisy_data.iloc[:, 1])),
                        r2_score(predictor_data, noisy_data.iloc[:, 1]),
                        pearsonr(predictor_data, noisy_data.iloc[:, 1])[0],
                        f"noise_{noise_level}"]
                    )
                else:
                    labels_noisy = np.where(noisy_data.iloc[:, 1] > self.barrier, 1, 0)

                    output.append(
                        [matthews_corrcoef(self.dataset_labels, labels_noisy),
                        roc_auc_score(self.dataset_labels, labels_noisy),
                        f"noise_{noise_level}"]
                    )
        if self.classifier:
            error_df = pd.DataFrame(output, columns=['matthews_corrcoef', 'roc_auc', 'noise'])
        else:
            error_df = pd.DataFrame(output, columns=['mae', 'mse', 'rmse', 'r2', 'pearsonr', 'noise'])
        return error_df


    def plot(self, noise_df=None):
        if not self.classifier:
            if noise_df is None:
                noise_df = self.noise_estimates
            
            df_melt = noise_df.melt(id_vars='original', var_name='noise', value_name='value')
            # add column of 1s for the hue
            df_melt['hue'] = 1
            g = sns.lmplot(x='original', y='value', col='noise', data=df_melt, aspect=1.0, fit_reg=False, palette=['black', 'red', 'blue'], hue='hue', legend=False)
            for i in range(len(noise_df.columns) - 1):
                g.axes[0, i].set_title(f'Noise: {noise_df.columns[i+1]}')
                g.axes[0, i].set_xlabel('Original')
                g.axes[0, i].set_ylabel('Noisy')

            return g
        else:
            if noise_df is None:
                noise_df = self.noise_estimates
            
            df_melt = noise_df.melt(id_vars='original', var_name='noise', value_name='value')
            
            noise_melt = self.noisy_labels.melt(id_vars='original', var_name='noise', value_name='value')
            # now change the value column to either 'tp/tn' or 'fp' or 'fn' depending on the original label
            noise_melt['value'] = noise_melt.apply(lambda row: 'tp/tn' if row['original'] == row['value'] else 'fp' if row['original'] == 1 else 'fn', axis=1)
            df_melt['classifier'] = noise_melt['value']
            g = sns.lmplot(x='original', y='value', col='noise', hue='classifier', data=df_melt, aspect=1.0, palette=['black', 'red', 'blue'], fit_reg=False)
            for i in range(len(noise_df.columns) - 1):
                g.axes[0, i].set_title(f'Noise: {noise_df.columns[i+1]}')
                g.axes[0, i].set_xlabel('Original')
                g.axes[0, i].set_ylabel('Noisy')
                g.axes[0, i].axvline(self.barrier, color='black', linestyle='--')

            return g


    def plot_bootstrap(self, metric=None, noise_df=None):
        if noise_df is None:
            noise_df = self.noise_bootstraps
        if metric is None:
            metric = 'r2'

        ax = sns.boxplot(x="noise",y=metric,data=noise_df)
        ax.set_title(f'{metric} for different noise levels')
        ax.set_xlabel('Noise level')
        ax.set_ylabel(metric)

        return ax


    def _estimate_gaussian(self, noise_level):
        error = np.random.normal(0, noise_level, len(self.dataset))
        output = self.dataset + error
        return output
    
    def _estimate_uniform(self, noise_level):
        raise NotImplementedError('Uniform noise not implemented yet')
    
    
    def _estimate_asymmetric(self, noise_level):
        if self.asym_bound is None:
            self.asym_bound = np.median(self.dataset)
        below_boundary = self.dataset[self.dataset <= self.asym_bound]
        above_boundary = self.dataset[self.dataset > self.asym_bound]

        if self.asym_noise_low is None:
            self.asym_noise_low = noise_level * 2
        if self.asym_noise_up is None:
            self.asym_noise_up = noise_level / 2

        error_below = np.random.normal(0, self.asym_noise_low, len(below_boundary))
        error_above = np.random.normal(0, self.asym_noise_up, len(above_boundary))
        
        below_boundary = below_boundary + error_below
        above_boundary = above_boundary + error_above
        output = pd.concat([below_boundary, above_boundary], axis=0).sort_index()
        return output


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    # print(canvas())
    print('test')
