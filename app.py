import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from noiseestimator import NoiseEstimator

st.set_page_config(
   page_title="Noise Estimator App",
   page_icon=":level_slider:",
   layout="wide",
   initial_sidebar_state="expanded",
)

st.title('Noise Estimator')

st.write('This app estimates the noise level in a dataset')

st.sidebar.header('Input Parameters')

add_selectbox = st.sidebar.selectbox(
    'What kind of dataset would you like to use?',
    ('Synthetic', 'Examples', 'My own')
)

default_datasets = pd.read_csv('data/datasets.csv', index_col=0)

# st.write(default_datasets)

if add_selectbox == 'Synthetic':
    st.sidebar.write('Synthetic dataset')
    no_datapoints = st.sidebar.number_input('Number of datapoints', value=100)
    noise_levels = st.sidebar.slider(' Exp. noise level (relative to dataset range)', min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    predictor_noise_level = st.sidebar.slider('Predictor noise level', min_value=0.0, max_value=0.5, value=0.0, step=0.01)
    dataset = pd.DataFrame(np.linspace(0, 1, no_datapoints)).rename(columns={0: 'labels'})
    dataset_label = 'labels'
    classifier = st.sidebar.checkbox('Classifier')
    if classifier:
        class_barrier = st.sidebar.slider('Class barrier', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        # class_labels = st.sidebar.multiselect('Class labels', [0, 1], default=[0, 1])
    else:
        class_barrier = None
        class_labels = None
    st.write('## Synthetic Dataset')


elif add_selectbox == 'Examples':
    st.sidebar.write('Dataset settings')
    dataset_select = st.sidebar.selectbox('Example dataset', default_datasets.index)
    # st.write(default_datasets.loc[dataset_select, 'dataset_name'])
    dataset = pd.read_csv(f'data/processed/{default_datasets.loc[dataset_select, "file_name"]}')
    dataset_label = default_datasets.loc[dataset_select, 'label']
    noise_levels = st.sidebar.number_input('Experimental error', value=default_datasets.loc[dataset_select, 'noise'])
    predictor_noise_level = st.sidebar.number_input('Predictor noise level', value=default_datasets.loc[dataset_select, 'noise'])
    st.write(f'## {dataset_select} Dataset')
    classifier = False
    class_barrier = None
    if default_datasets.loc[dataset_select, 'classification'] == 1:
        classifier = True
        class_barrier = float(default_datasets.loc[dataset_select, 'class_barrier'])

elif add_selectbox == 'My own':
    st.sidebar.write('Upload your own dataset')
    dataset = st.sidebar.file_uploader('Upload dataset', type=['csv', 'txt'])

    if dataset is not None:
        dataset = pd.read_csv(dataset)
        dataset_label = st.sidebar.selectbox('Select dataset labels', dataset.columns)

        noise_levels = st.sidebar.slider('Noise levels', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        # noise_levels = noise_levels * (dataset[dataset_label].max() - dataset[dataset_label].min())
        predictor_noise_level = st.sidebar.slider('Predictor noise level', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        # predictor_noise_level = predictor_noise_level * (dataset[dataset_label].max() - dataset[dataset_label].min())
        classifier = st.sidebar.checkbox('Classifier')
        class_barrier = None
        if classifier:
            class_barrier = st.sidebar.slider('Class barrier', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    st.write(f'## Custom Dataset')

if dataset is not None:
    
    st.write(dataset)
    st.write(f'Dataset range:', round(dataset[dataset_label].min(),2), '-', round(dataset[dataset_label].max(), 2))

    st.write('## Noise Estimation')
    NE_max = NoiseEstimator(dataset[dataset_label], noise_levels, n_bootstrap=100, classifier = classifier, class_barrier=class_barrier)
    NE_real = NoiseEstimator(dataset[dataset_label], noise_levels, predictor_noise_level=predictor_noise_level, n_bootstrap=100)
    # st.write(NE.noise_estimates)
    # st.write(NE.noise_bootstraps)
    if not classifier:
        st.write('### Maximum bounds (no predictor noise)')
        st.write('Pearson R:', round(NE_max.noise_bootstraps.pearsonr.mean(),2), '±', round(NE_max.noise_bootstraps.pearsonr.std(),2))
        st.write('R2:', round(NE_max.noise_bootstraps.r2.mean(),2), '±', round(NE_max.noise_bootstraps.r2.std(),2))
        st.write('MAE:', round(NE_max.noise_bootstraps.mae.mean(),2), '±', round(NE_max.noise_bootstraps.mae.std(),2))
        st.write('MSE:', round(NE_max.noise_bootstraps.mse.mean(),2), '±', round(NE_max.noise_bootstraps.mse.std(),2))
        st.write('RMSE:', round(NE_max.noise_bootstraps.rmse.mean(),2), '±', round(NE_max.noise_bootstraps.rmse.std(),2))

        st.write('### Realistic bounds')
        st.write('Pearson R:', round(NE_real.noise_bootstraps.pearsonr.mean(),2), '±', round(NE_real.noise_bootstraps.pearsonr.std(),2))
        st.write('R2:', round(NE_real.noise_bootstraps.r2.mean(),2), '±', round(NE_real.noise_bootstraps.r2.std(),2))
        st.write('MAE:', round(NE_real.noise_bootstraps.mae.mean(),2), '±', round(NE_real.noise_bootstraps.mae.std(),2))
        st.write('MSE:', round(NE_real.noise_bootstraps.mse.mean(),2), '±', round(NE_real.noise_bootstraps.mse.std(),2))
        st.write('RMSE:', round(NE_real.noise_bootstraps.rmse.mean(),2), '±', round(NE_real.noise_bootstraps.rmse.std(),2))
    else:
        st.write('### Maximum bounds (no predictor noise)')
        st.write('Matthews Corr Coef:', round(NE_max.noise_bootstraps.matthews_corrcoef.mean(),2), '±', round(NE_max.noise_bootstraps.matthews_corrcoef.std(),2))
        st.write('ROC AUC:', round(NE_max.noise_bootstraps.roc_auc.mean(),2), '±', round(NE_max.noise_bootstraps.roc_auc.std(),2))
    fig, ax = plt.subplots()
    ax = NE_max.plot()
    st.pyplot(ax)
    fig, ax = plt.subplots()
    if not classifier:
        ax = NE_real.plot_bootstrap('pearsonr')
    else:
        ax = NE_max.plot_bootstrap('roc_auc')
    st.pyplot(fig)


#dataset = st.sidebar.file_uploader('Upload dataset', type=['csv', 'txt'])

# if dataset is not None:
#     dataset = pd.read_csv(dataset)
#     st.write(dataset)

#     noise_levels = st.sidebar.slider('Noise levels', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
#     predictor_noise_level = st.sidebar.slider('Predictor noise level', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
#     noise_type = st.sidebar.selectbox('Noise type', ['gaussian', 'uniform', 'asymmetric'])
#     n_bootstrap = st.sidebar.number_input('Number of bootstraps', value=1000)
#     classifier = st.sidebar.checkbox('Classifier')

#     if classifier:
#         class_barrier = st.sidebar.slider('Class barrier', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
#         class_labels = st.sidebar.multiselect('Class labels', [0, 1], default=[0, 1])
#     else:
#         class_barrier = None
#         class_labels = None

#     noise_estimator = NoiseEstimator(dataset, noise_levels, predictor_noise_level=predictor_noise_level, noise_type=noise_type, n_bootstrap=n_bootstrap, classifier=classifier, class_barrier=class_barrier, class_labels=class_labels)

#     st.write(noise_estimator.noise_estimates)
#     st.write(noise_estimator.noise_bootstraps)