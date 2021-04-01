#!/usr/bin/env python3.8
import os
import sys
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import input_handler, LoadAE, Outlier
###############################################################################
ti = time.time()
###############################################################################
n_spectra, normalization_type, local = input_handler(script_arguments=sys.argv)
###############################################################################
# Relevant directories
training_data_dir = f'{spectra_dir}/normalized_data'
predicted_data_dir = f'{spectra_dir}/AE_predicted'
models_dir = f'{working_dir}/models/AE'
###############################################################################
# Loading training data
training_set_name = f'spectra_{n_spectra}_{normalization_type}'
training_set_path = f'{training_data_dir}/{training_set_name}.npy'

if os.path.exists(training_set_path):

    print(f'Loading training set: {training_set_name}.npy\n')

    training_set =  np.load(f'{training_set_path}', mmap_mode='r')

else:
    print(f'There is no file: {training_set_name}.npy\n')
    sys.exit()
###############################################################################
# Loading AEs predicted data for outlier detection

reconstructed_set_name = f'{training_set_name}_reconstructed'
reconstructed_set_path = f'{predicted_data_dir}/{reconstructed_set_name}.npy'

if os.path.exists(reconstructed_set_path):

    print(f'Loading reconstructed data set: {reconstructed_set_name}\n')

    reconstructed_set =  np.load(f'{reconstructed_set_path}', mmap_mode='r')

else:

    print(f'There is no file: {reconstructed_set_name}\n')
    print(f'Generateing reconstructed set\n')


    # Loading models
    ae_name = f'DenseAE_mse_{n_spectra}'
    encoder_name = f'DenseEncoder_mse_{n_spectra}'
    decoder_name = f'DenseDecoder_mse_{n_spectra}'

    if local:

        ae_name = f'{ae_name}_local'
        encoder_name = f'{encoder_name}_local'
        decoder_name = f'{decoder_name}_local'


    ae_path = f'{models_dir}/{ae_name}'
    encoder_path = f'{models_dir}/{encoder_name}'
    decoder_path = f'{models_dir}/{decoder_name}'

    ae = LoadAE(ae_path, encoder_path, decoder_path)

    ############################################################################
    reconstructed_set = ae.predict(training_set[:, :-5])
    np.save(f'{reconstructed_set_path}', reconstructed_set)
###############################################################################
# Outlier detection
metrics = ['mse']
for metric in metrics:

    if metric == "lp":
        p = 0.1
        outlier = Outlier(model_path = model_path, o_scores_path=o_scores_path,
            metric=metric, p=p)

    else:
        outlier = Outlier(model_path = model_path, o_scores_path=o_scores_path,
            metric=metric)

    print(f'Computing/loading outlier scores: the labels ')

    if os.path.exists(f'{o_scores_path}/{metric}_o_score.npy'):
        o_scores = np.load(f'{o_scores_path}/{metric}_o_score.npy')
    else:
        o_scores = outlier.score(O=training_data)
        np.save(f'{o_scores_path}/{metric}_o_score.npy', o_scores)

    t1 = time.time()
    # Check if they are normalized
    print(f't1: {t1-ti:.2f} s')
    ################################################################################
    ## Selecting top outliers
    print(f'Computing top reconstructions for {metric} metric')
    most_normal_ids, most_oulying_ids = outlier.top_reconstructions(O=training_data, n_top_spectra=n_top_spectra)

###############################################################################
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
