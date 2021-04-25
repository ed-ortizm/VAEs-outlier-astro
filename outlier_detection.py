#!/usr/bin/env python3.8
from argparse import ArgumentParser
import glob
import os
import sys
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from library_outlier import Outlier
from lib_VAE_outlier import load_data

###############################################################################
ti = time.time()
################################################################################
parser = ArgumentParser()
############################################################################
parser.add_argument('--server', '-s', type=str)
parser.add_argument('--number_spectra','-n_spec', type=int)
parser.add_argument('--normalization_type', '-n_type', type=str)
############################################################################
parser.add_argument('--model', type=str)
parser.add_argument('--encoder_layers', type=str)
parser.add_argument('--latent_dimensions', '-lat_dims', type=int)
parser.add_argument('--decoder_layers', type=str)
parser.add_argument('--loss', type=str)
############################################################################
parser.add_argument('--number_snr', '-n_snr', type=int)
############################################################################
parser.add_argument('--metrics', type=str, nargs='+')
parser.add_argument('--top_spectra', '-top', type=int)
############################################################################
script_arguments = parser.parse_args()
################################################################################
local = script_arguments.server == 'local'
number_spectra = script_arguments.number_spectra
normalization_type = script_arguments.normalization_type
############################################################################
model = script_arguments.model
layers_encoder = script_arguments.encoder_layers
number_latent_dimensions = script_arguments.latent_dimensions
layers_decoder = script_arguments.decoder_layers
loss = script_arguments.loss
layers_str = f'{layers_encoder}_{number_latent_dimensions}_{layers_decoder}'
############################################################################
number_snr = script_arguments.number_snr
############################################################################
metrics = script_arguments.metrics
number_top_spectra = script_arguments.top_spectra
################################################################################
# Relevant directories
data_dir = f'{spectra_dir}/procesesed_spectra'
generated_data_dir = f'{spectra_dir}/{model}_outlier/{layers_str}/{number_snr}'
################################################################################
# Loading data
data_set_name = f'spectra_{number_spectra}_{normalization_type}'
############################################################################
train_set_name = f'{data_set_name}_nSnr_{number_snr}_train'
train_set_path = f'{data_dir}/{train_set_name}.npy'

train_set = load_data(train_set_name, train_set_path)
############################################################################
test_set_name = f'{data_set_name}_nSnr_{number_snr}_test'
test_set_path = f'{data_dir}/{test_set_name}.npy'

test_set = load_data(test_set_name, test_set_path)
################################################################################
# Reconstructed data for outlier detection
tail_model_name = (f'{layers_str}_loss_{loss}_nTrain_{number_snr}_'
    f'nType_{normalization_type}')

tail_reconstructed = f'reconstructed_{model}_{tail_model_name}'
############################################################################
reconstructed_train_set_name = f'{train_set_name}_{tail_reconstructed}'
reconstructed_test_set_name = f'{test_set_name}_{tail_reconstructed}'

if local:
    reconstructed_train_set_name = f'{reconstructed_train_set_name}_local'
    reconstructed_test_set_name = f'{reconstructed_test_set_name}_local'
############################################################################
reconstructed_train_set_path = (
    f'{generated_data_dir}/{reconstructed_train_set_name}.npy')

train_set = load_data(reconstructed_train_set_name,
    reconstructed_train_set_path)
############################################################################
reconstructed_test_set_path = (
    f'{generated_data_dir}/{reconstructed_test_set_name}.npy')

test_set = = load_data(reconstructed_test_set_name,
    reconstructed_test_set_path)
################################################################################
# Outlier detection
tail_outlier_name = f'{model}_{layers_str}_loss_{loss}_{number_spectra}'

if local:
    tail_outlier_name = f'{tail_outlier_name}_local'

for metric in metrics:
    outlier = Outlier(metric=metric)
    ############################################################################
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    outlier_scores = outlier.score(O=train_set[:, :-5],
        R=reconstructed_set, percentages=percentages)

    for idx, o_scores in enumerate(outlier_scores):

        percentage = f'percentage_{int(percentages[idx] * 100)}'
    ############################################################################
        o_scores_name = (f'{metric}_o_score_{percentage}_'
            f'{tail_outlier_name}')

        np.save(
        f'{generated_data_dir}/{o_scores_name}.npy', o_scores)
    ############################################################################
        normal_ids, outlier_ids = outlier.top_reconstructions(
            scores=o_scores, n_top_spectra=number_top_spectra)

        print('Saving top outliers IDs')
        ########################################################################
        normal_name = (f'{metric}_normal_ids_{percentage}_'
            f'nTop_{number_top_spectra}_{tail_outlier_name}')
        np.save(f'{generated_data_dir}/{normal_name}.npy', normal_ids)
        ########################################################################
        outlier_name = (f'{metric}_outlier_ids_{percentage}_'
            f'nTop_{number_top_spectra}_{tail_outlier_name}')
        np.save(f'{generated_data_dir}/{outlier_name}.npy', outlier_ids)
    ############################################################################
        spec_top_outliers_name = (f'{metric}_outlier_spectra_{percentage}_'
            f'nTop_{number_top_spectra}_{tail_outlier_name}')

        spec_top_outliers = train_set[outlier_ids]

        spec_top_outliers = np.insert(spec_top_outliers, 0, outlier_ids)

        np.save(f'{generated_data_dir}/{spec_top_outliers_name}.npy',
            spec_top_outliers)
        ########################################################################
        R_top_outliers_name = (f'{metric}_outlier_reconstructed_spectra_'
            f'{percentage}_nTop_{number_top_spectra}_{tail_outlier_name}')

        R_top_outliers = reconstructed_set[outlier_ids]

        np.save(f'{generated_data_dir}/{R_top_outliers_name}.npy',
            R_top_outliers)
        ########################################################################
        spec_top_normal_name = (f'{metric}_normal_spectra_{percentage}_'
            f'nTop_{number_top_spectra}_{tail_outlier_name}')

        spec_top_normal = train_set[normal_ids]

        spec_top_normal = np.insert(spec_top_normal, 0, normal_ids)

        np.save(f'{generated_data_dir}/{spec_top_normal_name}.npy',
            spec_top_normal)
        ########################################################################
        R_top_normal_name = (f'{metric}_normal_reconstructed_spectra_'
            f'{percentage}_nTop_{number_top_spectra}_{tail_outlier_name}')

        R_top_normal = reconstructed_set[normal_ids]

        np.save(f'{generated_data_dir}/{R_top_normal_name}.npy',
            R_top_normal)
        ############################################################################
    ############################################################################
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
