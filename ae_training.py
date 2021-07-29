#!/usr/bin/env python3
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import time

import numpy as np

from lib_VAE_outlier import AEDense, Base36
from lib_VAE_outlier import load_data
from lib_VAE_outlier import plot_history
###############################################################################
ti = time.time()
###############################################################################
parser = ArgumentParser()
parser.add_argument('--server', '-s', type=str)
script_arguments = parser.parse_args()
local = script_arguments.server == 'local'
###############################################################################
# read them from the configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('ae.ini')
    ############################################################################
# network architecture
encoder_str = parser.get('network architecture', 'encoder_layers')
layers_encoder = [int(units) for units in encoder_str.split('_')]

latent_dimensions = parser.getint('network architecture', 'latent_dimensions')

decoder_str = parser.get('network architecture', 'decoder_layers')
layers_decoder = [int(units) for units in decoder_str.split('_')]

layers_str = f'{encoder_str}_{latent_dimensions}_{decoder_str}'
    ############################################################################
# network hyperparameters
loss = parser.get('network hyper-parameters', 'loss')
learning_rate = parser.getfloat('network hyper-parameters', 'learning_rate')
batch_size = parser.getint('network hyper-parameters', 'batch_size')
epochs = parser.getint('network hyper-parameters', 'epochs')
    ############################################################################
# data parameters
normalization_type = parser.get('data hyper-parameters', 'normalization_type')
number_spectra = parser.getint('data hyper-parameters', 'number_spectra')
number_snr = parser.getint('data hyper-parameters', 'number_snr')
    ############################################################################
# Relevant directories
data_dir = parser.get('paths', 'data_dir')
# Defining directorie to save the model once it is trained
models_dir = parser.get('paths', 'models_dir')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    ############################################################################
    ############################################################################
# # Loading train data
# data_set_name = f'spectra_{number_spectra}_{normalization_type}'
# data_set_path = f'{data_dir}/{data_set_name}.npy'
# ########################################################################
# data = load_data(data_set_name, data_set_path)
# data = data[: number_snr]
# # select the normal galaxies
# base36 = Base36()
# normal36, empty36 = base36.decode('SF'), base36.decode('')
#
# normal36_mask = data[:, -3] == normal36
# normal36_number = np.count_nonzero(normal36_mask)
#
# empty36_mask = data[:, -3] == empty36
# empty36_number = np.count_nonzero(empty36_mask)
#
# print(f'empty: {empty36_number}, normal: {normal36_number}')
# ############################################################################
# # Save test set
# test_set = data[~normal36_mask] #np.vstack((data[~normal36_mask], data[~empty36_mask]))
#
# test_set_name = f'{data_set_name}_nSnr_{number_snr}_noSF_test'
# test_set_path = f'{data_dir}/{test_set_name}.npy'
#
# np.save(f'{test_set_path}', test_set)
# ############################################################################
# train_set = data[normal36_mask] #np.vstack((data[normal36_mask], data[empty36_mask]))
#
# # Save train set
# train_set_name = f'{data_set_name}_nSnr_{number_snr}_SF_train'
# train_set_path = f'{data_dir}/{train_set_name}.npy'
#
# np.save(f'{train_set_path}', train_set)
# np.random.shuffle(train_set)
# ################################################################################
# # Parameters for the AEDense
# number_input_dimensions = train_set[:, :-8].shape[1]
number_input_dimensions = 3000
# ############################################################################
ae = AEDense(number_input_dimensions, layers_encoder, latent_dimensions,
    layers_decoder, batch_size, epochs, learning_rate, loss)

# ae.summary()
# ###############################################################################
# # train the model
# history = ae.fit(spectra=train_set[:, :-8])
# ################################################################################
# tail_model_name = (f'{layers_str}_loss_{loss}_nTrain_{number_snr}_'
#     f'nType_{normalization_type}')
#
# ae_name = f'DenseAE_{tail_model_name}'
# encoder_name = f'DenseEncoder_{tail_model_name}'
# decoder_name = f'DenseDecoder_{tail_model_name}'
#
# if local:
#
#     ae_name = f'{ae_name}_local'
#     encoder_name = f'{encoder_name}_local'
#     decoder_name = f'{decoder_name}_local'
#
# ae.save_ae(f'{models_dir}/{ae_name}')
# ae.save_encoder(f'{models_dir}/{encoder_name}')
# ae.save_decoder(f'{models_dir}/{decoder_name}')
#
# plot_history(history, f'{models_dir}/{ae_name}')
# ################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
