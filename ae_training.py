#!/usr/bin/env python3.8

import os
import sys
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import AEDense
from lib_VAE_outlier import input_handler
###############################################################################
ti = time.time()
###############################################################################
n_spectra, normalization_type, local = input_handler(script_arguments=sys.argv)
layers_list = [int(n_units) for n_units in sys.argv[4].split('_')]
n_layers_list = len(layers_list)
###############################################################################
# Relevant directories
training_data_dir = f'{spectra_dir}/normalized_data'
###############################################################################
# Loading training data
fname = f'spectra_{n_spectra}_{normalization_type}.npy'
fpath = f'{training_data_dir}/{fname}'

if os.path.exists(fpath):

    print(f'Loading training set: {fname}')

    training_set =  np.load(f'{fpath}')
    np.random.shuffle(training_set)

else:
    print(f'There is no file: {fname}')
    sys.exit()
###############################################################################
# Parameters for the AEDense
n_galaxies = training_set.shape[0]
n_input_dimensions = training_set[:, :-5].shape[1]
n_middle = int((n_layers_list-1)/2)
n_latent_dimensions = layers_list[n_middle]
print(f'n_latent: {n_latent_dimensions}')
###########################################
# encoder
n_layers_encoder = layers_list[ : n_middle]
print(f'n_encoder: {n_layers_encoder}')
# decoder
n_layers_decoder = layers_list[n_middle+1 : ]
print(f'n_decoder: {n_layers_decoder}')

# Other parameters
# 1% to take advantage of stochastic part of stochastic gradient descent
batch_size = int(sys.argv[5])
print(f'Batch size is: {batch_size}')

if local:
    epochs = 5
else:
    epochs = 15

learning_rate = float(sys.argv[6]) # default: 0.001
loss = 'mse'

ae = AEDense(n_input_dimensions, n_layers_encoder, n_latent_dimensions,
    n_layers_decoder, batch_size, epochs, learning_rate, loss)

ae.summary()
###############################################################################
# Training the model

ae.fit(spectra=training_set[:, :-5])
# PEnding to track history
###############################################################################
# Defining directorie to save the model once it is trained
models_dir = f'{working_dir}/models/AE'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# Models names
# layers for name
layers_encoder_str = '_'.join(str(unit) for unit in n_layers_encoder)
layers_decoder_str = '_'.join(str(unit) for unit in n_layers_decoder)
layers_str = f'{layers_encoder_str}_{n_latent_dimensions}_{layers_decoder_str}'

models_dir = f'{models_dir}/{layers_str}'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

ae_name = f'DenseAE_{loss}_{n_galaxies}_{layers_str}'
encoder_name = f'DenseEncoder_{loss}_{n_galaxies}_{layers_str}'
decoder_name = f'DenseDecoder_{loss}_{n_galaxies}_{layers_str}'

if local:

    print('Saving model trained in local machine')
    ae.save_ae(f'{models_dir}/{ae_name}_local')
    ae.save_encoder(f'{models_dir}/{encoder_name}_local')
    ae.save_decoder(f'{models_dir}/{decoder_name}_local')

else:
    ae.save_ae(f'{models_dir}/{ae_name}')
    ae.save_encoder(f'{models_dir}/{encoder_name}')
    ae.save_decoder(f'{models_dir}/{decoder_name}')
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
