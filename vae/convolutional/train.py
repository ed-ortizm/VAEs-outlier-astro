#!/usr/bin/env python3.8
from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import time
################################################################################
import numpy as np
################################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('vae.ini')
############################################################
# to import from src since I'm in a difrent leaf of the tree structure
work_directory = parser.get('directories', 'work')
sys.path.insert(0, f'{work_directory}')
################################################################################
from src.vae import VAE
################################################################################
ti = time.time()
################################################################################
# load data
data_directory = parser.get('directories', 'train')
data_name = parser.get('files', 'train')
data = np.load(f'{data_directory}/{data_name}')
input_dimensions = data.shape[1]
############################################################################
# network architecture
encoder_units = parser.get('architecture', 'encoder')
encoder_units = [int(units) for units in encoder_units.split('_')]

latent_dimensions = parser.getint('architecture', 'latent_dimensions')

decoder_units = parser.get('architecture', 'decoder')
decoder_units = [int(units) for units in decoder_units.split('_')]
############################################################################
# network hyperparameters
learning_rate = parser.getfloat('hyper-parameters', 'learning_rate')
batch_size = parser.getint('hyper-parameters', 'batch_size')
epochs = parser.getint('hyper-parameters', 'epochs')

reconstruction_weight = parser.getint(
                                        'hyper-parameters',
                                        'reconstruction_weight'
                                        )
################################################################################
vae = VAE(
    input_dimensions=input_dimensions,
    encoder_units=encoder_units,
    latent_dimensions=latent_dimensions,
    decoder_units=decoder_units,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=learning_rate,
    reconstruction_loss_weight=reconstruction_weight
    )

vae.summary()
################################################################################
# Training the model
history = vae.train(data)
# save model
models_directory = parser.get('directories', 'models')

if not os.path.exists(models_directory):
    os.makedirs(models_directory)
###############################################################################
# save the model once it is trained

# Defining directorie to save the model once it is trained
# vae_name = 'DenseVAE'
# encoder_name = 'DenseEncoder'
# decoder_name = 'DenseDecoder'
#
# vae.save_vae(f'{models_dir}/{vae_name}')
# vae.save_encoder(f'{models_dir}/{encoder_name}')
# vae.save_decoder(f'{models_dir}/{decoder_name}')
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
