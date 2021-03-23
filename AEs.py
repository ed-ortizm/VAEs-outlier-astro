#!/usr/bin/env python3

import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from AEs_lib import plot_2D, plt_spec_pca
from AEs_lib import AEpca, PcA, Outlier

from constants_AEs import sdss_dir
###############################################################################
ti = time.time()
###############################################################################
working_dir = f'/home/edgar/zorro/AEsII'

# Loading the data
print('Loading the data...')

input_data_fname =  f'{sdss_dir}/data/spec_99356.npy'
input_data_path = f'{working_dir}/{input_data_fname}'

if os.path.exists(input_data_path):
    input_data = np.load(input_data_path, mmap_mode='r')
else:
    print(f'There is no {input_data_fname}!')
###############################################################################
### Performing PCA
#print('Performing PCA...')
#
#pca = PcA(n_comps=2)
#pca.fit(input_data)
#latent_pca = pca.predict(input_data)
#np.save('latent_pca.npy', latent_pca)
#
## Creating the AE
#print('Training Auto Encoder...')
#
#AE = AEpca(in_dim=input_data.shape[1], epochs=30)
#AE.fit(input_data)
#
## Saving model
#AE.save()
#
## Latent space
#latent_AE = AE.encode(input_data)
#np.save('latent_AE.npy', latent_AE)
#
## Saving AE predictions
#pred = AE.predict(input_data)
#np.save('pred_99356.npy', pred)


pred = np.load(f'{working_dir}/data/pred_99356.npy')
# Outlier scores
outlier = Outlier(N=10)

# Area
outlier.area(input_data, pred)

# chi2
outlier.chi2(input_data, pred)

# mse
outlier.mse(input_data, pred)

# mad
outlier.mad(input_data, pred)

# lp 2, 1, 0.5, 0.3
outlier.lp(input_data, pred, p=2)
outlier.lp(input_data, pred, p=1)
outlier.lp(input_data, pred, p=0.5)
outlier.lp(input_data, pred, p=0.3)

# lpf 2, 1, 0.5, 0.3
outlier.lpf(input_data, pred, p=2)
outlier.lpf(input_data, pred, p=1)
outlier.lpf(input_data, pred, p=0.5)
outlier.lpf(input_data, pred, p=0.3)

###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
