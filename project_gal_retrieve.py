#! /usr/bin/env python3

import glob
import os
import shutil
import time

import numpy as np
import pandas as pd

ti = time.time()
################################################################################
path_data = '/home/edgar/zorro/SDSSdata/data_proc'
path_original_data = '/home/edgar/zorro/SDSSdata/sas/dr16/sdss/spectro/redux/*\
/spectra/lite/*'
path_outputs = './'
path_out_spectra = './spectra'
O = np.load('../spec_99356.npy')
P = np.load('../pred_AE.npy')
################################################################################

print('Retrieving metadata!')

path_fnames = glob.glob(f'{path_data}/*-*[0-9].npy')
sdss_fnames = [fname.split('/')[-1].split('.')[0] for fname in path_fnames]
# score_names = ['chi2_outlier_scores', 'mad_outlier_scores',
                 # 'mse_outlier_scores']
score_names = glob.glob('../outlier_scores/*')

print('Loading outlier scores!')
df_data = {}

for score_name in score_names:

    scores = np.load(f'{score_name}', mmap_mode='r')
    spec_idxs = np.argpartition(scores,[10, -20])
    tmp_idxs = list(spec_idxs[:10]) + list(spec_idxs[-30:])
    df_data[f'{score_name}'] = scores[tmp_idxs]
    sdss = [sdss_fnames[idx] for idx in tmp_idxs]
    df_data[f'sdss_name_{score_name.split("/")[-1][:-11]}'] = sdss

    for idx in tmp_idxs:
        np.save(f'{path_out_spectra}/O_{sdss_fnames[idx]}.npy', O[idx, :])
        np.save(f'{path_out_spectra}/P_{sdss_fnames[idx]}.npy', P[idx, :])


df = pd.DataFrame(data= df_data)
df.to_csv('outliers.csv')
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
