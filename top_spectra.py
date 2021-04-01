#! /usr/bin/env python3
import glob
import os
import time

import numpy as np

from constants_lime import sdss_dir, sdss_data_proc, AE_data_dir
from lib_explanations import Outlier
################################################################################
ti = time.time()
################################################################################
# Relevant paths
working_dir = '/home/edgar/zorro/AEsII/xAI/lime/'
AE_in_data_fname ='spec_99356.npy'
AE_pred_fname = 'pred_99356.npy'
model_path = '/home/edgar/zorro/AEsII/trained_models/AutoEncoder'

o_scores_path = "/home/edgar/zorro/AEsII/outlier_scores"
spec_top_path = "/home/edgar/zorro/AEsII/xAI/lime/data/top_spectra"
################################################################################
# Outlier scores to have a regression model
training_data = np.load(f'{sdss_dir}/data/{AE_in_data_fname}')
predicted_data = np.load(f'{AE_data_dir}/{AE_pred_fname}')
# Training data files
if os.path.exists(f'{working_dir}/data/sdss_spec_paths.txt'):
    with open(f'{working_dir}/data/sdss_spec_paths.txt', 'r') as file:
        sdss_spec_paths = file.readlines()
else:
    sdss_spec_paths = glob.glob(f'{sdss_data_proc}/*-*[0-9].npy')
    with open('./testing/sdss_spec_paths.txt', 'w') as file:
        file.writelines(f"{line}\n" for line in sdss_spec_paths)

metrics = ["lp", "mse", "chi2", "mad"]
n_top_spectra = 500

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
    print(f"Getting top spectra")
    print(f'Top outlier spectra...')
    top_outliers = []

    o_sdss_names = []
    o_sdss_paths = []

    for idx, spec_idx in enumerate(most_oulying_ids):

        sdss_name, sdss_name_path = outlier.metadata(spec_idx=spec_idx,
        training_data_files=sdss_spec_paths)

        o_sdss_names.append(sdss_name)
        o_sdss_paths.append(sdss_name_path)

        top_outliers.append(training_data[spec_idx, :])
        np.save(
            f"{spec_top_path}/{sdss_name}_model_input.npy",
            training_data[spec_idx, :])

        np.save(
            f"{spec_top_path}/{sdss_name}_model_pred.npy",
            predicted_data[spec_idx, :])

    t2 = time.time()
    print(f't2: {t2-t1:.2f} s')
################################################################################
    print(f'Top inlier spectra...')
    top_inliers = []

    o_sdss_names = []
    o_sdss_paths = []

    for idx, spec_idx in enumerate(most_normal_ids):

        sdss_name, sdss_name_path = outlier.metadata(spec_idx=spec_idx,
        training_data_files=sdss_spec_paths)

        o_sdss_names.append(sdss_name)
        o_sdss_paths.append(sdss_name_path)

        top_inliers.append(training_data[spec_idx, :])
        np.save(
            f"{spec_top_path}/{sdss_name}_model_input.npy",
            training_data[spec_idx, :])

        np.save(
            f"{spec_top_path}/{sdss_name}_model_pred.npy",
            predicted_data[spec_idx, :])

    t3 = time.time()
    print(f't3: {t3-t2:.2f} s')
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
