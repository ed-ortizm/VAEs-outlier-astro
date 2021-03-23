#! /usr/bin/env python3

from glob import glob
import os
import sys

import numpy as np

from constants_AEs import data_proc


## Retrieve the name of the spectra
#sys.stdout = Logger()


print('Retrieving metadata!')
fnames = glob.glob(f'{all_data_proc}/*-*[0-9].npy')

print('Loading outlier scores!')
scores = np.load('scores.npy', mmap_mode='r')

ids_prospecs = np.argpartition(scores, -20)[-20:]

print('Outlier scores for the weirdest spectra')

for n, idx in enumerate(ids_prospecs):
    print(f'{n+1:02} --> {scores[idx]}')
    print(f'The file is: {fnames[idx]}')
