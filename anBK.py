import eos
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from IPython.display import display, Latex
#import re
import yaml
#from tabulate import tabulate
from wquantiles import quantile
from collections import OrderedDict
import time


BASE_DIRECTORY='./test_base'

ANALYSIS_FILE='./afBK.yaml'

#POSTERIOR_NAME = 'BSZ-BqToK-wSR-wNFF-wCov'
#POSTERIOR_NAME = 'GRV2026-BqToK-HPQCD6'
POSTERIOR_NAME = 'GRV-BK-HPQCD'


#OPTIONS---------------
flag_sample = 1
flag_predict = 0
flag_gof = 1
#-----------------------

print(f"BASE: {BASE_DIRECTORY} \t-\t ANALYSIS: {ANALYSIS_FILE} \t-\t POSTERIOR: {POSTERIOR_NAME}")

time0 = time.perf_counter()

if flag_sample:
    eos.tasks.sample_nested(ANALYSIS_FILE, POSTERIOR_NAME, base_directory=BASE_DIRECTORY, nlive=50, dlogz=5.0, seed=42)

time_sample = time.perf_counter() - time0

# 'BToK-Re(H0)', 'BToK-Im(H0)', 'BToK-Re(H0(F))', 'BToK-Im(H0(F))', 'BToK-Re(Delta-H0)', 'BToK-Im(Delta-H0)'
# 'BToK-alpha-coeffs'
# 'BToK-strong-bound','BToK-Re(H0(-1,F))-over-F0','BToK-Im(H0(-1,F))-over-F0','BToK-Re(H0(-3,F))-over-F0','BToK-Im(H0(-3,F))-over-F0','BToK-Re(H0(-5,F))-over-F0','BToK-Im(H0(-5,F))-over-F0','BToK-Re(H0(-7,F))-over-F0','BToK-Im(H0(-7,F))-over-F0','BToKll-Normalized-BR-binned'
# 'BToK-AbsH0-over-F0'
# 'BToK-strong-bound','BToKll-Normalized-BR-binned','BToKll-Normalized-BR-binned-high-q2','C9NP',
# 'BToK-Re(H0(-1,F))-over-F0', 'BToK-Im(H0(-1,F))-over-F0',
# 'BToK-Re(H0(-3,F))-over-F0', 'BToK-Im(H0(-3,F))-over-F0',
# 'BToK-Re(H0(-5,F))-over-F0', 'BToK-Im(H0(-5,F))-over-F0',
# 'BToK-Re(H0(-7,F))-over-F0', 'BToK-Im(H0(-7,F))-over-F0',


if flag_predict:
    predicts = [
        'BToK-strong-bound','BToKll-Normalized-BR-binned','BToKll-Normalized-BR-binned-high-q2',
        'BToK-Re(H0(-1,F))-over-F0', 'BToK-Im(H0(-1,F))-over-F0',
        'BToK-Re(H0(-3,F))-over-F0', 'BToK-Im(H0(-3,F))-over-F0',
        'BToK-Re(H0(-5,F))-over-F0', 'BToK-Im(H0(-5,F))-over-F0',
        'BToK-Re(H0(-7,F))-over-F0', 'BToK-Im(H0(-7,F))-over-F0',
    ]
    for i in predicts:
        eos.tasks.predict_observables(ANALYSIS_FILE, POSTERIOR_NAME, i,  base_directory=BASE_DIRECTORY)

time_predict = time.perf_counter() - time0

if flag_gof:
    bfp, gof  = eos.tasks.find_mode(ANALYSIS_FILE, POSTERIOR_NAME, BASE_DIRECTORY, importance_samples=True, label='EOS', optimizations=50)
    #display(bfp)
    display(gof)

print(f"BASE: {BASE_DIRECTORY} - ANALYSIS: {ANALYSIS_FILE} - POSTERIOR: {POSTERIOR_NAME}")

total_time = time.perf_counter() - time0
print(f"Sampling time: {time_sample / 60.0:.2f} minutes")
print(f"Prediction time: {(time_predict - time_sample) / 60.0:.2f} minutes")
print(f"Total execution time: {total_time / 60.0:.2f} minutes")