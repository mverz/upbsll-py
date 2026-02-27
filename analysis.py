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


BASE_DIRECTORY='./data_base'
#BASE_DIRECTORY='./test_base'
#ANALYSIS_FILE='./an_file.yaml'
ANALYSIS_FILE='./an_file_hardcoded.yaml'

POSTERIOR_NAME = 'BSZ-BqToK-wSR-wNFF-wCov'

#OPTIONS---------------
flag_sample = 1
flag_predict = 0
flag_gof = 0
#-----------------------

time0 = time.perf_counter()

if flag_sample:
    eos.tasks.sample_nested(ANALYSIS_FILE, POSTERIOR_NAME, base_directory=BASE_DIRECTORY, nlive=200, dlogz=1.0, seed=42)

time1 = time.perf_counter() - time0

# 'BToK-F0', 'BToK-FT0', 'BToK-Re(H0(F))', 'BToK-Im(H0(F))', 'BToK-Re(Delta-H0)/Re(H0(F))', 'BToK-Im(Delta-H0)/Im(H0(F))', 'BToK-Re(H0)', 'BToK-Im(H0)'
# 'BToK-Re(H0(F))-over-F0', 'BToK-Im(H0(F))-over-F0'
# 'BToK-Re(H0)', 'BToK-Im(H0)', 'BToK-Re(H0(F))', 'BToK-Im(H0(F))', 'BToK-Re(Delta-H0)', 'BToK-Im(Delta-H0)'

if flag_predict:
    predicts=['BToK-Re(H0(F))-over-F0', 'BToK-Im(H0(F))-over-F0'] 
    for i in predicts:
        eos.tasks.predict_observables(ANALYSIS_FILE, POSTERIOR_NAME, i,  base_directory=BASE_DIRECTORY)    

if flag_gof:
    bfp, gof  = eos.tasks.find_mode(ANALYSIS_FILE, POSTERIOR_NAME, BASE_DIRECTORY, importance_samples=True, label='EOS', optimizations=50)
    display(bfp)
    display(gof)

total_time = time.perf_counter() - time0
print(f"Sampling time: {time1 / 3600.0} hours")
print(f"Total execution time: {total_time / 3600.0} hours")