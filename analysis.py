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


BASE_DIRECTORY='./data_base'
TEST_BASE_DIRECTORY='./test_base'
ANALYSIS_FILE='./an_file.yaml'

POSTERIOR_NAME = 'BSZ-BqToK-wSR-wNFF'

#OPTIONS---------------
flag_sample = 0
flag_predict = 1
flag_gof = 0
#-----------------------

if flag_sample:
    eos.tasks.sample_nested(ANALYSIS_FILE, POSTERIOR_NAME, base_directory=BASE_DIRECTORY, nlive=100, dlogz=2.0, seed=42)

#'BToK-F0', 'BToK-FT0', 'BToK-Re(H0(F))', 'BToK-Im(H0(F))', 'BToK-Re(Delta-H0)/Re(H0(F))', 'BToK-Im(Delta-H0)/Im(H0(F))', 'BToK-Re(H0)', 'BToK-Im(H0)'
# 'BToK-Re(H0(F))/F0', 'BToK-Im(H0(F))/F0'
# 'BToK-Re(H0)', 'BToK-Im(H0)', 'BToK-Re(H0(F))', 'BToK-Im(H0(F))'

if flag_predict:
    predicts=['BToK-Re(H0(F))/F0', 'BToK-Im(H0(F))/F0'] 
    for i in predicts:
        eos.tasks.predict_observables(ANALYSIS_FILE, POSTERIOR_NAME, i,  base_directory=BASE_DIRECTORY)    

if flag_gof:
    bfp, gof  = eos.tasks.find_mode(ANALYSIS_FILE, POSTERIOR_NAME, TEST_BASE_DIRECTORY, importance_samples=True, label='EOS', optimizations=50)
    display(bfp)
    display(gof)