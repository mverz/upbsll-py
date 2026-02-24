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
af=eos.AnalysisFile(ANALYSIS_FILE)


flag_sample = 1
if flag_sample:
    eos.tasks.sample_nested(af,'BSZ-BqToK-wSR', base_directory=BASE_DIRECTORY, nlive=250, dlogz=1.0, seed=42)

#'BToK-F0', 'BToK-FT0', 'BToK-Re(H0(F))', 'BToK-Im(H0(F))', 'BToK-Re(Delta-H0)/Re(H0(F))', 'BToK-Im(Delta-H0)/Im(H0(F))', 'BToK-Re(H0)', 'BToK-Im(H0)'
flag_predict = 1
if flag_predict:
    predicts=['BToK-Re(H0(F))', 'BToK-Im(H0(F))']
    for i in predicts:
        eos.tasks.predict_observables(af, 'BSZ-BqToK-wSR', i,  base_directory=BASE_DIRECTORY)    
