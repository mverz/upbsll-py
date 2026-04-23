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
from scipy import integrate
import mpmath as mp
import sympy as sp

pp = eos.Parameters.Defaults()
oo = eos.Options()
mm = eos.Model.make('SM', pp, oo)

mu = 4.18

# MY SET OF PARAMS
mb = mu
#mc = 1.44423
#alpha_s = mm.alpha_s(mu)
# pp.set('mass::c', mc)
pp.set('mass::b(MSbar)', mb)

# ms = pp['mass::s(2GeV)'].evaluate()

# MB = pp['mass::B_d'].evaluate()
# MK = pp['mass::K_d'].evaluate()
# MD = pp['mass::D^0'].evaluate()

# MBs_star = pp['mass::B_s^*'].evaluate()
# --------------------------

# NICO'S SET OF PARAMS
alpha_s = 0.217
ms = 0.1
mc = 0.29 * mb
pp.set('mass::c', mc)
pp.set('mass::s(2GeV)', ms)

MD = 1.8672
MK = 0.497614
MB = 5.27958
MBs_star = 5.4154
MJpsi = 3.070
Mpsi2S = 3.686
pp.set('mass::D^0', MD)
pp.set('mass::K_d', MK)
pp.set('mass::B_d', MB)
pp.set('mass::B_s^*', MBs_star)
pp.set('mass::J/psi', MJpsi)
pp.set('mass::psi(2S)', Mpsi2S)

pp.set('b->s::c1', -0.29063621)
pp.set('b->s::c2', 1.01029623)
# ---------------------------

def kallen(x,y,z):
    return  x*x + y*y + z*z - 2 * (x*y + x*z + y*z)

def lambda_k(s):
    return abs(kallen(mb*mb, ms*ms, s))

def Im_Pi_11_NLO_V(m1,m2,s):

    expression = ( 
        -1/24*(alpha_s*
        (np.sqrt(lambda_k(s))*(29*m1**6 + 64*m1**5*m2 - 
        29*m1**4*m2**2 - 128*m1**3*m2**3 - 29*m1**2*m2**4 + 64*m1*m2**5 + 29*m2**6 + 
        4*m1**4*s - 90*m1**3*m2*s - 164*m1**2*m2**2*s - 90*m1*m2**3*s + 4*m2**4*s - 
        39*m1**2*s**2 + 90*m1*m2*s**2 - 39*m2**2*s**2 + 6*s**3 - m1**6*np.log(4) - 
        m2**6*np.log(4) - m1**5*m2*np.log(16) - m1*m2**5*np.log(16) + 
        2*m1*m2*s**2*np.log(4096) + 2*s**3*np.log(4096) + m1**6*np.log(16384) + 
        m2**6*np.log(16384) - 2*m1**2*s**2*np.log(262144) - 2*m2**2*s**2*np.log(262144) - 
        2*m1**3*m2**3*np.log(16777216) - 2*m1**3*m2*s*np.log(16777216) - 
        2*m1*m2**3*s*np.log(16777216) + m1**5*m2*np.log(268435456) + 
        m1*m2**5*np.log(268435456) - 2*m1**2*m2**2*s*np.log(281474976710656) + 
        m1**4*m2**2*np.log(1125899906842624) + m1**2*m2**4*np.log(1125899906842624) - 
        m1**4*m2**2*np.log(4611686018427387904) - 
        m1**2*m2**4*np.log(4611686018427387904) + 
        6*(-3*m1**6 - 8*m1**5*m2 - 2*m1**3*m2*(-2*m2**2 + s) + 
        2*m1*m2*(m2**2 - s)*(2*m2**2 + s) - m1**4*(5*m2**2 + 2*s) + 
        (m2**2 - s)**2*(m2**2 + 4*s) + m1**2*(7*m2**4 - 4*m2**2*s + s**2))*np.log(abs(m1)) + 
        6*(m1**6 + 4*m1**5*m2 - 2*m1**3*m2*(-2*m2**2 + s) + m1**4*(7*m2**2 + 2*s) + 
        (m2**2 - s)*(-3*m2**4 - 5*m2**2*s - 4*s**2) + 
        2*m1*m2*(-4*m2**4 - m2**2*s - s**2) - m1**2*(5*m2**4 + 4*m2**2*s + 7*s**2))*
        np.log(m2) + 18*m1**6*np.log(mu**2) + 36*m1**5*m2*np.log(mu**2) - 
        18*m1**4*m2**2*np.log(mu**2) - 72*m1**3*m2**3*np.log(mu**2) - 
        18*m1**2*m2**4*np.log(mu**2) + 36*m1*m2**5*np.log(mu**2) + 18*m2**6*np.log(mu**2) - 
        36*m1**3*m2*s*np.log(mu**2) - 72*m1**2*m2**2*s*np.log(mu**2) - 
        36*m1*m2**3*s*np.log(mu**2) - 18*m1**2*s**2*np.log(mu**2) + 
        36*m1*m2*s**2*np.log(mu**2) - 18*m2**2*s**2*np.log(mu**2) + 
        12*m1**6*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**
        (-1)) + 24*m1**5*m2*np.log((-m1**2 - m2**2 + s + 
        np.sqrt(lambda_k(s)))**(-1)) - 
        12*m1**4*m2**2*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 48*m1**3*m2**3*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 
        12*m1**2*m2**4*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 24*m1*m2**5*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 
        12*m2**6*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**
        (-1)) - 48*m1**3*m2*s*np.log((-m1**2 - m2**2 + s + 
        np.sqrt(lambda_k(s)))**(-1)) - 
        96*m1**2*m2**2*s*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 48*m1*m2**3*s*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 
        36*m1**2*s**2*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 24*m1*m2*s**2*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 
        36*m2**2*s**2*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 24*s**3*np.log((-m1**2 - m2**2 + s + 
        np.sqrt(lambda_k(s)))**(-1)) + 
        4*m1**6*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1**5*m2*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 4*m1**4*m2**2*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**3*m2**3*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 4*m1**2*m2**4*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1*m2**5*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 4*m2**6*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**3*m2*s*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 32*m1**2*m2**2*s*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1*m2**3*s*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m1**2*s**2*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1*m2*s**2*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m2**2*s**2*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*s**3*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        4*m1**6*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1**5*m2*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 4*m1**4*m2**2*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**3*m2**3*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 4*m1**2*m2**4*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1*m2**5*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 4*m2**6*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**3*m2*s*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 32*m1**2*m2**2*s*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1*m2**3*s*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m1**2*s**2*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1*m2*s**2*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m2**2*s**2*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*s**3*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**6*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*m1**5*m2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        16*m1**4*m2**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        64*m1**3*m2**3*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        16*m1**2*m2**4*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*m1*m2**5*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        16*m2**6*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        64*m1**3*m2*s*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        128*m1**2*m2**2*s*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        64*m1*m2**3*s*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        48*m1**2*s**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*m1*m2*s**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        48*m2**2*s**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*s**3*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)) + 
        (m1**2 + 2*m1*m2 + m2**2 - s)*
        (4*m1**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 12*m1**4*m2**2*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 12*m1**2*m2**4*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m2**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        4*m1**4*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**3*m2*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1*m2**3*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        4*m2**4*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        16*m1**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m2**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        4*m1**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 12*m1**4*m2**2*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 12*m1**2*m2**4*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m2**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        4*m1**4*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1**3*m2*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1*m2**3*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        4*m2**4*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m1**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        16*m2**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        m1**6*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 6*m1**5*m2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        11*m1**4*m2**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**3*m2**3*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        m1**2*m2**4*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 6*m1*m2**5*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        3*m2**6*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 2*m1**4*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m1**2*m2**2*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1*m2**3*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m2**4*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 5*m1**2*s**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        18*m1*m2*s**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        21*m2**2*s**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        12*s**3*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 8*m1**6*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        8*m1**4*m2**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*m1**2*m2**4*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m2**6*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1**3*m2*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        32*m1**2*m2**2*s*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1*m2**3*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 48*m1*m2*s**2*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m2**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 16*s**3*np.log(1 - (4*m1**2*m2**2)/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m1**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**4*m2**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**2*m2**4*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m2**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1**3*m2*s*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m1**2*m2**2*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1*m2**3*s*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m1**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 24*m1*m2*s**2*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m2**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*s**3*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 3*m1**6*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m1**5*m2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - m1**4*m2**2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**3*m2**3*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        11*m1**2*m2**4*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m1*m2**5*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - m2**6*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m1**4*s*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1**3*m2*s*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 12*m1**2*m2**2*s*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        2*m2**4*s*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 21*m1**2*s**2*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 18*m1*m2*s**2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        5*m2**2*s**2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        12*s**3*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 8*m1**6*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        8*m1**4*m2**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*m1**2*m2**4*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m2**6*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1**3*m2*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        32*m1**2*m2**2*s*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1*m2**3*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 48*m1*m2*s**2*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m2**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 16*s**3*np.log(1 - (4*m1**2*m2**2)/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m1**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**4*m2**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**2*m2**4*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m2**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1**3*m2*s*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m1**2*m2**2*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1*m2**3*s*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m1**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 24*m1*m2*s**2*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m2**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*s**3*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 16*(m1**6 - m1**4*m2**2 - 6*m1**3*m2*s + 
        6*m1*m2*s*(-m2**2 + s) + (m2**2 - s)**2*(m2**2 + 2*s) - 
        m1**2*(m2**4 - 4*m2**2*s + 3*s**2))*mp.polylog(2, (4*m1**2*m2**2)/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2) + 
        8*(m1**6 - m1**4*m2**2 - 6*m1**3*m2*s + 6*m1*m2*s*(-m2**2 + s) + 
        (m2**2 - s)**2*(m2**2 + 2*s) - m1**2*(m2**4 - 4*m2**2*s + 3*s**2))*
        mp.polylog(2, (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*m1**6*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m1**4*m2**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m1**2*m2**4*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        8*m2**6*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        48*m1**3*m2*s*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        32*m1**2*m2**2*s*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        48*m1*m2**3*s*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**2*s**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        48*m1*m2*s**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m2**2*s**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*s**3*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))))))/
        (np.pi**2*(m1**2 + 2*m1*m2 + m2**2 - s)*s**2)
    )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    return expression

def Im_Pi_11_NLO_V_onshell(m1, m2, s):
    expression = (
        -1/24*(alpha_s*
        (np.sqrt(lambda_k(s))*(5*m1**6 + 16*m1**5*m2 - 5*m1**4*m2**2 - 
        32*m1**3*m2**3 - 5*m1**2*m2**4 + 16*m1*m2**5 + 5*m2**6 + 4*m1**4*s - 
        42*m1**3*m2*s - 68*m1**2*m2**2*s - 42*m1*m2**3*s + 4*m2**4*s - 15*m1**2*s**2 + 
        42*m1*m2*s**2 - 15*m2**2*s**2 + 6*s**3 - m1**6*np.log(4) - m2**6*np.log(4) - 
        m1**5*m2*np.log(16) - m1*m2**5*np.log(16) + 2*m1*m2*s**2*np.log(4096) + 
        2*s**3*np.log(4096) + m1**6*np.log(16384) + m2**6*np.log(16384) - 
        2*m1**2*s**2*np.log(262144) - 2*m2**2*s**2*np.log(262144) - 
        2*m1**3*m2**3*np.log(16777216) - 2*m1**3*m2*s*np.log(16777216) - 
        2*m1*m2**3*s*np.log(16777216) + m1**5*m2*np.log(268435456) + 
        m1*m2**5*np.log(268435456) - 2*m1**2*m2**2*s*np.log(281474976710656) + 
        m1**4*m2**2*np.log(1125899906842624) + m1**2*m2**4*np.log(1125899906842624) - 
        m1**4*m2**2*np.log(4611686018427387904) - 
        m1**2*m2**4*np.log(4611686018427387904) + 
        6*(3*m1**6 + 4*m1**5*m2 + 2*m1*m2*(2*m2**2 - 2*s)*(m2**2 - s) - 
        m1**4*(5*m2**2 + 2*s) + (m2**2 - s)**2*(m2**2 + 4*s) - 
        2*m1**3*m2*(4*m2**2 + 4*s) + m1**2*(m2**4 - 16*m2**2*s - 5*s**2))*np.log(abs(m1)) + 
        6*(m1**6 + 4*m1**5*m2 + m1**4*(m2**2 + 2*s) - 2*m1**3*m2*(4*m2**2 + 4*s) + 
        (m2**2 - s)*(3*m2**4 + m2**2*s - 4*s**2) + 
        2*m1*m2*(2*m2**4 - 4*m2**2*s + 2*s**2) - m1**2*(5*m2**4 + 16*m2**2*s + 
        7*s**2))*np.log(m2) + 12*m1**6*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 
        24*m1**5*m2*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 12*m1**4*m2**2*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 
        48*m1**3*m2**3*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 12*m1**2*m2**4*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 
        24*m1*m2**5*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 12*m2**6*np.log((-m1**2 - m2**2 + s + 
        np.sqrt(lambda_k(s)))**(-1)) - 
        48*m1**3*m2*s*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 96*m1**2*m2**2*s*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 
        48*m1*m2**3*s*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 36*m1**2*s**2*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 
        24*m1*m2*s**2*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) - 36*m2**2*s**2*
        np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**(-1)) + 
        24*s**3*np.log((-m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))**
        (-1)) + 4*m1**6*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 8*m1**5*m2*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        4*m1**4*m2**2*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 16*m1**3*m2**3*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        4*m1**2*m2**4*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 8*m1*m2**5*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        4*m2**6*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**3*m2*s*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 32*m1**2*m2**2*s*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1*m2**3*s*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m1**2*s**2*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1*m2*s**2*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m2**2*s**2*
        np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*s**3*np.log(-((m1**2 - m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        4*m1**6*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1**5*m2*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 4*m1**4*m2**2*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**3*m2**3*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 4*m1**2*m2**4*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1*m2**5*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 4*m2**6*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**3*m2*s*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 32*m1**2*m2**2*s*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1*m2**3*s*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m1**2*s**2*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*m1*m2*s**2*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 12*m2**2*s**2*
        np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) + 
        8*s**3*np.log(-((-m1**2 + m2**2 + s + np.sqrt(lambda_k(s)))/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))) - 
        16*m1**6*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*m1**5*m2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        16*m1**4*m2**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        64*m1**3*m2**3*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        16*m1**2*m2**4*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*m1*m2**5*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        16*m2**6*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        64*m1**3*m2*s*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        128*m1**2*m2**2*s*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        64*m1*m2**3*s*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        48*m1**2*s**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*m1*m2*s**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) + 
        48*m2**2*s**2*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2) - 
        32*s**3*np.log((m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s) - 
        2*(m1**2 + m2**2 - s)*np.sqrt(lambda_k(s)) + 
        lambda_k(s))/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)) + 
        (m1**2 + 2*m1*m2 + m2**2 - s)*
        (4*m1**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 12*m1**4*m2**2*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 12*m1**2*m2**4*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m2**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        4*m1**4*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**3*m2*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1*m2**3*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        4*m2**4*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        16*m1**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m2**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        4*m1**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 12*m1**4*m2**2*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 12*m1**2*m2**4*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m2**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        4*m1**4*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1**3*m2*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1*m2**3*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        4*m2**4*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m1**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        16*m2**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        m1**6*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 6*m1**5*m2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        11*m1**4*m2**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**3*m2**3*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        m1**2*m2**4*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 6*m1*m2**5*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        3*m2**6*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 2*m1**4*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m1**2*m2**2*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1*m2**3*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m2**4*s*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 5*m1**2*s**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        18*m1*m2*s**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        21*m2**2*s**2*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        12*s**3*np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 8*m1**6*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        8*m1**4*m2**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*m1**2*m2**4*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m2**6*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1**3*m2*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        32*m1**2*m2**2*s*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1*m2**3*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 48*m1*m2*s**2*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m2**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 16*s**3*np.log(1 - (4*m1**2*m2**2)/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m1**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**4*m2**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**2*m2**4*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m2**6*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1**3*m2*s*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m1**2*m2**2*s*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1*m2**3*s*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m1**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 24*m1*m2*s**2*
        np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m2**2*s**2*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*s**3*np.log(1 - (2*m1**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 3*m1**6*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m1**5*m2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - m1**4*m2**2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**3*m2**3*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        11*m1**2*m2**4*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m1*m2**5*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - m2**6*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        6*m1**4*s*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1**3*m2*s*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 12*m1**2*m2**2*s*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        2*m2**4*s*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 21*m1**2*s**2*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 18*m1*m2*s**2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        5*m2**2*s**2*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        12*s**3*np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 8*m1**6*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        8*m1**4*m2**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*m1**2*m2**4*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m2**6*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1**3*m2*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        32*m1**2*m2**2*s*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 48*m1*m2**3*s*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m1**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 48*m1*m2*s**2*
        np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        24*m2**2*s**2*np.log(1 - (4*m1**2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 16*s**3*np.log(1 - (4*m1**2*m2**2)/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2)*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m1**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**4*m2**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 4*m1**2*m2**4*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 4*m2**6*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1**3*m2*s*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*m1**2*m2**2*s*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 24*m1*m2**3*s*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m1**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 24*m1*m2*s**2*
        np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s))))*np.log((-2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        12*m2**2*s**2*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*s**3*np.log(1 - (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s))))*
        np.log((-2*m2**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) - 16*(m1**6 - m1**4*m2**2 - 6*m1**3*m2*s + 
        6*m1*m2*s*(-m2**2 + s) + (m2**2 - s)**2*(m2**2 + 2*s) - 
        m1**2*(m2**4 - 4*m2**2*s + 3*s**2))*mp.polylog(2, (4*m1**2*m2**2)/
        (m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))**2) + 
        8*(m1**6 - m1**4*m2**2 - 6*m1**3*m2*s + 6*m1*m2*s*(-m2**2 + s) + 
        (m2**2 - s)**2*(m2**2 + 2*s) - m1**2*(m2**4 - 4*m2**2*s + 3*s**2))*
        mp.polylog(2, (2*m1**2)/(m1**2 + m2**2 - s - np.sqrt(lambda_k(s)))) + 8*m1**6*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m1**4*m2**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        8*m1**2*m2**4*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        8*m2**6*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        48*m1**3*m2*s*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        32*m1**2*m2**2*s*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        48*m1*m2**3*s*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m1**2*s**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        48*m1*m2*s**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) - 
        24*m2**2*s**2*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))) + 
        16*s**3*mp.polylog(2, (2*m2**2)/(m1**2 + m2**2 - s - 
        np.sqrt(lambda_k(s)))))))/
        (np.pi**2*(m1**2 + 2*m1*m2 + m2**2 - s)*s**2)
    ) 

    return expression