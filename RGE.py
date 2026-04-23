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

print("EOS: " + eos.__version__)

pp = eos.Parameters.Defaults()
oo = eos.Options()
mm = eos.Model.make('SM', pp, oo)

mu_b = 4.18

pp.set("mass::B_d", 5.27931)
MB = pp['mass::B_d'].evaluate()

#---

BASE_DIRECTORY='./test_base'
ANALYSIS_FILE='./an_file_BK.yaml'
POSTERIOR_NAME = 'BSZ-BqToK-wSR-wNFF-wCov'

bfp, gof = eos.tasks.find_mode(
        ANALYSIS_FILE,
        POSTERIOR_NAME,
        BASE_DIRECTORY,
        importance_samples=True,
        label='EOS',
        optimizations=50
    )
display(bfp)
display(gof)

for par, val in zip(bfp.analysis.varied_parameters, bfp.point):
    pp.set(par.name(), float(val))


opt = eos.Options({'form-factors':'BSZ2015', 'nonlocal-formfactor':'GRvDV2022order5', 'model':'SM'})

def F(s):
    kin = eos.Kinematics({"q2": float(s)})

    re_obs = eos.Observable.make("B->K::F_plus(q2)", pp, kin, opt).evaluate()

    return re_obs
# def F(s):
#     if s==-6:
#         return 0.3090910
#     elif s==6:
#         return 0.5264856
#     else:
#         raise ValueError("s value not recognized")
def FT(s):
    kin = eos.Kinematics({"q2": float(s)})

    re_obs = eos.Observable.make("B->K::F_plus_T(q2)", pp, kin, opt).evaluate()

    return re_obs
# def FT(s):
#     if s==-6:
#         return -0.05572973
#     elif s==6:
#         return 0.09951136
#     else:
#         raise ValueError("s value not recognized")


#---

nf = 5

#---

anom_dim_coeffs = [-1/3, -181/72 + nf * 13/108]
beta_coeffs = [11 - 2/3 * nf, 102 - 38/3 * nf]

anom_dim = 0
beta_alpha_s = 0

alpha_s = sp.symbols(R'\alpha_s', real = True)

for i in range(len(anom_dim_coeffs)):
    anom_dim += anom_dim_coeffs[i] * (alpha_s / np.pi)**(i + 1)

for i in range(len(beta_coeffs)):
    beta_alpha_s += (-alpha_s) * beta_coeffs[i] * (alpha_s / (4*np.pi))**(i + 1) 

to_integrate = sp.lambdify((alpha_s,), anom_dim / beta_alpha_s, modules="mpmath")

def tensor_current_evol(alpha_s_mu_0, alpha_s_mu):
    return np.exp( integrate.quad(to_integrate, alpha_s_mu_0, alpha_s_mu, epsrel=1e-9)[0] )

def print_tensor_current_evol_error(alpha_s_mu_0, alpha_s_mu):
    print(f"e^{integrate.quad(to_integrate, alpha_s_mu_0, alpha_s_mu, epsrel=1e-9)[1]}" )

# print(tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu_b*2)))
# print(tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu_b/2)))

#evaluate the amplitude at 3 different q2 points: [-6,0,6] GeV^2 

def wc_at(mu):
    return mm.wilson_coefficients_b_to_s(mu, 1, False)

def DC7(s, mu): # which is purely NLO (alpha_s)

    test_flag = 0

    wc = wc_at(mu)
    
    mc = 1.44423

    p = eos.Parameters.Defaults()
    # p.set('b->s::c1', -0.287213)
    # p.set('b->s::c2', 1.009)
    p.set('b->s::c1', np.real(wc.c1()))
    p.set('b->s::c2', np.real(wc.c2()))
    #p.set("mass::b(MSbar)", mm.m_b_msbar(mu))
    p.set("mass::c", mc)
    p.set("sb::mu", mu)
    #alpha_s = 0.226964 # from Nico's Mathematica at mu_b = 4.18

    q2_Re = s 
    kinematics = {"q2": q2_Re}

    re_name = "b->s::Re{{{}}}(q2)".format(f"Delta_C7_Qc")
    im_name = "b->s::Im{{{}}}(q2)".format(f"Delta_C7_Qc")

    if test_flag:
        print("Observable names:")
        print(re_name)
        print(im_name)

    # make observables and evaluate
    real_part = eos.Observable.make(re_name, p, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()
    imag_part = eos.Observable.make(im_name, p, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()

    if test_flag:
        print("Real and Imaginary parts:")
        print("Re = {}, Im = {}".format(real_part, imag_part))

    # Must add the fLO

    DC_Re = real_part
    DC_Im = imag_part

    return DC_Re + 1j * DC_Im

def DC9(s, mu):

    test_flag = 0

    wc = wc_at(mu)

    mc = 1.44423
    
    p = eos.Parameters.Defaults()
    # p.set('b->s::c1', -0.287213)
    # p.set('b->s::c2', 1.009)
    p.set('b->s::c1', np.real(wc.c1()))
    p.set('b->s::c2', np.real(wc.c2()))
    #p.set("mass::b(MSbar)", mm.m_b_msbar(mu))
    p.set("mass::c", mc)
    p.set("sb::mu", mu)
    #alpha_s = 0.226964 # from Nico's Mathematica at mu_b = 4.18

    q2_Re = s
    kinematics = {"q2": q2_Re}

    re_name = "b->s::Re{{{}}}(q2)".format(f"Delta_C9_Qc")
    im_name = "b->s::Im{{{}}}(q2)".format(f"Delta_C9_Qc")

    if test_flag:
        print("Observable names:")
        print(re_name)
        print(im_name)

    # make observables and evaluate
    real_part = eos.Observable.make(re_name, p, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()
    imag_part = eos.Observable.make(im_name, p, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()

    if test_flag:
        print("Real and Imaginary parts:")
        print("Re = {}, Im = {}".format(real_part, imag_part))

    # Must add the fLO

    DC_Re = real_part
    DC_Im = imag_part

    return DC_Re + 1j * DC_Im


def H(s, mu):

    return -1/(16*np.pi**2) * ( s/(2*MB**2) * DC9(s, mu) * F(s) + mm.m_b_msbar(mu)/MB * DC7(s, mu) * FT(s) * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) )



def A_0L(s, mu):

    wc = wc_at(mu)

    return (wc.c9()- wc.c10()) * F(s) + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT(s) * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )

def A_0L_local(s, mu):

    wc = wc_at(mu)

    return (wc.c9()- wc.c10()) * F(s) + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT(s) * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) )

def A_0L_nonlocal(s, mu):

    wc = wc_at(mu)

    return 2 * mm.m_b_msbar(mu)*MB / s * ( - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )


def A_0R(s, mu):

    wc = wc_at(mu)

    return (wc.c9()+ wc.c10()) * F(s) + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT(s) * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )

def A_0R_local(s, mu):

    wc = wc_at(mu)

    return (wc.c9()+ wc.c10()) * F(s) + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT(s) * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) )

def A_0R_nonlocal(s, mu):

    wc = wc_at(mu)

    return 2 * mm.m_b_msbar(mu)*MB / s * ( - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )

# L amplitude

# for mu in [mu_b/2, mu_b, mu_b*2]:
#     print(f"\nmu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0L = {A_0L(s, mu)}, mod_sq = {abs(A_0L(s, mu))**2}")

# for mu in [mu_b/2, mu_b, mu_b*2]:
#     print(f"\nmu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0L = {A_0L_local(s, mu)}, mod_sq = {abs(A_0L_local(s, mu))**2}")

# for mu in [mu_b/2, mu_b, mu_b*2]:
#     print(f"\nmu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0L = {A_0L_nonlocal(s, mu)}, mod_sq = {abs(A_0L_nonlocal(s, mu))**2}")

# R amplitude

# for mu in [mu_b/2, mu_b, mu_b*2]:
#     print(f"\nmu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0R = {A_0R(s, mu)}, mod_sq = {abs(A_0R(s, mu))**2}")

# for mu in [mu_b/2, mu_b, mu_b*2]:
#     print(f"\nmu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0R = {A_0R_local(s, mu)}, mod_sq = {abs(A_0R_local(s, mu))**2}")

# for mu in [mu_b/2, mu_b, mu_b*2]:
#     print(f"\nmu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0R = {A_0R_nonlocal(s, mu)}, mod_sq = {abs(A_0R_nonlocal(s, mu))**2}")


# PRINT NICO'S TABLE

for q2 in [-6.0, 1.0, 3.0, 6.0]:
    if(False):
        print(f"F({q2}) = {F(q2)}, FT({q2}) = {FT(q2)}\n")
    for mu in [mu_b/2, mu_b, mu_b*2]:
        if(True):
            print(f"q2 = {q2} GeV^2, mu = {mu} GeV\n")
            
        print(f"AKL = {A_0L(q2, mu):<.6f}, ReRatio = {A_0L(q2, mu).real/A_0L(q2, mu_b).real:<.6f}, ImRatio = {A_0L(q2, mu).imag/A_0L(q2, mu_b).imag:<.6f}, "+
              f"AKR = {A_0R(q2, mu):<.6f}, ReRatio = {A_0R(q2, mu).real/A_0R(q2, mu_b).real:<.6f}, ImRatio = {A_0R(q2, mu).imag/A_0R(q2, mu_b).imag:<.6f}, "+
              f"|L|^2+|R|^2 = {abs(A_0L(q2, mu))**2 + abs(A_0R(q2, mu))**2:<.6f}, Ratio = {(abs(A_0L(q2, mu))**2 + abs(A_0R(q2, mu))**2)/(abs(A_0L(q2, mu_b))**2 + abs(A_0R(q2, mu_b))**2):<.6f}\n")