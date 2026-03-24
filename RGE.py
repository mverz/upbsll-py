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

print(f'c7 at 4.18 = {mm.wilson_coefficients_b_to_s(4.18, 1, False).c7()}, at 8.36 = {mm.wilson_coefficients_b_to_s(8.36, 1, False).c7()}, at 2.09 = {mm.wilson_coefficients_b_to_s(2.09, 1, False).c7()}\n\n')

mu_b = 4.18

MB=5.27931

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
    p.set("mass::b(MSbar)", mm.m_b_msbar(mu))
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
    p.set("mass::b(MSbar)", mm.m_b_msbar(mu))
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

F = [3.090910e-01,
     5.264856e-01]

FT = [-5.572973e-02,
      9.951136e-02]

def H(s, mu):
    if(s == -6.0): i=0
    elif(s == 0.0): i=1
    elif(s == 6.0): i=2
    else: raise ValueError("s value not recognized")

    return -1/(16*np.pi**2) * ( s/(2*MB**2) * DC9(s, mu) * F[i] + mm.m_b_msbar(mu)/MB * DC7(s, mu) * FT[i] * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) )



def A_0L(s, mu):
    if(s == -6.0): i=0
    elif(s == 0.0): i=1
    elif(s == 6.0): i=2
    else: raise ValueError("s value not recognized")

    wc = wc_at(mu)

    return (wc.c9()- wc.c10()) * F[i] + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT[i] * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )

def A_0L_local(s, mu):
    if(s == -6.0): i=0
    elif(s == 0.0): i=1
    elif(s == 6.0): i=2
    else: raise ValueError("s value not recognized")

    wc = wc_at(mu)

    return (wc.c9()- wc.c10()) * F[i] + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT[i] * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) )

def A_0L_nonlocal(s, mu):
    if(s == -6.0): i=0
    elif(s == 0.0): i=1
    elif(s == 6.0): i=2
    else: raise ValueError("s value not recognized")

    wc = wc_at(mu)

    return 2 * mm.m_b_msbar(mu)*MB / s * ( - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )


def A_0R(s, mu):
    if(s == -6.0): i=0
    elif(s == 0.0): i=1
    elif(s == 6.0): i=2
    else: raise ValueError("s value not recognized")

    wc = wc_at(mu)

    return (wc.c9()+ wc.c10()) * F[i] + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT[i] * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )

def A_0R_local(s, mu):
    if(s == -6.0): i=0
    elif(s == 0.0): i=1
    elif(s == 6.0): i=2
    else: raise ValueError("s value not recognized")

    wc = wc_at(mu)

    return (wc.c9()+ wc.c10()) * F[i] + 2 * mm.m_b_msbar(mu)*MB / s * ( wc.c7() * FT[i] * tensor_current_evol(mm.alpha_s(mu_b), mm.alpha_s(mu)) )

def A_0R_nonlocal(s, mu):
    if(s == -6.0): i=0
    elif(s == 0.0): i=1
    elif(s == 6.0): i=2
    else: raise ValueError("s value not recognized")

    wc = wc_at(mu)

    return 2 * mm.m_b_msbar(mu)*MB / s * ( - 16*np.pi**2 * MB/mm.m_b_msbar(mu) * H(s, mu) )


# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0L = {A_0L(s, mu)}, mod_sq = {abs(A_0L(s, mu))**2}")

# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0L = {A_0L_local(s, mu)}, mod_sq = {abs(A_0L_local(s, mu))**2}")

# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0L = {A_0L_nonlocal(s, mu)}, mod_sq = {abs(A_0L_nonlocal(s, mu))**2}")



# print("\n\n")



# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0R = {A_0R(s, mu)}, mod_sq = {abs(A_0R(s, mu))**2}")

# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0R = {A_0R_local(s, mu)}, mod_sq = {abs(A_0R_local(s, mu))**2}")

# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: A_0R = {A_0R_nonlocal(s, mu)}, mod_sq = {abs(A_0R_nonlocal(s, mu))**2}")



# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     for s in [-6.0, 6.0]:
#         print(f"s = {s} GeV^2: |L|^2 + |R|^2 = {abs(A_0L(s, mu))**2 + abs(A_0R(s, mu))**2}")



# for mu in [mu_b, mu_b*2, mu_b/2]:
#     print(f"mu = {mu} GeV")
#     print(f"C1 = {wc_at(mu).c1()}, C2 = {wc_at(mu).c2()}, C7 = {wc_at(mu).c7()}, C9 = {wc_at(mu).c9()}, C10 = {wc_at(mu).c10()}\n")