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
print(eos.__version__)

from scipy import integrate
#import mpmath as mp
import sympy as sp

pp = eos.Parameters.Defaults()
oo = eos.Options()
mm = eos.Model.make('SM', pp, oo)

mb = 4.18
mu = mb
alpha_s = mm.alpha_s(mu)

mc = 1.44423
ms = pp['mass::s(2GeV)'].evaluate()

def DC7(s): # which is purely NLO (alpha_s)

    test_flag = 0
    
    p = eos.Parameters.Defaults()
    p.set('b->s::c1', -0.287213)
    p.set('b->s::c2', 1.009)
    p.set("mass::b(MSbar)", mb)
    p.set("mass::c", mc)
    p.set("sb::mu", mb)
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

def DC9(s):

    test_flag = 0
    
    p = eos.Parameters.Defaults()
    p.set('b->s::c1', -0.287213)
    p.set('b->s::c2', 1.009)
    p.set("mass::b(MSbar)", mb)
    p.set("mass::c", mc)
    p.set("sb::mu", mb)
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

def DC9_LO(s):

    test_flag = 0
    
    p = eos.Parameters.Defaults()
    p.set('b->s::c1', -0.287213)
    p.set('b->s::c2', 1.009)
    p.set("mass::b(MSbar)", mb)
    p.set("mass::c", mc)
    p.set("sb::mu", mb)
    #alpha_s = 0.226964 # from Nico's Mathematica at mu_b = 4.18

    q2_Re = s 
    kinematics = {"q2": q2_Re}

    re_name = "b->s::Re{{{}}}(q2)".format(f"Delta_C9_Qc_LO")
    im_name = "b->s::Im{{{}}}(q2)".format(f"Delta_C9_Qc_LO")

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

def DC9_NLO(s):
    return DC9(s) - DC9_LO(s)

def C31_LO_ms(s):
    return abs(DC9_LO(s))**2 
def mix_C31_C31(s): # LO times NLOstar
    return (DC9_LO(s) * np.conj(DC9_NLO(s))).real
def mix_C31_C32(s):
    return (DC9_LO(s) * np.conj(DC7(s))).real

def kallen(x,y,z):
    return  x*x + y*y + z*z -2 * (x*y + x*z + y*z)
def lambda_k(s):
    return max(kallen(mb*mb, ms*ms, s), 0.0)

def Disc11_LO(s):
    return -1j * np.sqrt(lambda_k(s)) * (lambda_k(s) + 3*s * (mb*mb + ms*ms - s)) / (8*np.pi * s*s) # Htheta s > (mb + ms)^2

def Disc12_LO(s):
    return -3j * mb * ms * np.sqrt(lambda_k(s)) * (mb*mb - ms*ms + s) / (4*np.pi * s*s) # Htheta s > (mb + ms)^2

def Disc22_LO(s): # ONLY CONTRIBUTING AT O(alpha_s^2) OR HIGHER
    return -1j * mb*mb * np.sqrt(lambda_k(s)) * (2*lambda_k(s) + 3*s * (mb*mb + ms*ms - s)) / (2*np.pi * s*s*s) # Htheta s > (mb + ms)^2

#-------------------------------------------------------------------------------------------------------------------
# Physical
S = sp.symbols('s', real = True)
mA, mB = sp.symbols('m_A m_B', positive = True)
# Intermediate
A = - mA*mA / S
B = - mB*mB / S

lam = 1 + 2*A + 2*B + (A-B)**2
xa = 2*A / (1 + A + B + sp.sqrt(lam))
xb = 2*B / (1 + A + B + sp.sqrt(lam))
J = -2 * (4*sp.polylog(2, xa * xb) - 2*sp.polylog(2, xa) - 2*sp.polylog(2, xb) + 2 * sp.log(sp.Abs(xa * xb)) * sp.log(1 - xa * xb) - sp.log(sp.Abs(xa)) * sp.log(1 - xa) - sp.log(sp.Abs(xb)) * sp.log(1 - xb))
Jp = 4 * sp.sqrt(lam) * (sp.log(1 - xa*xb) + (xa*xb)/(1 - xa*xb) * sp.log(sp.Abs(xa*xb))) - (B - A + sp.sqrt(lam)) * (sp.log(1 - xa) + xa/(1 - xa) * sp.log(sp.Abs(xa))) - (A - B + sp.sqrt(lam)) * (sp.log(1 - xb) + xb/(1 - xb) * sp.log(sp.Abs(xb)))

block1 = 1/6 * (11 + 19*A + 19*B + 12*A*B - 5*lam) * sp.sqrt(lam)
block2 = 4/3 * (A-B) * (sp.log(1-xb) - sp.log(1-xa))
block3 = 1/6 * (3 * (1+A+B) * ((B-A) * sp.sqrt(lam) + 1 + A + B + 2*A*B - lam) + 8*A + 26*A*B) * sp.log(sp.Abs(xa))
block4 = 1/6 * (3 * (1+A+B) * ((A-B) * sp.sqrt(lam) + 1 + A + B + 2*A*B - lam) + 8*B + 26*A*B) * sp.log(sp.Abs(xb))
block5 = 1/3 * ((A+B-2) * lam - 12*A*B) * J - 2/3 * (3*(1+A+B) - lam) * Jp
Im_Pi_T_plus_dividedby_pi_expr = block1 + block2 + block3 + block4 + block5

to_lambdify = sp.pi * Im_Pi_T_plus_dividedby_pi_expr.subs({mA: mb, mB: ms})
Im_Pi_T_plus_expr = sp.lambdify((S,), to_lambdify, modules="mpmath")
#-------------------------------------------------------------------------------------------------------------------

def Im_Pi_T_plus(s): # [9309298]
    # return float(Im_Pi_T_plus_expr(s))
    return Im_Pi_T_plus_expr(s)

def Disc11_NLO(s):
    return 1j * s * alpha_s / (4 * pow(np.pi, 3)) * Im_Pi_T_plus(s)

def to_integrate(s,Q2):
    return 1/(2j * np.pi) * (C31_LO_ms(s)*(Disc11_LO(s) + Disc11_NLO(s)) + 2 * mix_C31_C31(s)*Disc11_LO(s) + 2 * mix_C31_C32(s)*Disc12_LO(s)) / ((s - Q2)**3)

flag_check_real = 0
if (flag_check_real):
    s_vals = np.linspace((mb + ms)**2, 10000, 1000)
    for s in s_vals:
        print("s = {}, to_integrate = {}".format(s, to_integrate(s, -mb**2)))

def chi_OPE(Q2, s0, epsr = 1e-10, subd_lim = 200):
    val, err = integrate.quad(to_integrate.real, s0, np.inf, args = (Q2,), epsrel = epsr, limit = subd_lim)
    return val, err

Q2_vals = [- mb**2, 0]
flag_compute = 1
if (flag_compute):
    for Q2 in Q2_vals:
        val, err = chi_OPE(Q2, (mb + ms)**2)
        print("Q^2:\t", Q2)
        print(val, "\t+/-\t", err)