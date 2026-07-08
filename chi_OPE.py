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

import long_exprs as exprs

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
MBs1 = 5.829
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


def wc(mu):
    return mm.wilson_coefficients_b_to_smumu(mu, False)


wc = wc(mu)


def DC7(s): # which is purely NLO (alpha_s)

    test_flag = 0

    p = eos.Parameters.Defaults()
    p.set('b->s::c1', -0.29063621)
    p.set('b->s::c2', 1.01029623)
    # p.set('b->s::c1', np.real(wc.c1()))
    # p.set('b->s::c2', np.real(wc.c2()))
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
    real_part = eos.Observable.make(re_name, pp, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()
    imag_part = eos.Observable.make(im_name, pp, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()

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
    p.set('b->s::c1', -0.29063621)
    p.set('b->s::c2', 1.01029623)
    # p.set('b->s::c1', np.real(wc.c1()))
    # p.set('b->s::c2', np.real(wc.c2()))
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
    real_part = eos.Observable.make(re_name, pp, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()
    imag_part = eos.Observable.make(im_name, pp, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()

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
    p.set('b->s::c1', -0.29063621)
    p.set('b->s::c2', 1.01029623)
    # p.set('b->s::c1', np.real(wc.c1()))
    # p.set('b->s::c2', np.real(wc.c2()))
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
    real_part = eos.Observable.make(re_name, pp, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()
    imag_part = eos.Observable.make(im_name, pp, eos.Kinematics(kinematics), eos.Options({"model":"SM"})).evaluate()

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
    return  x*x + y*y + z*z - 2 * (x*y + x*z + y*z)

def lambda_k(s):
    return abs(kallen(mb*mb, ms*ms, s))

def Disc11_LO_V(s, mb = mb, ms = ms):
    return -(1j * (mb**4 + ms**4 - 6 * mb * ms * s + ms**2 * s - 2 * s**2 + mb**2 * (-2 * ms**2 + s)) * np.sqrt(lambda_k(s)))/(16 * np.pi * s**2) # Htheta s > (mb + ms)^2
def Disc11_LO_A(s, mb = -mb, ms = ms):
    return -(1j * (mb**4 + ms**4 - 6 * mb * ms * s + ms**2 * s - 2 * s**2 + mb**2 * (-2 * ms**2 + s)) * np.sqrt(lambda_k(s)))/(16 * np.pi * s**2) # Htheta s > (mb + ms)^2

def Disc12_LO_V(s, mb = mb, ms = ms):
    return 3j * mb * (mb + ms) * (mb**2 - 2 * mb*ms + ms**2 - s) * np.sqrt(lambda_k(s)) / (8 * np.pi * s**2) # Htheta s > (mb + ms)^2
def Disc12_LO_A(s, mb = -mb, ms = ms):
    return 3j * mb * (mb + ms) * (mb**2 - 2 * mb*ms + ms**2 - s) * np.sqrt(lambda_k(s)) / (8 * np.pi * s**2) # Htheta s > (mb + ms)^2

def Disc22_LO_V(s, mb = mb, ms = ms): # ONLY CONTRIBUTING AT O(alpha_s^2) OR HIGHER
    return 1j * mb**2/s**2 * (-2 * mb**4 - 2 * ms**4 + 6 * mb * ms * s + ms**2 * s + s**2 + mb**2 * (4 * ms**2 + s)) * np.sqrt(lambda_k(s))/(4 * np.pi * s) # Htheta s > (mb + ms)^2
def Disc22_LO_A(s, mb = -mb, ms = ms): # ONLY CONTRIBUTING AT O(alpha_s^2) OR HIGHER
    return 1j * mb**2/s**2 * (-2 * mb**4 - 2 * ms**4 + 6 * mb * ms * s + ms**2 * s + s**2 + mb**2 * (4 * ms**2 + s)) * np.sqrt(lambda_k(s))/(4 * np.pi * s) # Htheta s > (mb + ms)^2

def Im_Pi_11_NLO_V(s):
    expression = exprs.Im_Pi_11_NLO_V(mb,ms,s)
    return expression
def Im_Pi_11_NLO_V_onshell(s):
    expression = exprs.Im_Pi_11_NLO_V_onshell(mb,ms,s)
    return expression
def Im_Pi_11_NLO_A(s):
    expression = exprs.Im_Pi_11_NLO_V(-mb,ms,s)
    return expression
def Im_Pi_11_NLO_A_onshell(s):
    expression = exprs.Im_Pi_11_NLO_V_onshell(-mb,ms,s)
    return expression

def Disc11_NLO_V(s):
    return (1/2.0)**2 * 2j * Im_Pi_11_NLO_V(s)
def Disc11_NLO_V_onshell(s):
    return (1/2.0)**2 * 2j * Im_Pi_11_NLO_V_onshell(s)
def Disc11_NLO_A(s):
    return (1/2.0)**2 * 2j * Im_Pi_11_NLO_A(s)
def Disc11_NLO_A_onshell(s):
    return (1/2.0)**2 * 2j * Im_Pi_11_NLO_A_onshell(s)

n_subtractions_plus_1 = 3

with_res_factor = True
print("Use the res function? (y/n)")
if (input() == 'y'):
    with_res_factor = True
    print("Including the resonance factor in the integrand.")
else:
    with_res_factor = False
    print("Not including the resonance factor in the integrand.")

# modify n_subtractions_plus_1 to change the number of subtractions in the dispersion relation. Must be >= 3 for convergence of the integral, but can be higher if desired (e.g. to suppress more the high-s region).
def to_integrate_V(s, Q2, n_subtractions_plus_1 = n_subtractions_plus_1, with_res_factor=with_res_factor):
    res_factor_V = (s - MBs_star**2)**2 / (s + Q2)**2

    if(with_res_factor):
        output = (1/(2j * np.pi) * res_factor_V * (C31_LO_ms(s)*(Disc11_LO_V(s) + Disc11_NLO_V(s)) + 2 * mix_C31_C31(s)*Disc11_LO_V(s) + 2 * mix_C31_C32(s)*Disc12_LO_V(s)) / ((s + Q2)**(n_subtractions_plus_1))).real
    else:
        output = (1/(2j * np.pi) * (C31_LO_ms(s)*(Disc11_LO_V(s) + Disc11_NLO_V(s)) + 2 * mix_C31_C31(s)*Disc11_LO_V(s) + 2 * mix_C31_C32(s)*Disc12_LO_V(s)) / ((s + Q2)**(n_subtractions_plus_1))).real

    return output


def to_integrate_V_onshell(s, Q2, n_subtractions_plus_1 = n_subtractions_plus_1, with_res_factor=with_res_factor):
    res_factor_V = (s - MBs_star**2)**2 / (s + Q2)**2

    if(with_res_factor):
        output = (1/(2j * np.pi) * res_factor_V * (C31_LO_ms(s)*(Disc11_LO_V(s) + Disc11_NLO_V_onshell(s)) + 2 * mix_C31_C31(s)*Disc11_LO_V(s) + 2 * mix_C31_C32(s)*Disc12_LO_V(s)) / ((s + Q2)**(n_subtractions_plus_1))).real
    else:
        output = (1/(2j * np.pi) * (C31_LO_ms(s)*(Disc11_LO_V(s) + Disc11_NLO_V_onshell(s)) + 2 * mix_C31_C31(s)*Disc11_LO_V(s) + 2 * mix_C31_C32(s)*Disc12_LO_V(s)) / ((s + Q2)**(n_subtractions_plus_1))).real

    return output


def to_integrate_A(s, Q2, n_subtractions_plus_1 = n_subtractions_plus_1, with_res_factor=with_res_factor):
    res_factor_A = (s - MBs1**2)**2 / (s + Q2)**2

    if(with_res_factor):
        output = (1/(2j * np.pi) * res_factor_A * (C31_LO_ms(s)*(Disc11_LO_A(s) + Disc11_NLO_A(s)) + 2 * mix_C31_C31(s)*Disc11_LO_A(s) - 2 * mix_C31_C32(s)*Disc12_LO_A(s)) / ((s + Q2)**(n_subtractions_plus_1))).real
    else:
        output = (1/(2j * np.pi) * (C31_LO_ms(s)*(Disc11_LO_A(s) + Disc11_NLO_A(s)) + 2 * mix_C31_C31(s)*Disc11_LO_A(s) - 2 * mix_C31_C32(s)*Disc12_LO_A(s)) / ((s + Q2)**(n_subtractions_plus_1))).real

    return output


def to_integrate_A_onshell(s, Q2, n_subtractions_plus_1 = n_subtractions_plus_1, with_res_factor=with_res_factor):
    res_factor_A = (s - MBs1**2)**2 / (s + Q2)**2

    if(with_res_factor):
        output = (1/(2j * np.pi) * res_factor_A * (C31_LO_ms(s)*(Disc11_LO_A(s) + Disc11_NLO_A_onshell(s)) + 2 * mix_C31_C31(s)*Disc11_LO_A(s) - 2 * mix_C31_C32(s)*Disc12_LO_A(s)) / ((s + Q2)**(n_subtractions_plus_1))).real
    else:
        output = (1/(2j * np.pi) * (C31_LO_ms(s)*(Disc11_LO_A(s) + Disc11_NLO_A_onshell(s)) + 2 * mix_C31_C31(s)*Disc11_LO_A(s) - 2 * mix_C31_C32(s)*Disc12_LO_A(s)) / ((s + Q2)**(n_subtractions_plus_1))).real

    return output

flag_check_real = False
if (flag_check_real): # Note that the integrand is real for s > (mb + ms)^2, so we take the real part of the expression.
    s_vals = np.linspace((mb + ms)**2, 10000, 100)
    for s in s_vals:
        print("s = {}, to_integrate = {}".format(s, to_integrate_V(s, mb**2)))

def chi_OPE_V(Q2, s_plus = (mb + ms)**2, epsrel = 1e-5, subd_lim = 25):
    val, err = integrate.quad(to_integrate_V, s_plus, np.inf, args = (Q2,), epsrel = epsrel, limit = subd_lim)
    return val

def chi_OPE_V_onshell(Q2, s_plus = (mb + ms)**2, epsrel = 1e-5, subd_lim = 25):
    val, err = integrate.quad(to_integrate_V_onshell, s_plus, np.inf, args = (Q2,), epsrel = epsrel, limit = subd_lim)
    return val

def chi_OPE_A(Q2, s_plus = (mb + ms)**2, epsrel = 1e-5, subd_lim = 25):
    val, err = integrate.quad(to_integrate_A, s_plus, np.inf, args = (Q2,), epsrel = epsrel, limit = subd_lim)
    return val

def chi_OPE_A_onshell(Q2, s_plus = (mb + ms)**2, epsrel = 1e-5, subd_lim = 25):
    val, err = integrate.quad(to_integrate_A_onshell, s_plus, np.inf, args = (Q2,), epsrel = epsrel, limit = subd_lim)
    return val


Q2_vals = [+ mb**2]
flag_compute_V = False
if (flag_compute_V):
    for Q2 in Q2_vals:
        val = chi_OPE_V(Q2, (mb + ms)**2)
        print("MSbar\nQ^2 = ", Q2)
        print("chi_V = " , val)

Q2_vals = [+ mb**2]
flag_compute_V_onshell = False
if (flag_compute_V_onshell):
    for Q2 in Q2_vals:
        val = chi_OPE_V_onshell(Q2, (mb + ms)**2)
        print("On-shell\nQ^2 = ", Q2)
        print("chi_V = " , val)


Q2_vals = [+ mb**2]
flag_compute_A = False
if (flag_compute_A):
    for Q2 in Q2_vals:
        val = chi_OPE_A(Q2, (mb + ms)**2)
        print("MSbar\nQ^2 = ", Q2)
        print("chi_A = " , val)

Q2_vals = [+ mb**2]
flag_compute_A_onshell = False
if (flag_compute_A_onshell):
    for Q2 in Q2_vals:
        val = chi_OPE_A_onshell(Q2, (mb + ms)**2)
        print("On-shell\nQ^2 = ", Q2)
        print("chi_A = " , val)



# chi Tilde OPE implementation ------ sG --- sminus --- splus ------

if(True):

    BASE_DIRECTORY='./test_base'
    ANALYSIS_FILE='./afBK.yaml'
    POSTERIOR_NAME = 'GRV-BK-HPQCD'

    bfp, gof = eos.tasks.find_mode(
            ANALYSIS_FILE,
            POSTERIOR_NAME,
            BASE_DIRECTORY,
            importance_samples=True,
            label='EOS',
            optimizations=25
        )
    display(bfp)
    display(gof)

    for par, val in zip(bfp.analysis.varied_parameters, bfp.point):
        pp.set(par.name(), float(val))

    options = {'form-factors':'G2026', 'nonlocal-formfactor':'GRV2026', 'model':'WET'}
    opt = eos.Options(options)

    def F_0_BToK(s):

        kin = eos.Kinematics({"q2": float(s)})

        re_obs = eos.Observable.make("B->K::F_plus(q2)", pp, kin, opt).evaluate()

        return re_obs

    def FT_0_BToK(s):
        kin = eos.Kinematics({"q2": float(s)})

        re_obs = eos.Observable.make("B->K::F_plus_T(q2)", pp, kin, opt).evaluate()

        return re_obs

    def H_0_BToK_modsq(s):

        s_min = (MB - MK)**2
        s_plus = (MB + MK)**2
        s_G = 4 * MD**2

        if (s <= s_min and s >= s_G):
            # kin = eos.Kinematics({"q2": float(s)})

            # re_obs = eos.Observable.make("B->K::Re{H_plus}(q2)", pp, kin, opt).evaluate()
            # im_obs = eos.Observable.make("B->K::Im{H_plus}(q2)", pp, kin, opt).evaluate()

            # return re_obs**2 + im_obs**2

            return 1/(16*np.pi**2)**2 * ( s**2/(4 * MB**4) * abs(DC9(s))**2 * (F_0_BToK(s))**2 + mb**2/MB**2 * abs(DC7(s))**2 * (FT_0_BToK(s))**2 + s*mb/(2*MB**3) * 2*(DC9(s) * DC7(s).conjugate()).real * (F_0_BToK(s) * FT_0_BToK(s)) )

        if (s > s_min and s <= s_plus):
            K_estimate = 100.0

            return 1/(16*np.pi**2)**2 * ( s**2/(4 * MB**4) * abs(DC9(s))**2 + mb**2/MB**2 * abs(DC7(s))**2 + s*mb/(2*MB**3) * 2*(DC9(s) * DC7(s).conjugate()).real ) * K_estimate*(s_G/s)**2

        else:
            raise ValueError("s value out of range for H_0_BToK(s)")



    def to_integrate_delta_V(s, Q2, n_subtractions_plus_1 = n_subtractions_plus_1, with_res_factor = with_res_factor):
        res_factor = (s - MBs_star**2)**2 / (s + Q2)**2

        if(with_res_factor):
            output = res_factor * 32*np.pi**2/3.0 * MB**4 * pow( abs(kallen(MB**2,MK**2,s)) , 1.5 )/( s**4 * (s + Q2)**(n_subtractions_plus_1) ) * H_0_BToK_modsq(s)
        else:
            print("Warning: not including the resonance factor in the integrand is not supported.")
            output = 0.0

        return output

    def Delta_chi_V(Q2, s_G = 4 * MD**2, s_plus = (MB + MK)**2, epsrel = 1e-6, subd_lim = 100):
        val, err = integrate.quad(to_integrate_delta_V, s_G, s_plus, args = (Q2,), epsrel = epsrel, limit = subd_lim)
        return val

    def Delta_chi_V1(Q2, s_G = 4 * MD**2, s_minus = (MB - MK)**2, epsrel = 1e-6, subd_lim = 100):
        val, err = integrate.quad(to_integrate_delta_V, s_G, s_minus, args = (Q2,), epsrel = epsrel, limit = subd_lim)
        return val
    def Delta_chi_V2(Q2, s_minus = (MB - MK)**2, s_plus = (MB + MK)**2, epsrel = 1e-6, subd_lim = 100):
        val, err = integrate.quad(to_integrate_delta_V, s_minus, s_plus, args = (Q2,), epsrel = epsrel, limit = subd_lim)
        return val


    def to_integrate_delta_A(s, Q2, n_subtractions_plus_1 = n_subtractions_plus_1, with_res_factor = with_res_factor):
        res_factor = (s - MBs1**2)**2 / (s + Q2)**2

        if(with_res_factor):
            output = res_factor * 32*np.pi**2/3.0 * MB**4 * pow( abs(kallen(MB**2,MK**2,s)) , 1.5 )/( s**4 * (s + Q2)**(n_subtractions_plus_1) ) * H_0_BToK_modsq(s)
        else:
            print("Warning: not including the resonance factor in the integrand is not supported.")
            output = 0.0

        return output

    def Delta_chi_A(Q2, s_G = 4 * MD**2, s_plus = (MB + MK)**2, epsrel = 1e-5, subd_lim = 25):
        val, err = integrate.quad(to_integrate_delta_A, s_G, s_plus, args = (Q2,), epsrel = epsrel, limit = subd_lim)
        return val


    if(True):
        for Q2 in [mb**2]:
            tot = Delta_chi_V(Q2)
            part1 = Delta_chi_V1(Q2)
            part2 = Delta_chi_V2(Q2)
            print(f"Delta chi_V (Q2 = {Q2:<.3e}) = {tot:<.4e}")
            print(f"{part1:<.4e} + {part2:<.4e} = {(part1 + part2):<.4e}")
            #print(f"Delta chi_V (Q2 = {Q2:<.3e}) = {Delta_chi_V(Q2):<.4e} \nDelta chi_A (Q2 = {Q2:<.3e}) = {Delta_chi_A(Q2):<.4e}")