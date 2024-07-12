# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:00:57 2023

@author: d/dt Lucas
"""


import numpy as np
from solvegnlse import SolveNLS, Raman
from solvepcnlse import Solve_pcNLSE
from solvepcGNLSE import Solve_pcGNLSE
from commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, find_k
from plotter import plotinst, plotevol, plotspecgram, plotenergia, plotfotones, plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from functools import partial


#Time
import time

#AW, AT, sim, fibra = loader("soliton_gen/sg1", resim = True)
AW, AT, sim, fibra = loader("power_test/0-5", resim = True)
#AW, AT, metadata = loader("power_test/6-5", resim = False)
zlocs = np.linspace(0, 300, len(AT))

AT = np.vstack(AT)
AW = np.vstack(AW)

#%% Plotting

plotevol(sim, fibra, zlocs, AT, AW, wavelength=True)

plt.figure()
plt.plot(sim.tiempo, Pot(AT[39]))
plt.show()

plotinst(sim, fibra, AT, AW, wavelength=True, zeros=True, Wlim=[1400,1700], end=49)

#%% k-test

A_test = np.zeros_like(AW)

ks, phis = find_k(A_test, zlocs[0]-zlocs[1])

plt.plot(fftshift(ks[0]))

#%%

#Soliton peak power vs. signal peak power

#t0_signal = 2.1ps
sig_peak = np.array([0,10,20,30,40,50,60,70,80,90,100])
sol_peak = np.array([243,282,460,570,610,640,695,693,810,807,866])

#t0_signal = 0.5ps
sig_peak2 = np.array([10,15,20,30,40,50,60,70,80,90,110,130])
sol_peak2 = np.array([223,225,258,270,326,352,356,445,498.5,475,527.2,559])
ref_lam   = np.array([1548,1548,1548,1550,1551,1552,1554.5,1557,1558,1559.5,1562,1564.2])


sol_dk  = fibra.gamma * sol_peak  / 2 - fibra.gamma * 243/2
sol_dk2 = fibra.gamma * sol_peak2 / 2 - fibra.gamma * 243/2

# =============================================================================
# plt.figure()
# plt.plot(sig_peak, sol_k)
# plt.ylabel("$\\frac{1}{2}\gamma P_0$ (m$^{-1}$)")
# plt.xlabel("signal peak power (W)")
# plt.grid(True, alpha=.3)
# plt.show()
# 
# =============================================================================

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
#ax1.plot(sig_peak2, sol_peak2, 'g.-')
ax1.plot(sig_peak2, sol_dk2, 'g.-')
ax2.plot(sig_peak2, ref_lam, 'b.-')

ax1.set_xlabel('signal peak power (W)')
ax1.set_ylabel('soliton $\Delta k$ (m$^{-1}$)', color='g')
ax2.set_ylabel('reflected wavelength (nm)', color='b')
plt.tight_layout()
plt.show()

#%% Finding reflected wavelength



c = 299792458 * (1e9)/(1e12)

lambda_0 = 1600 #(nm)

lambda_i = 1480 #(nm)

omega_i = 2*np.pi*c/lambda_i - 2*np.pi*c/lambda_0

def lam_from_omega(omega, lambda_0):
    c = 299792458 * (1e9)/(1e12)
    lam = 2*np.pi*c/( omega + 2*np.pi*c/lambda_0  )
    return lam

def beta2w(freq, fib:Fibra):
    return fib.beta2 + fib.beta3  * (2*np.pi*freq)

def beta1w(freq, fib:Fibra):
    return fib.beta2 * (2*np.pi*freq) + fib.beta3/2  * (2*np.pi*freq)**2
    
#lam_from_omega(-omega_i, lambda_0)

omega_r = omega_i - 2*beta1w(omega_i/(2*np.pi), fibra)/beta2w(omega_i/(2*np.pi), fibra)

lam_from_omega(omega_r, lambda_0)

#%%



#%%

delta_k = .1

def ref_omega(x,fibra:Fibra, omega_i, delta_k):
    eq = fibra.beta2*omega_i**2/2 + fibra.beta3*omega_i**3/6 - fibra.beta2*x**2/2 - fibra.beta3*x**3/6 - delta_k
    return eq

ref_omega_p = partial(ref_omega, fibra=fibra, omega_i = omega_i, delta_k = delta_k)

omega_r = fsolve(ref_omega_p, -90)

lambda_r = lam_from_omega(omega_r, lambda_0)

print("Reflected wavelength = "+str(lambda_r[0])+" nm")

#%%

def Omega_p(Tr, beta21, z, T1):
    return -8 * Tr * np.abs(beta21) * z / (15 * T1**4) 


def delta_r(omega_p0, beta21, beta22, delta_i):
    return 2*omega_p0*beta21/beta22 - delta_i   

z_aprox = 120

lambda_signal_0 = 1480
nu_0  = c / 1600
nu_signal_0 = c/lambda_signal_0
dnu_signal_0 = (nu_signal_0 - nu_0)

Tr = 3e-3
beta21 = beta2w(0, fibra)
beta22 = beta2w(dnu_signal_0,fibra)
beta32 = fibra.beta3
omega_p0 = Omega_p(fibra.TR, beta21, z_aprox, 85e-3)*1

d_r      = delta_r(omega_p0, beta21, beta22, dnu_signal_0)

lambda_reflection = c / (d_r + nu_0)

print("Estimated reflection wavelength: "+str(np.trunc(lambda_reflection))+" nm")

# =============================================================================
# def ref_omega_raman(x, beta21, beta22, beta32, omega_i, delta_k, omega_p0):
#     left_side = omega_p0*beta21*omega_i + beta22/2*omega_i**2 + beta32/6*omega_i**3 - delta_k
#     eq = left_side - omega_p0*beta21*x - beta22/2*x**2 - beta32/6*x**3
#     return eq
# =============================================================================

def ref_omega_raman(x, beta2, beta3, omega_i, delta_k, omega_p0):
    left_side = omega_p0*beta2*omega_i + beta2/2*omega_i**2 + beta3/6*omega_i**3 - delta_k
    eq = left_side - omega_p0*beta2*x - beta2/2*x**2 - beta3/6*x**3
    return eq


ref_omega_raman_p = partial(ref_omega_raman, beta2=fibra.beta2, beta3=fibra.beta3, omega_i=omega_i, delta_k=delta_k, omega_p0=omega_p0)

omega_r_raman = fsolve(ref_omega_raman_p, -90)

lambda_r_raman = lam_from_omega(omega_r_raman, lambda_0)

print("Reflected wavelength = "+str(lambda_r_raman[0])+" nm")