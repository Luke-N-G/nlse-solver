# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:00:57 2023

@author: d/dt Lucas
"""


import numpy as np
from common.commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from common.commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, find_k
from common.plotter import plotinst, plotcmap, plotspecgram, plotenergia, plotfotones, plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from functools import partial

#Time
import time

AW, AT, sim, fibra = loader("soliton_gen/1470", resim = True)
zlocs = np.linspace(0, 300, len(AT))


#%% Plotting

plotcmap(sim, fibra, zlocs, AT, AW, legacy=True, dB=True, wavelength=True,cmap="magma",
         vlims=[-30,0,-60,0], Tlim=[-50,50], Wlim=[1400,1700], zeros=True, plot_type="both")


#%% Plot 1

c = 299792458 * (1e9)/(1e12)

Lambda0= 1600
omega0 = 2*np.pi*c/Lambda0

lambda_znw_1650 = 1650
w_znw1650 = 2*np.pi*c/lambda_znw_1650
gam1_1650 = -fibra.gamma/(w_znw1650 - omega0)*1

lambda_znw_1450 = 1470
w_znw1450 = 2*np.pi*c/lambda_znw_1450
gam1_1450 = -fibra.gamma/(w_znw1450 - omega0)*1

def gamma_w(gamma, gamma1, w):
    return gamma + gamma1 * w

omega_vec = 2*np.pi*sim.freq
    
gamma1650_w = gamma_w(fibra.gamma, gam1_1650, omega_vec)
gamma1450_w = gamma_w(fibra.gamma, gam1_1450, omega_vec)


#Plots
fig, ax = plt.subplots()
ax.plot(omega_vec, gamma1650_w, label="ZNW = 1650 nm")
ax.plot(omega_vec, gamma1450_w, label="ZNW = 1450 nm")
ax.legend(loc="best")
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("$\\gamma(\\nu)$")
ax.set_xlim([-60,130])
ax.axvline(x= fibra.w_zdw-omega0, linestyle="--", color="grey")
ax.axvline(x= w_znw1450-omega0, linestyle="--", color="black")
ax.axvline(x= w_znw1650-omega0, linestyle="--", color="black")
ax.axhline(y=fibra.gamma)
plt.ylim([0,2*fibra.gamma])
plt.show()

#%% Plot en lambda

lam, gamma1650_l = Adapt_Vector(sim.freq, omega0, gamma1650_w)
lam, gamma1450_l = Adapt_Vector(sim.freq, omega0, gamma1450_w)

#Plots
fig, ax = plt.subplots()
ax.plot(lam, gamma1650_l*1e3, color="red", label="ZNW @ 1650 nm")
ax.plot(lam, gamma1450_l*1e3, color="blue", label="ZNW @ 1470 nm")
ax.axvline(x= fibra.zdw, linestyle=":", color="grey", label="ZDW @ 1555.5 nm")
ax.legend(loc="best")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("$\\gamma$ ($\mathrm{W}^{-1} \mathrm{km}^{-1}$)")
ax.set_xlim([1400,1700])
#ax.axhline(y=fibra.gamma)
plt.ylim([-1,2*fibra.gamma*1e3])
plt.tight_layout()
plt.show()

#%% Plot inset

from matplotlib import ticker
from matplotlib.colors import LogNorm

def format_func(value, tick_number):
    return f'$10^{{{int(value/10)}}}$' 

#Labels y tamaños
cbar_tick_size = 15
tick_size      = 15
m_label_size   = 15
M_label_size   = 15
vlims          = [-0,40]
cmap           = "magma"
Tlim           = [-40, 40]

#Vectores de potencia, y listas "extent" para pasarle a imshow
P_T = Pot(np.stack(AT))
textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]


#Escala dB

P_T = 20*np.log10(P_T)
    
#Limites del colorbar
vmin_t = vlims[0]
vmax_t = vlims[1]

#Ploteamos

fig, ax1 = plt.subplots(1,1,figsize=(6,5))

#fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8.76,5))

#---Plot en tiempo---
#Imshow 1
im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                 extent=textent, vmin=vmin_t, vmax=vmax_t)
ax1.tick_params(labelsize=tick_size)
ax1.set_ylabel("Distance (m)", size=m_label_size)
ax1.set_xlabel("Time (ps)", size=m_label_size)
ax1.set_xticks([-20,0,20])
ax1.set_yticks([0,150,300])
#Colorbar 1: Es interactivo
# =============================================================================
# cbar1 = fig.colorbar(im1, ax=ax1, label='$|A|^2$', location="bottom", aspect=30 )
# cbar1.set_ticks([-20,0, 20, 40, 60])
# cbar1.ax.tick_params(labelsize=cbar_tick_size)
# cbar1.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
# =============================================================================
ax1.set_xlim(Tlim)        
plt.tight_layout()
plt.show()
