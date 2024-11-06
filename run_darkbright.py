# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:30:58 2024

@author: gutie
"""

import numpy as np
from solvers.solvedark import Solve_dark
from common.commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from common.commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, Adapt_Vector
from common.commonfunc import find_k
from common.plotter import plotinst, plotcmap, plotspecgram, plotenergia, plotfotones, plt, plot_time
from scipy.signal import find_peaks

#Time
import time

#Boludez
import chime
chime.theme('mario')

import cmasher as cmr

#%%Parámetros de fibra y simulación

#Parametros para la simulación
N = int(2**14) #puntos
Tmax = 70      #ps

c = 299792458 * (1e9)/(1e12)

Lambda0= 1555                    #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0

#Parametros para la fibra
L     = 10                         #Lfib:   m
b2    = -1e-3                  #Beta2:  ps^2/km
b3    = 0               #Beta3:  ps^3/km
b4    = 1*1e-6
betas = [b2,b3,b4]
gam   = 1.4e-3                  #Gamma:  1/Wkm
gam1 = 0
alph  = 0                        #alpha:  dB/m
TR    = 0                   #TR:     fs
fR    = 0                   #fR:     adimensional (0.18)

#Frecuencias con GVD matching.
f1 = -np.sqrt(6*np.abs(b2)/np.abs(b4))
f2 = -f1

#Cargo objetos con los parámetros:
sim    = Sim(N, Tmax)
fibra  = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, betas=betas, alpha=alph, lambda0=Lambda0, TR = TR, fR = fR)


#%%Parámetros del dark soliton

def simple_ds(time, beta2, gamma, width, maxwindow):
    u = np.sqrt(np.abs(beta2)/(gamma * width**2))
    squarewell_factor = maxwindow/width - 10
    return u * np.tanh(time/width) #* SuperGauss(time,1,width*squarewell_factor,0,0,50)

t0 = 0.09#500e-3

darksol = simple_ds(sim.tiempo, fibra.beta2_w(f1), fibra.gamma_w(f1), width=t0, maxwindow=Tmax)#*np.exp(-1j*f1*sim.tiempo)


#%% Parámetros pulso dispersivo

amp    = 2
ancho  = 0.5
offset = 1.5
lam_d    = 1540
#freq   = fibra.lambda_to_omega(lam_d)
f_d    = f2+1.5

d_pulse = np.sqrt(amp)*(1/np.cosh( (sim.tiempo + offset)/ancho))*np.exp(-1j*f_d*sim.tiempo)

soliton = Soliton(sim.tiempo, t0, fibra.beta2, fibra.gamma, 1)#*np.exp(-1j*f1*sim.tiempo)

m_d = darksol #+ d_pulse#soliton + d_pulse

#m_d = molecule + d_pulse 

#%% Corriendo el código

#---NLSE molecule---
time_0 = time.time()
zlocs, AW, AT = Solve_dark(sim, fibra, m_d, z_locs=300, window = 500, pbar=True)
time_1 = time.time()

total_n = time_1 - time_0
print("Fiber 1 DONE. Time",np.round(total_n/60,2),"(min)")
chime.success()

#%% Plots

plotinst(sim, fibra, AT, AW)

plotcmap(sim, fibra, zlocs, AT, AW, dB=True, cmap="RdBu", vlims=[-3.5,-0.9,-20,-80], Tlim=[-3,3], Wlim=[-20,20])

plotcmap(sim, fibra, zlocs, AT, AW, dB=False, cmap="RdBu", vlims=[140,175, 10000,50000], Tlim=[-3,3], Wlim=[11,16])
