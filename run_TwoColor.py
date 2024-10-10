# -*- coding: utf-8 -*-
""" Run TwoColor 
Created on Wed Oct  9 15:05:02 2024

@author: d/dt Lucas
"""

import numpy as np
from solvers.solvegnlse import SolveNLS, Raman
from solvers.solvepcnlse import Solve_pcNLSE
from solvers.solvepcGNLSE import Solve_pcGNLSE
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
L     = 1000                         #Lfib:   m
b2    = 1e-3                  #Beta2:  ps^2/km
b3    = 0               #Beta3:  ps^3/km
b4    = -1*1e-6
betas = [b2,b3,b4]
gam   = 1.4e-3                  #Gamma:  1/Wkm
gam1 = 0
alph  = 0                        #alpha:  dB/m
TR    = 0                   #TR:     fs
fR    = 0                   #fR:     adimensional (0.18)

#Cargo objetos con los parámetros:
sim    = Sim(N, Tmax)
fibra  = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, betas=betas, alpha=alph, lambda0=Lambda0, TR = TR, fR = fR)


#%%Parámetros de la molécula

def TwoColor(fib:Fibra, sim:Sim, t0):
    F = np.sqrt(8*fib.betas[0]/(3*fib.gamma*t0**2)) * 1/np.cosh(sim.tiempo/t0)
    A = F * np.cos( np.sqrt(6*fib.betas[0]/np.abs(fib.betas[2]))*sim.tiempo  )
    return A

#Frecuencias de la molécula
f1 = -np.sqrt(6*b2/np.abs(b4))
f2 = -f1

t0 = 100e-3

molecule = TwoColor(fibra, sim, t0)
molecule_spectrum = FT(molecule)

z0 = np.pi/2 * t0**2/fibra.betas[0]

#%% Parámetros pulso dispersivo

amp    = 80
ancho  = 1
offset = 5
lam_d    = 1540
freq   = fibra.lambda_to_omega(lam_d)

d_pulse = np.sqrt(amp)*(1/np.cosh(sim.tiempo + offset/ancho))*np.exp(-1j*freq*sim.tiempo)

m_d = molecule + d_pulse

#%% Corriendo el código

#---NLSE molecule---
time_0 = time.time()
zlocs, AW, AT = SolveNLS(sim, fibra, m_d, z_locs=300, raman=False, pbar=True)
time_1 = time.time()

total_n = time_1 - time_0 #Implementar en Solve_pcGNLSE
print("Fiber 1 DONE. Time",np.round(total_n/60,2),"(min)")
chime.success()

#%% Plots

plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=True, Tlim=[-20,20],
          vlims=[-30,0,-30,0], zeros=False)

#%% Save

saver(AW, AT, sim, fibra, "twocolor/1540nm", f'{[lam_d, amp, ancho, offset, t0] = }')

#%% EXTRAS

#Plot profile

beta_w = b2/2 * (2*np.pi*sim.freq)**2 + b3/6 * (2*np.pi*sim.freq)**3 + b4/24 * (2*np.pi*sim.freq)**4

lam, beta_l = Adapt_Vector(sim.freq, fibra.omega0, beta_w)
lam, mol_l  = Adapt_Vector(sim.freq, fibra.omega0, molecule_spectrum)

plt.figure()
plt.plot(lam, beta_l/np.max(beta_l), label="$\\beta(\lambda)$")
plt.plot(lam, Pot(mol_l)/np.max(Pot(mol_l)))
plt.grid(True,alpha=.3)
plt.xlim([1400, 1750])
plt.ylim([-0.02,1.4])
plt.xlabel("Wavelength (nm)")
plt.ylabel("$\\beta(\lambda)$ (m$^{-1}$)")
plt.legend(loc="best")
plt.show()


