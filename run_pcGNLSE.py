# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:43:33 2023

@author: d/dt Lucas
"""

import numpy as np
from solvers.solvegnlse import SolveNLS, Raman
from solvers.solvepcnlse import Solve_pcNLSE
from solvers.solvepcGNLSE import Solve_pcGNLSE
from common.commonfunc import SuperGauss, Soliton, Sim, Fibra, Pot, fftshift, FT, IFT, find_shift, find_chirp
from common.plotter import plotinst, plotevol, plotcmap, plotenergia, plotfotones, plt
from scipy.signal import find_peaks

#Parametros para la simulación
N = int(2**14)  #puntos
Tmax = 70       #ps

c = 299792458 * (1e9)/(1e12)   #Vel. luz en nm/ps

lambda0 = 1550                       #Longitud de onda central (nm)
omega0  = 2*np.pi*c/lambda0

#Parametros para la fibra
L     = 12.5*5                       #Lfib:   m
b2    = -20e-3*0                    #Beta2:  ps^2/m
b3    = 10e-3                         #Beta3:  ps^3/m
gam   = 0#1.4e-3                    #Gamma:  1/Wm
gam1  = gam/omega0*0              #Gamma1: 0
alph  = 0                         #alpha:  1/m
TR    = 3e-3*0                    #TR:     fs
fR    = 0.18*0                    #fR:     adimensional (0.18)

#Parámetros pulso gaussiano:
amp    = 1                 #Amplitud:  sqrt(W), Po = amp**2
ancho  = .4                  #Ancho T0:  ps
offset = 0                  #Offset:    ps
chirp  = 0                  #Chirp:     1/m
orden  = 1                  #Orden

#Cargo objetos con los parámetros:
sim   = Sim(N, Tmax)
fibra = Fibra(L, b2, b3, gam, gam1, alph, lambda0, TR, fR)

#Calculamos el pulso inicial

#Supergaussiano
pulso = SuperGauss(sim.tiempo, amp, ancho, offset, chirp, orden)

#Solitón
soliton = Soliton(sim.tiempo, ancho, fibra.beta2, fibra.gamma, orden = 1)


#%% Corriendo la simulación

zlocs, AW, AT = Solve_pcGNLSE(sim, fibra, pulso, z_locs=100)


#%% Plotting

plotinst(sim, fibra, AT, AW, Wlim=[-1,1], Tlim=[-7,7])
plotcmap(sim, fibra, zlocs, AT, AW, Wlim=[-1,1], Tlim=[-7,7])


#%% Extra

plt.plot(sim.tiempo/ancho, Pot(AT)[-1]/np.max(Pot(AT)[-1]))
plt.xlim([-6,16])
plt.ylim([0,1.1])