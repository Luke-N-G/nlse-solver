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

#%% Funciones

#Función Molécula
def TwoColor(fib:Fibra, sim:Sim, t0):
    F = np.sqrt(8*fib.betas[0]/(3*fib.gamma*t0**2)) * 1/np.cosh(sim.tiempo/t0)
    A = F * np.cos( np.sqrt(6*fib.betas[0]/np.abs(fib.betas[2]))*sim.tiempo  )
    return A

def meta(T, t0, order, fib:Fibra):
    f1 = np.sqrt(6*fibra.betas[0]/np.abs(fibra.betas[2]))
    f2 = -f1    
    nu = -1/2 + (1/4 + 4 * np.abs( (fibra.gamma_w(f2) * fibra.beta2_w(f1))/(fibra.gamma_w(f1)*fibra.beta2_w(f2)) )  )**(1/2)
    print(nu)
    if order == 0:
        phi = (1/np.cosh(T/t0))**nu
    if order == 1:
        phi = (1/np.cosh(T/t0))**(nu-1) * np.tanh(T/t0)
    return phi

def meta_spectrum(AW, sim, zlocs):
    #Convertimos todas las frecuencias negativas a cero
    AW_cut = np.zeros_like(AW)
    for i in range(len(zlocs)):
        AW_cut[i] = AW[i] * [sim.freq<-15] * [sim.freq>-20]
    AT_cut = IFT(AW_cut)
    return AT_cut, AW_cut


#%%Parámetros de fibra y simulación

#Parametros para la simulación
N = int(2**14) #puntos
Tmax = 2      #ps

c = 299792458 * (1e9)/(1e12)

Lambda0= 1555                    #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0

#Parametros para la fibra
L     = 1000                        
b2    = 1                 
b3    = 0               
b4    = -1*1e-6
betas = [b2,b3,b4]
gam   = 1e6 #1.4e-3
                 
#lambda_znw = 1650
w_znw = -420*2*np.pi#2*np.pi*c/lambda_znw
gam1 = -gam/w_znw#-gam/(w_znw - omega0)*1

alph  = 0                      
TR    = 0                  
fR    = 0.18  *0                


#Parámetros Molécula
t0 = 8e-3
f1 = -np.sqrt(6*np.abs(b2)/np.abs(b4))
f2 = -f1


#Cargo objetos con los parámetros:
sim    = Sim(N, Tmax)
fibra  = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, betas=betas, alpha=alph, lambda0=Lambda0, TR = TR, fR = fR)
z0 = np.abs( np.pi/2 * t0**2/fibra.beta2_w(f1) )
L = 50*z0
fibra  = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, betas=betas, alpha=alph, lambda0=Lambda0, TR = TR, fR = fR)


#%% Armando los pulsos

soliton = Soliton(sim.tiempo, t0, fibra.beta2_w(f1), fibra.gamma_w(f1), orden=1)*np.exp(-1j*f1*sim.tiempo)

#soliton = Soliton(sim.tiempo, t0, fibra.beta2, fibra.gamma, orden=1)

meta_0 = np.sqrt(1e-7) * np.max(soliton) * meta(sim.tiempo, t0, 0, fibra)*np.exp(-1j*f2*sim.tiempo)

meta_1 = np.sqrt(1e-7) * np.max(soliton) * meta(sim.tiempo, t0, 1, fibra)*np.exp(-1j*f2*sim.tiempo)

meta_s = np.sqrt(1e-7) * np.max(soliton) * (meta(sim.tiempo, t0, 0, fibra) + 5*meta(sim.tiempo, t0, 1, fibra)) *np.exp(-1j*f2*sim.tiempo)

molecule = TwoColor(fibra, sim, t0)

CW = 1/5000*np.ones(len(soliton))*np.exp(-1j*(f1+2*np.pi*13.2)*sim.tiempo)

pulse = molecule#soliton+meta_0*0#molecule


#%% Corriendo el código

#---NLSE molecule---
time_0 = time.time()
#zlocs, AW, AT = SolveNLS(sim, fibra, pulse, z_locs=300, raman=True, pbar=True)
zlocs, AW, AT = Solve_pcGNLSE(sim, fibra, pulse, z_locs=300, pbar=True)
time_1 = time.time()

total_n = time_1 - time_0 #Implementar en Solve_pcGNLSE
print("Fiber 1 DONE. Time",np.round(total_n/60,2),"(min)")
chime.success()

AT_cut, AW_cut = meta_spectrum(AW, sim, zlocs)

 #%% Plots

plotcmap(sim, fibra, zlocs/z0, AT, AW, dB=True, vlims=[-40,0,-90,0], Tlim=[-1,2], Wlim=[-550,550])

plotcmap(sim, fibra, zlocs/z0, AT, AW, dB=False, Tlim=[-1,2], Wlim=[-550,550])

#plotcmap(sim, fibra, zlocs/z0, AT_cut, AW_cut, dB=True, vlims=[-20,0,-30,0], Tlim=[-1,2], Wlim=[200,600])

#%% Saving

dic  = "molecule_tests/"
file = "phi0"

#Saving data
saver(AW, AT, sim, fibra, dic+file)


#%%
'''
----------------------------------------------------------------------------------------------------------
                                         VARIACION DE PARAMETROS 
----------------------------------------------------------------------------------------------------------
'''

#Parametros para la simulación
N = int(2**14) #puntos
Tmax = 70      #ps

c = 299792458 * (1e9)/(1e12)

Lambda0= 1555                    #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0

#Parametros para la fibra
L     = 1000                        
b2    = 20e-3                 
b3    = 0               
b4    = -1e-5
betas = [b2,b3,b4]
gam   = 10e-3                 
gam1  = 0
alph  = 0                      
TR    = 0                  
fR    = 0.18 * 1                 


#Parámetros Molécula
t0 = 500e-3#100e-3
f1 = -np.sqrt(6*np.abs(b2)/np.abs(b4))
f2 = -f1


#Cargo objetos con los parámetros:
sim    = Sim(N, Tmax)
fibra  = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, betas=betas, alpha=alph, lambda0=Lambda0, TR = TR, fR = fR)
z0 = np.abs( np.pi/2 * t0**2/fibra.beta2_w(f1) )
L = 50*z0
fibra  = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, betas=betas, alpha=alph, lambda0=Lambda0, TR = TR, fR = fR)


#%% Armando los pulsos

soliton = Soliton(sim.tiempo, t0, fibra.beta2_w(f2), fibra.gamma, orden=1)*np.exp(-1j*f2*sim.tiempo)

meta_0 = np.sqrt(1e-7) * np.max(soliton) * meta(sim.tiempo, t0, 0, fibra)*np.exp(-1j*f1*sim.tiempo)

meta_1 = np.sqrt(1e-7) * np.max(soliton) * meta(sim.tiempo, t0, 1, fibra)*np.exp(-1j*f1*sim.tiempo)

meta_s = np.sqrt(1e-7) * np.max(soliton) * (meta(sim.tiempo, t0, 0, fibra) + 5*meta(sim.tiempo, t0, 1, fibra)) *np.exp(-1j*f1*sim.tiempo)

molecule = TwoColor(fibra, sim, t0)

CW = 0.05*np.ones(len(soliton))*np.exp(-1j*(f1+2*np.pi*13.2)*sim.tiempo)

pulse = soliton*3 + meta_1 #+ CW


#%% Corriendo el código

#---NLSE molecule---
time_0 = time.time()
zlocs, AW, AT = SolveNLS(sim, fibra, pulse, z_locs=300, raman=True, pbar=True)
time_1 = time.time()

total_n = time_1 - time_0 #Implementar en Solve_pcGNLSE
print("Fiber 1 DONE. Time",np.round(total_n/60,2),"(min)")
chime.success()

AT_cut, AW_cut = meta_spectrum(AW, sim, zlocs)

#%% Plots

plotcmap(sim, fibra, zlocs/z0, AT, AW, dB=True, vlims=[-40,0,-90,0], Tlim=[-4,4], Wlim=[-30,30])

plotcmap(sim, fibra, zlocs/z0, AT_cut, AW_cut, dB=True, vlims=[-20,0,-30,0], Tlim=[-4,4], Wlim=[-30,-10], cmap=cmr.ember)
plotcmap(sim, fibra, zlocs/z0, AT_cut, AW_cut, dB=True, vlims=[-20,0,-30,0], Tlim=[-4,4], Wlim=[-30,-10])


