# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:43:33 2023

@author: d/dt Lucas
"""

import numpy as np
from solvegnlse import SolveNLS, Raman
from solvepcnlse import Solve_pcNLSE
from commonfunc import SuperGauss, Soliton, Sim, Fibra, Pot, fftshift, FT, IFT, find_shift, find_chirp
from plotter import plotinst, plotevol, plotenergia, plotfotones, plt
from scipy.signal import find_peaks


#Parametros para la simulación
ventana = 200 #ps
N_r = int(2**14)
dt_r = ventana/N_r #1e-3
Tmax_r = 200 #ps

c = 299792458 * (1e9)/(1e12)

lambda_r= 1550                       #Longitud de onda central (nm)
omega0_r = 2*np.pi*c/lambda_r

#Parametros para la fibra
pasos_r = 1
L_r     = 2e3                       #Lfib:   m
b2_r    = -20  / 1e3  *0               #Beta2:  ps^2/km
b3_r    = 0   / 1e3                  #Beta3:  ps^3/km
gam_r   = .02#.02                        #Gamma:  1/Wkm
gam1_r  = gam_r/omega0_r
alph_r  = 0                          #alpha:  dB/*m

#Parámetros pulso gaussiano:
amp_r    = 1                        #Amplitud:  sqrt(W), Po = amp**2
ancho_r  = 1                        #Ancho T0:  ps
offset_r = 0
chirp_r  = 0
orden_r  = 1

zshock = .39*ancho_r/(np.abs(gam1_r)*amp_r**2)
zshock
#Cargo objetos con los parámetros:
#sim_r   = Sim(N_r,dt_r)
sim_r   = Sim(N_r, Tmax_r)
fibra_r = Fibra(pasos_r, zshock, b2_r, b3_r, gam_r, gam1_r, alph_r, lambda0=lambda_r)

#Calculamos el pulso inicial
pulso_r = SuperGauss(sim_r.tiempo, amp_r, ancho_r, offset_r, chirp_r, orden_r)

#soliton_r = Soliton(sim_r.tiempo, ancho_r, fibra_r.beta2, fibra_r.gamma, orden = 1)


#%% Corriendo la simulación

#gNLSE
zlocs_r, A_wr, A_tr = SolveNLS(sim_r, fibra_r, pulso_r, raman=True, z_locs=120)

#pcNLSE
zlocs_pc, A_wpc, A_tpc = Solve_pcNLSE(sim_r, fibra_r, pulso_r, z_locs=120)


#%% Plotting
plotinst(sim_r, A_tr, A_wr, Tlim=[-.25,.25] ,Wlim=[-50,50])

plotinst(sim_r, A_tpc, A_wpc, Tlim=[-.25,.25] ,Wlim=[-50,50])

#plotevol(sim_r, zlocs_r, A_tr, A_wr, Tlim=[-2,2], Wlim=[-10,10])

#plotenergia(sim_r, zlocs_r, A_tr, A_wr)
#plotfotones(sim_r, zlocs_r, A_wr, lambda0 = lambda_r)


#%% Extra

