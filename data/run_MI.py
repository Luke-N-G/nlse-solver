# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:32:03 2023

@author: Luquitas
"""


import numpy as np
from solvegnlse import SolveNLS, Raman
from commonfunc import SuperGauss, Soliton, Sim, Fibra, Pot, fftshift, FT, IFT
from plotter import plotinst, plotevol, plotenergia, plotfotones, plt
from scipy.signal import find_peaks


#Parametros para la simulación
ventana = 200 #ps
N_r = int(2**14)
dt_r = ventana/N_r #1e-3

c = 299792458 * (1e9)/(1e12)

lambda_r= 1550                      #Longitud de onda central (nm)
omega0_r = 2*np.pi*c/lambda_r

#Parametros para la fibra
pasos_r = 1
L_r     = .1e3                      #Lfib:   m
b2_r    = -1  / 1e3                 #Beta2:  ps^2/km
b3_r    = 0   / 1e3                 #Beta3:  ps^3/km
gam_r   = .1                        #Gamma:  1/Wkm
gam1_r  = gam_r/omega0_r*0
alph_r  = 0                         #alpha:  dB/*m
TR_r    = 3e-3*0                    #TR:     fs
fR_r    = 0.18                      #fR:     adimensional (0.18)

#Parámetros pulso gaussiano:
amp_r    = 1                        #Amplitud:  sqrt(W), Po = amp**2
ancho_r  = .2                       #Ancho T0:  ps
offset_r = 0
chirp_r  = 0
orden_r  = 1

#Cargo objetos con los parámetros:
sim_r   = Sim(N_r,dt_r)
fibra_r = Fibra(pasos_r, L_r, b2_r, b3_r, gam_r, gam1_r, alph_r, TR_r, fR_r)

#Calculamos el pulso inicial
pulso_r = SuperGauss(sim_r.tiempo, amp_r, ancho_r, offset_r, chirp_r, orden_r)

soliton_r = Soliton(sim_r.tiempo, ancho_r, fibra_r.beta2, fibra_r.gamma, orden = 1)

CW_r = amp_r*np.random.normal(1,.0001,len(sim_r.tiempo))

#%% Corriendo la simulación

zlocs_r, A_wr, A_tr = SolveNLS(sim_r, fibra_r, CW_r, raman=False, z_locs=120)


#%% Plotting

plotinst(sim_r, A_tr, A_wr, Tlim=[-2,2] ,Wlim=[-70,70], dB=True)

plotevol(sim_r, zlocs_r, A_tr, A_wr)

