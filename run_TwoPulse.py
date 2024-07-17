# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:00:57 2023

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


#Parametros para la simulación
N = int(2**14) #puntos
Tmax = 70      #ps

c = 299792458 * (1e9)/(1e12)

Lambda0= 1600                    #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0


#Parámetros pulso 1:
Lambda1  = Lambda0
amp_1    = 243                   #Amplitud:  sqrt(W), Po = amp**2
ancho_1  = 85e-3                 #Ancho T0:  ps
offset_1 = 0


#Parámetros pulso 2:
Lambda2  = 1500#1480
amp_2    = 10#30*1
ancho_2  = 2100e-3
offset_2 = 20 


#Parametros para la fibra
L     = 500                   #Lfib:   m
b2    = -4.4e-3*1                  #Beta2:  ps^2/km
b3    = 0.13e-3*1                  #Beta3:  ps^3/km
gam   = 2.5e-3*1                   #Gamma:  1/Wkm

lambda_znw = 1650
w_znw = 2*np.pi*c/lambda_znw
gam1 = -gam/(w_znw - omega0)*1

alph  = 0                        #alpha:  dB/m
TR    = 3e-3*0                   #TR:     fs
fR    = 0.18*0                   #fR:     adimensional (0.18)

#Diferencia de frecuencias
nu2     = c/Lambda2
nu1     = c/Lambda1
dnu2    = nu2 - nu1

#Cargo objetos con los parámetros:
sim   = Sim(N, Tmax)
fibra = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, alpha=alph, lambda0=Lambda1, TR = TR, fR = fR)


#Calculamos el pulso inicial
pulso = Two_Pulse(sim.tiempo, amp_1, amp_2, ancho_1, ancho_2, offset_1, offset_2, dnu2, pulses = "sp")



#%% Corriendo la simulación

#---pcgNLSE---
t0 = time.time()
zlocs, AW, AT = Solve_pcGNLSE(sim, fibra, pulso, z_locs=300)
t1 = time.time()

total_n = t1 - t0 #Implementar en Solve_pcGNLSE
print("Time",np.round(total_n/60,2),"(min)")
chime.success()

#saver(AW, AT, sim, fibra, "save_test", f'{[Lambda1, amp_1, ancho_1, offset_1, Lambda2, amp_2, ancho_2, offset_2] = }')


#%%


plotinst(sim, fibra, AT, AW, dB=False , wavelength=True, zeros=True, end=-1)

plotinst(sim, fibra, AT, AW, dB=False, wavelength=True, zeros=True, end=210)

plotspecgram(sim, fibra, AT, zeros=True)

plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=True, Tlim=[-30,30], Wlim=[1400,1700],
          vlims=[-20,50,0,120], zeros=True)


#%% Extra


#Para guardar en loop con i el parametro variado
#j = str(i).replace(".", "," )
# =============================================================================
#     
#     plotcmap(sim, fibra, zlocs, AT, AW, Tlim=[-50,50], dB=True, wavelength=True, save=savedic+str(j), zeros=True, noshow=True)
#     plotspecgram(sim, fibra, AT[-1], save=savedic+str(j)+"_specgram", dB=False, zeros=True, noshow=True)
#     
# ============================================================================
#------- NO OLVIDARSE DE CAMBIAR LOS OTROS PARAMETROS! -------
#saver(AW,AT,sim,fibra,savedic+str(j), f'{[Lambda1, amp_1, ancho_1, offset_1, Lambda2, amp_2, ancho_2, offset_2] = }')

#Po = 10W, To = 2.1ps, lambda_i = 1500 nm
# =============================================================================
# reflect_wave_t = [1525.95643739, 1528.68813412, 1530.99614865, 1535.59401619,
#        1537.67139725, 1538.91257145, 1538.91257145, 1540.35459015,
#        1540.97034682, 1541.38002821, 1541.38002821, 1542.40112176,
#        1542.60477423, 1543.41735052]
# 
# z_s = [267.2,280.6,294.8,309.8,322.5,338.2,352.4,368.1,390.5,
#        408.4,433.1,456.3,472.0,492.2]
# =============================================================================

reflect_wave_t = [1527.63785688, 1531.41528825, 1535.38575916, 1542.19727487,
       1546.03250148, 1548.01155099, 1551.66612959, 1555.30091614,
       1559.21611957, 1561.90664954]

z_s = [233,239,242.5,248.9,256.2,268.2,277.2,289.2,302.9,315.8]

plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=False, Tlim=[-30,30], Wlim=[1400,1700],zeros=True)
          #vlims=[-20,50,0,120], zeros=True)
plt.plot(reflect_wave_t,z_s, color="white")
