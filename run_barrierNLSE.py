# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:00:57 2023

@author: d/dt Lucas
"""


import numpy as np
from solvers.solvebarrierNLSE import Solve_barrierNLSE
from common.commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from common.commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, Adapt_Vector
from common.commonfunc import find_k
from common.plotter import plotinst, plotevol, plotcmap, plotspecgram, plotenergia, plotfotones, plt, plot_time
from scipy.signal import find_peaks

#Time
import time

#Boludez
import chime
chime.theme('mario')

import cmasher as cmr


#Parametros para la simulación
N_r = int(2**14)
Tmax_r = 70 #ps

c = 299792458 * (1e9)/(1e12)

Lambda0= 1498                 #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0

#Parámetros pulso 1:
Lambda1  = Lambda0
amp_1    = 35               #Amplitud:  sqrt(W), Po = amp**2
ancho_1  = 2                #Ancho T0:  ps
offset_1 = 0

#Parametros para la fibra
L_r     = 100*0.1                        #Lfib:   m
db1_r   = 0.29761970102977475#0.1                         #Beta1:  ps/m
b2_r    = 0.019925777373085#5e-3*1                     #Beta2:  ps^2/km
b3_r    = 0.0006066055875799886#0.13e-3*0                 #Beta3:  ps^3/km
b4_r    = 7.563380967967562e-06
bb_r    = 5                          #Betab:  1/m
TB_r    = 10#5                       #HeavT:  ps
betas_r = [b2_r,b3_r,b4_r]

gam_r  = 2.5e-3*1
gam1_r = 0

alph_r  = 0                        #alpha:  dB/m
TR_r    = 3e-3*0                   #TR:     fs
fR_r    = 0.18*0                   #fR:     adimensional (0.18)

#Cargo objetos con los parámetros:
sim_r   = Sim(N_r, Tmax_r)
fibra_r = Fibra(L=L_r, beta2=b2_r, beta3=b3_r, gamma=gam_r, gamma1=gam1_r, alpha=alph_r, lambda0=Lambda1, TR = TR_r, fR = fR_r, betas=betas_r)

#Pulso a simular
pulso_r = SuperGauss(sim_r.tiempo, amp_1, ancho_1, 0, 0, 1)

#%% Corriendo la simulación

#---pcgNLSE---
t0 = time.time()
zlocs_r, A_wr, A_tr, ysol = Solve_barrierNLSE(sim_r, fibra_r, pulso_r, db1_r, TB_r, bb_r,  z_locs=300)
t1 = time.time()
    
total_n = t1 - t0 #Implementar en Solve_pcGNLSE
print("Time",np.round(total_n/60,2),"(min)")
chime.success()
    

#%% Graficando

plotinst(sim_r, fibra_r, A_tr, A_wr,dB=False , wavelength=True, zeros=False, end=-1)

plotspecgram(sim_r, fibra_r, A_tr[-1], zeros=True)

plotcmap(sim_r, fibra_r, zlocs_r, A_tr, A_wr, wavelength=True, dB=True, Tlim=[-30,30], Wlim=[1400,1700])


#%% Beta_dispersive calculator

def betaw_f(sim:Sim, fib:Fibra, deltab1, wavelength=False):
    beta  = deltab1 * (2*np.pi*sim.freq) + fib.beta2/2 * (2*np.pi*sim.freq)**2 + fib.beta3/6 * (2*np.pi*sim.freq)**3
    beta1 = fib.beta2 * (2*np.pi*sim.freq) + fib.beta3/2 * (2*np.pi*sim.freq)**2
    beta2 = fib.beta2 + fib.beta3 * (2*np.pi*sim.freq)
    if wavelength:
        lam, beta = Adapt_Vector(sim.freq, fib.omega0, beta)
        lam, beta1 = Adapt_Vector(sim.freq, fib.omega0, beta1)
        lam, beta2 = Adapt_Vector(sim.freq, fib.omega0, beta2)
        return lam, [beta,beta1,beta2]
    else:
        return 2*np.pi*sim.freq,[beta,beta1,beta2]


def betas_disp(lambda_dispersive, lambda_zdw):
    
    #Parámetros de betas para el solitón originalmente centrado en 1600
    c = 299792458 * (1e9)/(1e12)
    lambda0 = 1600               #Longitud de onda central del solitón en 1600
    omega0  = 2*np.pi*c/lambda0
    b2 = -4.4e-3                 #Beta 2 en 1600
    #lambda_zdw = 1555            #ZDW
    w_zdw = 2*np.pi*c/lambda_zdw #Frecuencia de ZDW
    b3 = -b2/(w_zdw - omega0)*1  #Beta 3 que asegura la ZDW
    
    #En base a los datos anteriores, hallamos los valores de los betas para el dispersivo centrado en lambda_d
    omega_d = 2*np.pi*c/lambda_dispersive - omega0
    b1_d = b2 * omega_d + b3/2 *omega_d**2
    b2_d = b2 + b3*omega_d
    b3_d = b3
    
    return [b1_d,b2_d,b3_d]

lambda_disp = 1475
lambda_zdw  = 1555.3

betas_test = betas_disp(lambda_disp, lambda_zdw)

fibra_test2 = Fibra(300, betas_test[1], betas_test[2], gam_r*0, gam1_r*0, alph_r*0, lambda_disp)

lam_disp_t, b_disp_t = betaw_f(sim_r, fibra_test2, betas_test[0], wavelength=True) 
plt.figure()
plt.plot(lam_disp_t, b_disp_t[0], label="$\\beta(\lambda)$")
plt.plot(lam_disp_t, b_disp_t[0]+bb_r, label="$\\beta_B(\lambda)$")
plt.axvline(lambda_zdw, color="darkgrey", linestyle=":")
plt.grid(True,alpha=.3)
plt.xlim([1460, 1655])
plt.ylim([-4,2])
plt.xlabel("Wavelength (nm)")
plt.ylabel("$\\beta(\lambda)$ (m$^{-1}$)")
plt.legend(loc="best")
plt.show()

#%% Caso cuartico n

def betas_disp4(lambda_dispersive, lambda_zdw1, lambda_zdw2):
    
    #Parámetros de betas para el solitón originalmente centrado en 1600
    c = 299792458 * (1e9)/(1e12)
    lambda0 = 1600               #Longitud de onda central del solitón en 1600
    omega0  = 2*np.pi*c/lambda0
    b2 = -4.4e-3                 #Beta 2 en 1600
    #lambda_zdw = 1555            #ZDW
    w_zdw = 2*np.pi*c/lambda_zdw #Frecuencia de ZDW
    b3 = -b2/(w_zdw - omega0)*1  #Beta 3 que asegura la ZDW
    
    lambda0 = 1600
    omega0 = 2*np.pi*c/lambda0
    b2 = -4.4e-3
    w_zdw1 = 2*np.pi*c/lambda_zdw1
    dw_zdw1 = (w_zdw1 - omega0)
    w_zdw2 = 2*np.pi*c/lambda_zdw2
    dw_zdw2 = (w_zdw2 - omega0)
    b4 = 2*b2 * (dw_zdw2/dw_zdw1 - 1) / (-dw_zdw1*dw_zdw2 + dw_zdw2**2)
    b3 = -b2/dw_zdw1 - b4/2 * dw_zdw1
    
    #En base a los datos anteriores, hallamos los valores de los betas para el dispersivo centrado en lambda_d
    omega_d = 2*np.pi*c/lambda_dispersive - omega0
    b1_d = b2 * omega_d + b3/2 *omega_d**2 + b4/6*omega_d**3
    b2_d = b2 + b3*omega_d + b4/2*omega_d**2
    b3_d = b3 + b4*omega_d
    b4_d = b4
    
    return [b1_d,b2_d,b3_d,b4_d]

    
def betaw4_f(sim:Sim, fib:Fibra, deltab1, wavelength=False):
    beta  = deltab1 * (2*np.pi*sim.freq) + fib.betas[0]/2 * (2*np.pi*sim.freq)**2 + fib.betas[1]/6 * (2*np.pi*sim.freq)**3 + fib.betas[2]/24 * (2*np.pi*sim.freq)**4
    beta1 = fib.betas[0] * (2*np.pi*sim.freq) + fib.betas[1]/2 * (2*np.pi*sim.freq)**2 + fib.betas[2]/6 * (2*np.pi*sim.freq)**3
    beta2 = fib.betas[0] + fib.betas[1] * (2*np.pi*sim.freq) + fib.betas[2]/2 * (2*np.pi*sim.freq)**3
    if wavelength:
        lam, beta = Adapt_Vector(sim.freq, fib.omega0, beta)
        lam, beta1 = Adapt_Vector(sim.freq, fib.omega0, beta1)
        lam, beta2 = Adapt_Vector(sim.freq, fib.omega0, beta2)
        return lam, [beta,beta1,beta2]
    else:
        return 2*np.pi*sim.freq, [beta,beta1,beta2]
    
lambda_disp  = 1498
lambda_zdw1  = 1555
lambda_zdw2  = 1647.8

betas_t2 = betas_disp4(lambda_disp, lambda_zdw1, lambda_zdw2)

fibra_t2 = Fibra(300, betas_test[1], betas_test[2], gam_r*0, gam1_r*0, alph_r*0, lambda_disp, betas=betas_t2[1:])

lam_disp_t2, b_disp_t2 = betaw4_f(sim_r, fibra_t2, betas_t2[0], wavelength=True) 
plt.figure()
plt.plot(lam_disp_t2, b_disp_t2[0], label="$\\beta(\lambda)$")
plt.plot(lam_disp_t2, b_disp_t2[0]+bb_r, label="$\\beta_B(\lambda)$")
plt.axvline(lambda_zdw1, color="darkgrey", linestyle=":")
plt.axvline(lambda_zdw2, color="darkgrey", linestyle=":")
plt.grid(True,alpha=.3)
plt.xlim([1450, 1750])
plt.ylim([-4,8])
plt.xlabel("Wavelength (nm)")
plt.ylabel("$\\beta(\lambda)$ (m$^{-1}$)")
plt.legend(loc="best")
plt.show()

