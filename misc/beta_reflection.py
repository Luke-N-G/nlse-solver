# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:32:16 2024

@author: d/dt Lucas
"""

import numpy as np
from solvers.solvegnlse import SolveNLS, Raman
from solvers.solvepcnlse import Solve_pcNLSE
from solvers.solvepcGNLSE import Solve_pcGNLSE
from common.commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from common.commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, Adapt_Vector
from common.commonfunc import find_k
from common.plotter import plotinst, plotevol, plotspecgram, plotenergia, plotfotones, plt, plot_time
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

Lambda0= 1600                       #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0


#Parámetros pulso 1:
Lambda1  = Lambda0
amp_1    = 243                     #Amplitud:  sqrt(W), Po = amp**2
ancho_1  = 85e-3                 #Ancho T0:  ps
offset_1 = 0


#Parámetros pulso 2:
Lambda2  = 1500#1480
amp_2    = 10#30*1
ancho_2  = 300e-3#2100e-3
offset_2 = 20 


#Parametros para la fibra
L_r     = 500*0.0001#300                        #Lfib:   m
b2_r    = -4.4e-3*1                  #Beta2:  ps^2/km
b3_r    = 0.13e-3*1                  #Beta3:  ps^3/km
gam_r   = 2.5e-3*1                   #Gamma:  1/Wkm

#betas = [b2_r ,-9.62314295e-04,  9.69699403e-05, -2.89675421e-06]

lambda_znw = 1650
w_znw = 2*np.pi*c/lambda_znw
gam1_r = -gam_r/(w_znw - omega0)*1


alph_r  = 0                        #alpha:  dB/m
TR_r    = 3e-3*0                   #TR:     fs
fR_r    = 0.18*0                   #fR:     adimensional (0.18)


nu2     = c/Lambda2
nu1     = c/Lambda1
dnu2    = nu2 - nu1


#Cargo objetos con los parámetros:
sim_r   = Sim(N_r, Tmax_r)
fibra_r = Fibra(L=L_r, beta2=b2_r, beta3=b3_r, gamma=gam_r, gamma1=gam1_r, alpha=alph_r, lambda0=Lambda1, TR = TR_r, fR = fR_r)


#Calculamos el pulso inicial
pulso_r = Two_Pulse(sim_r.tiempo, amp_1, amp_2, ancho_1, ancho_2, offset_1, offset_2, dnu2, pulses = "s")



#%% Beta(omega) reflection

from common.commonfunc import beta2w

beta2w(dnu2*0, fibra_r)

betaw = fibra_r.beta2/2 * (2*np.pi*sim_r.freq)**2 + fibra_r.beta3/6 * (2*np.pi*sim_r.freq)**3
beta1w = fibra_r.beta2 * (2*np.pi*sim_r.freq) + fibra_r.beta3/2 * (2*np.pi*sim_r.freq)**2
beta2ww = fibra_r.beta2 + fibra_r.beta3 * (2*np.pi*sim_r.freq)

lam, betalam = Adapt_Vector(sim_r.freq, omega0, betaw)
lam, beta1lam = Adapt_Vector(sim_r.freq, omega0, beta1w)
lam, beta2lam = Adapt_Vector(sim_r.freq, omega0, beta2ww)
beta_soliton = betalam + fibra_r.gamma * amp_1 

#betaw_lam = fibra_r.beta2/2 * (2*np.pi/lam-omega0)**2 + fibra_r.beta3/6 * (2*np.pi/lam-omega0)**3

plt.figure()
# =============================================================================
# plt.plot( fftshift(sim_r.freq), fftshift(betaw), label="$\\beta(\omega)$")
# plt.plot( fftshift(sim_r.freq), fftshift(beta1w), label="$\\beta_1(\omega)$")
# plt.plot( fftshift(sim_r.freq), fftshift(beta2ww), label="$\\beta_2(\omega)$")
# =============================================================================
plt.plot( lam, betalam, label="$\\beta(\lambda)$")
plt.plot( lam, beta_soliton, label="$\\beta_s(\lambda)$", linestyle="--")
plt.axvline(fibra_r.zdw, color="darkgrey", linestyle=":")
plt.legend(loc="best")
plt.xlabel("Wavelength (nm)")
plt.xlim([1460, 1655])
plt.ylim([-4,2])
plt.ylabel("$\\beta(\lambda)$ (m$^{-1}$)")
plt.grid(True,alpha=.3)
plt.tight_layout()
plt.show()

#%% Recoil dW test (Pickartz)

w_i = 2*np.pi*c/Lambda2 - omega0 #Freq. de onda incidente
dw_s = 3.8#3.1#6 #Delta W del solitón a la salida del choque
shift_c = b2_r * dw_s + b3_r/2 * dw_s**2 #Coeficiente
c_coef = -b3_r/6 * w_i**3 - b2_r/2 * w_i**2 + shift_c * w_i #D(wi), con el agregado del solitón.
coefs = [b3_r/6, b2_r/2, -shift_c, c_coef] #Coeficientes de la ecuación a resolver

def omega_to_lambda(w, w0): #Función para pasar de Omega a lambda.
    return 2*np.pi*c/(w0+w)

#Función para hallar lambda reflejado siguiendo el trabajo de Pickartz (Adiabatic theory...)
def find_reflection(fib:Fibra, lambda_i, lambda_s): 
    w_i     = 2*np.pi*c/lambda_i - fib.omega0
    dw_s    = 2*np.pi*c/lambda_s - 2*np.pi*c/fib.lambda0
    shift_c = fib.beta2 * dw_s + fib.beta3 * dw_s**2
    c_coef  = -fib.beta3/6 * w_i**3 - fib.beta2/2 * w_i**2 + shift_c * w_i
    coefs   = [fib.beta3/6, fib.beta2/2, -shift_c, c_coef]
    raices  = np.roots(coefs)
    #print(omega_to_lambda(raices,omega0))
    return omega_to_lambda(raices, fib.omega0)

#Función para hallar lambda soliton en función del reflejado (Adiabatic theory...)
def find_solitonlambda(fib:Fibra, lambda_i, lambda_r): 
    w_i     = 2*np.pi*c/lambda_i - fib.omega0
    w_r    = 2*np.pi*c/lambda_r - 2*np.pi*c/fib.lambda0
    a_coef = (w_r - w_i)*fib.beta3/2
    b_coef = (w_r - w_i)*fib.beta2
    c_coef = fib.beta2/2 * w_i**2 + fib.beta3/6 * w_i**3 - fib.beta2/2 * w_r**2 - fib.beta3/6 * w_r**3
    coefs   = [a_coef, b_coef, c_coef]
    raices  = np.roots(coefs)
    #print(omega_to_lambda(raices,omega0))
    return omega_to_lambda(raices, fib.omega0)

#Función para hallar lambda reflejado siguiendo el trabajo de Pickartz (Adiabatic theory...)
def find_reflection_test(fib:Fibra, lambda_i, lambda_s, deltaP): 
    w_i     = 2*np.pi*c/lambda_i - fib.omega0
    dw_s    = 2*np.pi*c/lambda_s - 2*np.pi*c/fib.lambda0
    shift_c = fib.beta2 * dw_s + fib.beta3 * dw_s**2
    c_coef  = -fib.beta3/6 * w_i**3 - fib.beta2/2 * w_i**2 + shift_c * w_i + fib.gamma * deltaP
    coefs   = [fib.beta3/6, fib.beta2/2, -shift_c, c_coef]
    raices  = np.roots(coefs)
    return omega_to_lambda(raices, fib.omega0)
    

raices = np.roots(coefs)
print(raices)
print(omega_to_lambda(raices,omega0))


#Caso lambda_i=1500, corrimiento en lambda del soliton en función de z:
#Po=10W, To=2.1ps
z_s1 = [267.2,280.6,294.8,309.8,322.5,338.2,352.4,368.1,390.5,
       408.4,433.1,456.3,472.0,492.2]

lambda_sol1 = [1600.9,1599.6,1598.5,1596.3,1595.3,1594.7,1594.7,
              1594,1593.7,1593.5,1593.5,1593,1592.9,1592.5]

reflect_wave1 = np.zeros_like(lambda_sol1)
for i in range( len(lambda_sol1) ):    
    reflect_wave1[i] = find_reflection(fibra_r, 1500, lambda_sol1[i])[1]

#Po=20W, To=3ps
z_s2 = [233,239,242.5,248.9,256.2,268.2,277.2,289.2,302.9,315.8]

lambda_sol2= [1600.1,1598.3,1596.4,1593.1,1591.2,1590.2,1588.3,1586.3,1583.9,1581.9]

reflect_wave2 = np.zeros_like(lambda_sol2)
for i in range( len(lambda_sol2) ):    
    reflect_wave2[i] = find_reflection(fibra_r, 1500, lambda_sol2[i])[1]
    
#Lambda soliton en función de lambda reflejado: Po = 10W, To = 2.1 ps
z_s3 = [286.3,304.4,323.6,344.4,374.4,425.6,494.9]

lambda_ref = [1531.75,1531.94,1532.89,1534.03,1535.17,1535.45,1535.55]

soliton_wave1 = np.zeros_like(lambda_ref)
for i in range( len(lambda_ref) ):
    soliton_wave1[i] = find_solitonlambda(fibra_r, 1500, lambda_ref[i])[1]

#%%

def betaw_f(sim:Sim, fib:Fibra, wavelength=False):
    beta  = fib.beta2/2 * (2*np.pi*sim.freq)**2 + fib.beta3/6 * (2*np.pi*sim.freq)**3
    beta1 = fib.beta2 * (2*np.pi*sim.freq) + fib.beta3/2 * (2*np.pi*sim.freq)**2
    beta2 = fib.beta2 + fib.beta3 * (2*np.pi*sim.freq)
    if wavelength:
        lam, beta = Adapt_Vector(sim.freq, fib.omega0, beta)
        lam, beta1 = Adapt_Vector(sim.freq, fib.omega0, beta1)
        lam, beta2 = Adapt_Vector(sim.freq, fib.omega0, beta2)
        return lam, [beta,beta1,beta2]
    else:
        return [beta,beta1,beta2]
   
lambda0_test = 1600
omega0 = 2*np.pi*c/lambda0_test
b2_test = -4.4e-3*1
lambda_zdw = 1555
w_zdw = 2*np.pi*c/lambda_zdw
b3_test = -b2_test/(w_zdw - omega0)*1

    
fibra_test = Fibra(300, b2_test, b3_test, gam_r, gam1_r, alph_r, lambda0_test)

lam_test, b_test = betaw_f(sim_r, fibra_test, wavelength=True)
amp_soliton = abs(b2_test)/(fibra_test.gamma * ancho_1**2)
beta_t_soliton = b_test[0] + fibra_r.gamma * amp_soliton 

plt.figure()
plt.plot(lam_test, b_test[0], label="$\\beta(\lambda)$")
plt.plot( lam_test, beta_t_soliton, label="$\\beta_s(\lambda)$", linestyle="--")
plt.axvline(lambda_zdw, color="darkgrey", linestyle=":")
plt.grid(True,alpha=.3)
plt.xlim([1460, 1655])
plt.ylim([-4,2])
plt.xlabel("Wavelength (nm)")
plt.ylabel("$\\beta(\lambda)$ (m$^{-1}$)")
plt.legend(loc="best")
plt.show()

#%% Beta cuártico test

def betaw4_f(sim:Sim, fib:Fibra, wavelength=False):
    beta  = fib.betas[0]/2 * (2*np.pi*sim.freq)**2 + fib.betas[1]/6 * (2*np.pi*sim.freq)**3 + fib.betas[2]/24 * (2*np.pi*sim.freq)**4
    beta1 = fib.betas[0] * (2*np.pi*sim.freq) + fib.betas[1]/2 * (2*np.pi*sim.freq)**2 + fib.betas[2]/6 * (2*np.pi*sim.freq)**3
    beta2 = fib.betas[0] + fib.betas[1] * (2*np.pi*sim.freq) + fib.betas[2]/2 * (2*np.pi*sim.freq)**3
    if wavelength:
        lam, beta = Adapt_Vector(sim.freq, fib.omega0, beta)
        lam, beta1 = Adapt_Vector(sim.freq, fib.omega0, beta1)
        lam, beta2 = Adapt_Vector(sim.freq, fib.omega0, beta2)
        return lam, [beta,beta1,beta2]
    else:
        return [beta,beta1,beta2]

lambda0_test = 1600
omega0 = 2*np.pi*c/lambda0_test
b2_t2 = -4.4e-3*1
lambda_zdw1 = 1555
lambda_zdw2 = 1647.8
w_zdw1 = 2*np.pi*c/lambda_zdw1
dw_zdw1 = (w_zdw1 - omega0)
w_zdw2 = 2*np.pi*c/lambda_zdw2
dw_zdw2 = (w_zdw2 - omega0)
b4_t2 = 2*b2_t2 * (dw_zdw2/dw_zdw1 - 1) / (-dw_zdw1*dw_zdw2 + dw_zdw2**2)
b3_t2 = -b2_t2/dw_zdw1 - b4_t2/2 * dw_zdw1

fibra_t2 = Fibra(300, b2_test, b3_test, gam_r, gam1_r, alph_r, lambda0_test, betas=[b2_t2,b3_t2,b4_t2])

lam_t2, b_t2 = betaw4_f(sim_r, fibra_t2, wavelength=True)

plt.figure()
plt.plot(lam_t2, b_t2[0], label="$\\beta(\lambda)$")
plt.axvline(lambda_zdw, color="darkgrey", linestyle=":")
plt.grid(True,alpha=.3)
plt.xlim([1400, 1800])
plt.ylim([-4,2])
plt.xlabel("Wavelength (nm)")
plt.ylabel("$\\beta(\lambda)$ (m$^{-1}$)")
plt.legend(loc="best")
plt.show()