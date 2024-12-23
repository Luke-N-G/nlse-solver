# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:12:56 2024

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

Lambda0= 1555                    #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0


#Parámetros pulso 1:
Lambda1  = Lambda0
amp_1    = 5e3                   #Amplitud:  sqrt(W), Po = amp**2
ancho_1  = 1#20e-3                  #Ancho T0:  ps
offset_1 = 0


#Parámetros pulso 2:
Lambda2  = 1310
amp_2    = 10*0
ancho_2  = 1
offset_2 = -43.5-3.3#-23.5 


#Parametros para la fibra
L     = 10                         #Lfib:   m
Ls   = 10e-3
b2    = -21e-3*1                  #Beta2:  ps^2/km
b3    = 9.3994e-5*1               #Beta3:  ps^3/km
gam   = 1.4e-3                  #Gamma:  1/Wkm
gams  = 23.4e-3

lambda_znw = 1650
w_znw = 2*np.pi*c/lambda_znw
gam1 = -gam/(w_znw - omega0)*0

alph  = 0                        #alpha:  dB/m
TR    = 3e-3*0                   #TR:     fs
fR    = 0.18*0                   #fR:     adimensional (0.18)

#Diferencia de frecuencias
nu2     = c/Lambda2
nu1     = c/Lambda1
dnu2    = nu2 - nu1

#Cargo objetos con los parámetros:
sim    = Sim(N, Tmax)
fibra  = Fibra(L=L, beta2=b2, beta3=b3, gamma=gam, gamma1=gam1, alpha=alph, lambda0=Lambda1, TR = TR, fR = fR)
sensor = Fibra(L=Ls,beta2=b2, beta3=b3, gamma=gams, gamma1=gam1, alpha=alph, lambda0=Lambda1, TR = TR, fR = fR)

#Asegurando que el pulso 1 sea soliton
ancho_1 = np.sqrt(np.abs(b2)/(gam*amp_1))

#Calculamos el pulso inicial
pulso = Two_Pulse(sim.tiempo, amp_1, amp_2, ancho_1, ancho_2, offset_1, offset_2, dnu2, pulses = "sp")

#%% Chirp

'''
Agrawal pag. 57
Solución analítica de un pulso gaussiano chirpeado.
'''
def gauss_chirp(z, T, C, T0, b2):
    return T0/np.sqrt(T0**2 - 1j*b2*z*(1+1j*C)) * np.exp( -(1+1j*C)*T**2/(2*(T0**2-1j*b2*z*(1+1j*C)))   )


'''
Agrawal pag. 57
Calcula el ancho de un pulso inicialmente chirpeado
'''
def new_width(T0, C, z, b2):
    new_T0 = np.sqrt( (1+C*b2*z/T0**2)**2 + (b2*z/T0**2)**2  ) * T0
    return new_T0

'''
Agrawal pag. 58
Calcula el ancho mínimo, y la distancia a la que se lo logra
'''
def min_width(T0, C, b2):
    z_min = np.abs(C) * T0**2 / (np.abs(b2) * (1 + C**2))
    T_min = T0/np.sqrt(1 + C**2)
    return T_min, z_min

'''
Cuenta mia horrible, potencia pico de un pulso chirpeado (en módulo)
'''
def new_peak(T0, C, z, b2):
    new_P0 = T0**2 / np.sqrt( (T0**2 + b2*z*C)**2 + (b2*z)**2 )
    return new_P0
    
def max_peak(C,b2):
    P_max = 1 / np.sqrt( (1 + np.sign(b2*C) * C**2/(1+C**2))**2 + (C/(1+C**2))**2  )
    return P_max


#%%

zlocs = np.linspace(0,200, 1000)
T0 = 1


Lambda1  = 1550
amp_1    = 10e-3/16.6
ancho_1  = 105.26#20e-3
chirp_1  = 2770.083
offset_1 = 0

gauss_matrix = np.zeros([len(zlocs), len(sim.tiempo)], dtype=complex)
for i, j in enumerate(zlocs):
   gauss_matrix[i] = gauss_chirp(j, sim.tiempo, chirp_1, ancho_1, b2)


plt.figure()
plt.imshow(Pot(gauss_matrix), aspect='auto', origin='lower')
plt.show()

plt.figure()
plt.plot(zlocs, [np.max(Pot(gauss_matrix[i])) for i in range(len(zlocs))])
plt.title("Power")
plt.show()

plt.figure()
plt.plot(zlocs, [new_width(ancho_1, chirp_1, z, b2) for z in zlocs])
plt.title("Width")
plt.show()

plt.figure()
plt.plot(zlocs, [new_peak(ancho_1, chirp_1, z, b2) for z in zlocs])
plt.title("Peak")
plt.show()

#%%

T_min = np.zeros(100)
z_min = np.zeros(100)
P_max = np.zeros(100)
chirp = np.arange(0,100)
for i in chirp:
    T_min[i], z_min[i] = min_width(ancho_1, i, b2)
    P_max[i] = np.max(Pot(gauss_chirp(z_min[i], sim.tiempo, i, T0, b2)))
    

plt.figure()
plt.plot(chirp, z_min)
plt.title('zmin')
plt.show()

plt.figure()
plt.plot(chirp, T_min)
plt.title('Tmin')
plt.show()

plt.figure()
plt.plot(chirp, P_max)
plt.title('Pmax')
plt.show()
