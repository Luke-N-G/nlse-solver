# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:43:29 2024

@author: d/dt Lucas
"""

#%% Imports

import numpy as np
import sys
import pickle
import datetime
import time
from common.commonfunc    import Sim, Fibra, Two_Pulse
from common.plotter       import plotcmap
from solvers.solvepcGNLSE import Solve_pcGNLSE

#%% Guardado alternativo: Solo guardamos el espectro

def modsaver(AW, sim:Sim, fib:Fibra, filename, other_par = None):
    # Guardando los parametros de simulación y de la fibra en un diccionario.
    metadata = {'Sim': sim.__dict__, 'Fibra': fib.__dict__} #sim.__dict__ = {'puntos'=N, 'Tmax'=70, ...}

    # Guardando los datos en filename-data.txt con pickle para cargar después.
    with open(f"{filename}-data.txt", 'wb') as f:
        pickle.dump((AW, metadata), f)
        
    # Guardar parametros filename-param.txt para leer directamente.
    with open(f"{filename}-param.txt", 'w') as f:
        f.write('-------------Parameters-------------\n')
        f.write(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}\n\n')
        for class_name, class_attrs in metadata.items():
            f.write(f'\n-{class_name}:\n\n')
            for attr, value in class_attrs.items():
                f.write(f'{attr} = {value}\n')
        if other_par:
            f.write("\n\n-Other Parameters:\n\n")
            if isinstance(other_par, str):
                f.write(f'{other_par}\n')
            else:
                for i in other_par:
                    f.write(f'{str(i)}\n')
                    
#%% Para el loop del cluster

def lambda_gain_raman(lam0, tau1):
    c = 299792458 * (1e9)/(1e12)
    delta_lambda = lam0**2/(2*np.pi*tau1*c)
    return delta_lambda

tau1_vec = lambda_gain_raman(1550, np.linspace(1,200,20))

task_id = 10#int(sys.argv[1])
tau1 = tau1_vec[task_id-1]

print("tau1: "+str(tau1))
print("freq: "+str(1/(tau1*2*np.pi)))
print("lambda: "+str(lambda_gain_raman(1550, tau1)))

#%% Parámetros de simulación

N = int(2*2**14) #puntos
Tmax = 2*70      #ps

c = 299792458 * (1e9)/(1e12)

Lambda0= 1600                    #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0

#Parámetros pulso 1:
Lambda1  = Lambda0
amp_1    = 243                   #Amplitud:  sqrt(W), Po = amp**2
ancho_1  = 85e-3                 #Ancho T0:  ps
offset_1 = 0

#Parámetros pulso 2:
Lambda2  = 1480
amp_2    = 30
ancho_2  = 5.75
offset_2 = 20 


#Parametros para la fibra
L     = 300                   #Lfib:   m
b2    = -4.4e-3               #Beta2:  ps^2/km
b3    = 0.13e-3               #Beta3:  ps^3/km
gam   = 2.5e-3                #Gamma:  1/Wkm

lambda_znw = 1650
w_znw = 2*np.pi*c/lambda_znw
gam1 = -gam/(w_znw - omega0)

alph  = 0                        #alpha:  dB/m
TR    = 3e-3                   #TR:     fs
fR    = 0.18                   #fR:     adimensional (0.18)

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
zlocs, AW, AT = Solve_pcGNLSE(sim, fibra, pulso, tau1=tau1, z_locs=100, pbar=True)
t1 = time.time()

total_n = t1 - t0 #Implementar en Solve_pcGNLSE
print("Time",np.round(total_n/60,2),"(min)")

#%% Guardando el resultado

savedic = "data/ramansweep/"

modsaver(AW, sim, fibra, savedic+str(task_id), f'{[Lambda1, amp_1, ancho_1, offset_1, Lambda2, amp_2, ancho_2, offset_2] = }')

plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=True, Tlim=[-30,30], Wlim=[1450,1750],
          vlims=[-20,50,0,120], zeros=False, save=savedic+"plots/"+str(task_id)+"plotcmap")


