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
from common.commonfunc import Sim, Fibra, Two_Pulse
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

def get_parameters(task_id):
    # Define parameter ranges
    peak_power_values = range(0, 101, 10)  # 0, 10, 20, ..., 100
    temporal_width_values = [i * 8/10 for i in range(1, 11)]  # 0.1, 1.1, 2.1, ..., 10.1

    # Calculate the total number of combinations
    total_combinations = len(peak_power_values) * len(temporal_width_values)

    if task_id > total_combinations:
        raise ValueError("Task ID exceeds the number of parameter combinations")

    # Determine the indices for peak_power and temporal_width
    peak_power_index = (task_id - 1) // len(temporal_width_values)
    temporal_width_index = (task_id - 1) % len(temporal_width_values)

    peak_power = peak_power_values[peak_power_index]
    temporal_width = temporal_width_values[temporal_width_index]

    return peak_power, temporal_width

task_id = int(sys.argv[1])
peak_power, temporal_width = get_parameters(task_id)

get_parameters(task_id)
print("peak power: "+str(peak_power))
print("temporal width: "+str(temporal_width))

#%% Parámetros de simulación

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
Lambda2  = 1480
amp_2    = peak_power
ancho_2  = temporal_width
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
zlocs, AW, AT = Solve_pcGNLSE(sim, fibra, pulso, z_locs=100)
t1 = time.time()

total_n = t1 - t0 #Implementar en Solve_pcGNLSE
print("Time",np.round(total_n/60,2),"(min)")

#%% Guardando el resultado

savedic = "data/firstsweep/"+str(task_id)

modsaver(AW, sim, fibra, savedic, f'{[Lambda1, amp_1, ancho_1, offset_1, Lambda2, amp_2, ancho_2, offset_2] = }')


