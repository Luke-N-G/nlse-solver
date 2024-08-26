# -*- coding: utf-8 -*-
"""
Created on Thr Aug 22 10:10:32 2024

@author: d/dt Lucas
"""

#%% Imports

import numpy as np
from common.commonfunc import ReSim, FT, IFT, fftshift, Pot, Fibra, Sim, Adapt_Vector
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pickle

#%% Nuevas Funciones

# Mod loader (solo carga espectro y metadata)
def modloader(filename, resim = None):
    with open(f"{filename}-data.txt", 'rb') as f:
        AW, metadata = pickle.load(f)
    if resim:
        sim, fibra = ReSim(metadata)
        return AW, sim, fibra
    else:
        return AW, metadata
    
#Función de fiteo
def soliton_fit(T, amplitude, center, width, offset):
    carrier = np.sqrt(amplitude)*( 1/np.cosh( (T - center)/width) ) + offset*0
    return np.abs( carrier )**2

#Conteo de solitones
def soliton_number(fib:Fibra, sim:Sim, AW):
    
    prominence  = 50  #Prominencia, para hallar picos
    window_size = 100 #Número de puntos alrededor de cada pico
    z_index     = -1  #A que z analizamos (se podría pasar como variable de función)
    
    #Buscamos freq. donde enmascarar
    mask_i     = fib.lambda_to_omega( fib.zdw )/(2*np.pi)
    mask_f     = fib.lambda_to_omega( fib.znw )/(2*np.pi)
    
    #Buscamos los índices donde enmascarar
    mask_i_idx = np.argmin( np.abs(sim.freq - mask_i) )
    mask_f_idx = np.argmin( np.abs(sim.freq - mask_f) )
    
    #Enmascaramos, teniendo en cuenta de que el array AW esta shifteado
    AW_mask = AW[z_index][:mask_i_idx]
    AW_mask = np.append(AW_mask, np.zeros_like(sim.freq[mask_i_idx:mask_f_idx]) )
    AW_mask = np.append(AW_mask, AW[z_index][mask_f_idx:])
    
    #Vamos a dominio del tiempo
    AT_mask = IFT(AW_mask)
    
    #Buscamos los índices de los picos
    peaks, _ = find_peaks( Pot(AT_mask), prominence = prominence )
    
    soliton_count = 0
    
    for peak in peaks:
        #Extraemos una ventana alrededor de un pico
        window = AT_mask[peak-window_size:peak+window_size]
        
        #Ventana temporal
        t_window = sim.tiempo[peak-window_size:peak+window_size]
        
        #Parámetros iniciales de ajuste
        p0 = [Pot(AT_mask[peak]), sim.tiempo[peak], 1, 0]
        
        #Tomamos el ajuste
        popt, pcov = curve_fit(soliton_fit, t_window, Pot(window), p0 = p0, maxfev = 10000)
        
        #Vemos calidad del ajuste
        residuals = Pot(window) - soliton_fit(t_window, *popt)
        rss = np.sum(residuals**2)

        #Estudiamos el espectro
        
        #Armamos un vector que sea = 0 en todos lados, menos en el pulso de estudio
        pulse_window = np.zeros_like(sim.tiempo, dtype=complex)
        pulse_window[peak-window_size:peak+window_size] = window
        
        #Transformamos a espectro
        spectral_window = IFT(pulse_window)
        
        #Buscamos el máximo espectral
        max_peak_index = np.argmax( Pot(spectral_window) )
        
        #Frecuencia central del pulso, en base a esto calculamos gamma y beta
        pulse_centerfreq = sim.freq[max_peak_index]*-1
        pulse_gamma = fib.gamma_w(pulse_centerfreq)
        pulse_beta2 = fib.beta2_w(pulse_centerfreq)
        
        #Parámetros para hallar el N del solitón
        pulse_amp   = Pot(AT_mask[peak])
        pulse_width = popt[2]
        soliton_order = np.sqrt(pulse_amp * pulse_width**2 * pulse_gamma / np.abs(pulse_beta2) )
        
        #Si el orden está entre 0.5 y 1.5, contamos
        if 0.5 <= soliton_order <= 1.5*2:
            soliton_count += 1
    
    return soliton_count

#%% Testing

savedic = "soliton_gen/"

AW, sim, fibra = modloader(savedic+"100", resim=True)