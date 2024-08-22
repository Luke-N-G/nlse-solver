# -*- coding: utf-8 -*-
"""
Created on Thr Aug 22 10:10:32 2024

@author: d/dt Lucas
"""

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
