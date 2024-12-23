# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:37:39 2023

@author: d/dt Lucas
"""

import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq
import pickle
import datetime


# =============================================================================
# FFTs viejas, conservar en caso de que las nuevas no funcionen bien.
# #FFT siguiendo la convención -iw (Agrawal)
# def FT(pulso):
#     return ifft(pulso) * len(pulso)
# 
# #IFFT siguiendo la convención -iw
# def IFT(espectro):
#     return fft(espectro) / len(espectro)
# =============================================================================

# FFT siguiendo la convención -iw (Agrawal)
def FT(pulso):
    if pulso.ndim == 1:
        return ifft(pulso) * len(pulso)
    elif pulso.ndim == 2:
        return ifft(pulso, axis=1) * pulso.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

# IFFT siguiendo la convención -iw
def IFT(espectro):
    if espectro.ndim == 1:
        return fft(espectro) / len(espectro)
    elif espectro.ndim == 2:
        return fft(espectro, axis=1) / espectro.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

#Pasamos de array tiempo a array frecuencia
def t_a_freq(t_o_freq):
    return fftfreq( len(t_o_freq) , d = t_o_freq[1] - t_o_freq[0])

#Defino función para una super gaussiana
def SuperGauss(t,amplitud,ancho,offset,chirp,orden):
    return np.sqrt(amplitud)*np.exp(- (1+1j*chirp)/2*((t-offset)/(ancho))**(2*np.floor(orden)))*(1+0j)

#Solitón con función Sech
def Soliton(t, ancho, b2, gamma, orden):
    return orden * np.sqrt( np.abs(b2)/(gamma * ancho**2) ) * (1/ np.cosh(t / ancho) )

#Esquema para colisión de pulsos
# pulse = "s", solo signal, "p" solo pump, "sp" ambos.
def Two_Pulse(T, amp1, amp2, ancho1, ancho2, offset1, offset2, nu, pulses):
    t_1 = T/ancho1
    t_2 = T/ancho2
    
    np.seterr(over='ignore') #Silenciamos avisos de overflow (1/inf = 0 para estos casos)
    
    pump   = np.sqrt(amp1)*(1/np.cosh(t_1))
    signal = np.sqrt(amp2)*(1/np.cosh(t_2 + offset2/ancho2))*np.exp(-2j*np.pi*nu*T)
    
    np.seterr(over='warn') #Reactivamos avisos de overflow.

    pulse_dict = {"s": signal, "p": pump, "sp": pump + signal}
    if pulses not in pulse_dict:
        raise ValueError(f"Tipo de pulso no valido: {pulses}. Los tipos son: 's', 'p', 'sp'.")

    return pulse_dict.get(pulses)

#Función que calcula la potencia
def Pot(pulso): 
    return np.abs(pulso)**2

#Función energía
def Energia(t_o_freq, señal):
    return np.trapz(Pot(señal), t_o_freq)

#Función numero de fotones
def num_fotones(freq, espectro, w0):
    return np.sum( np.abs(espectro)**2 / (freq + w0)  )

def find_chirp(t, señal):
    fase = np.unwrap( np.angle(señal) ) #Angle busca la fase, Unwrap la extiende de [0,2pi) a todos los reales.
    #fase = fase - fase[ int(len(fase)/2)  ] #Centramos el array
    df = np.diff( fase, prepend = fase[0] - (fase[1]  - fase[0]  ),axis=0)
    dt = np.diff( t, prepend = t[0]- (t[1] - t[0] ),axis=0 )
    chirp = -df/dt
    return chirp

def find_shift(zlocs, freq, A_wr):
    peaks = np.zeros( len(zlocs), dtype=int )
    dw    = np.copy(peaks)
    for index_z in range( len(zlocs) ):
        peaks[index_z] = np.argmax( Pot( fftshift(A_wr[index_z]) ) )
        #dw[index_z] = fftshift(2*np.pi*sim_r.freq)[peaks[index_z]]
    dw = fftshift(2*np.pi*freq)[peaks]
    return dw

def find_k(Aw, dz, rel_thr=1e-6):
    Aw = Aw.T  
    ks    = np.zeros_like(Aw, dtype='float64')
    phis  = np.zeros_like(ks,  dtype='float64')
    mask = abs(Aw)>rel_thr*np.max(np.abs(Aw))
    phis[mask] = np.unwrap( np.angle(Aw[mask]), axis=0 )
    ks = np.diff(phis)/dz
    ks[np.diff(mask) != 0] = 0
    return ks, phis


def Adapt_Vector(freq, omega0, Aw):
    lambda_vec = 299792458 * (1e9)/(1e12) / (freq + omega0/(2*np.pi))
    sort_indices = lambda_vec.argsort()
    lambda_vec_ordered = lambda_vec[sort_indices]
    
    #Esto es para identificar los distintos tipos de arrays con los que trabajamos
    if Aw.ndim == 1:
        if Aw.dtype == object:  # Aw es un array de objetos
            Alam = np.array([aw[sort_indices] for aw in Aw])
        else:  # Aw es un array 1D
            Alam = Aw[sort_indices]
    elif Aw.ndim == 2:  # Aw es un array 2D (El estandar)
        Alam = Aw[:, sort_indices]
    else:
        raise ValueError("AW debe ser un array 1D o 2D")

    return lambda_vec_ordered, Alam


#Defino clases que guarden tanto los parámetros de la simulación como los de la fibra

class Sim:
    def __init__(self, puntos, Tmax):
        self.puntos = puntos                         #Número de puntos sobre el cual tomar el tiempo
        self.Tmax   = Tmax
        self.paso_t = 2.0*Tmax/puntos
        self.tiempo = np.arange(-puntos/2,puntos/2)*self.paso_t
        self.dW     = np.pi/Tmax
        self.freq   = fftshift( np.pi * np.arange(-puntos/2,puntos/2) / Tmax )/(2*np.pi)
        

class Fibra:
    def __init__(self, L, beta2, beta3, gamma, gamma1, alpha, lambda0, TR=3e-3, fR=0.18, betas=0 ):
        self.L  = L         #Longitud de la fibra
        self.beta2 = beta2  #beta_2 de la fibra, para calcular GVD
        self.beta3 = beta3  #beta_3 de la fibra, para calcular TOD
        self.betas = betas  #Vector con los coeficientes beta
        self.gamma = gamma  #gamma de la fibra, para calcular SPM
        self.gamma1= gamma1 #gamma1 de la fibra, para self-steepening
        self.alpha = alpha  #alpha de la fibra, atenuación
        self.TR    = TR     #TR de la fibra, para calcular Raman
        self.fR    = fR     #fR de la fibra, para calcular Raman (y self-steepening)
        self.lambda0 = lambda0 #Longitud de onda central
        self.omega0  = 2*np.pi* 299792458 * (1e9)/(1e12) /lambda0 #Frecuencia (angular) central
        if self.beta3 != 0:
            self.w_zdw = -self.beta2/self.beta3 + self.omega0
            self.zdw   = 2*np.pi* 299792458 * (1e9)/(1e12) / self.w_zdw
        else:
            self.w_zdw = None
            self.zdw   = None
        if self.gamma1 != 0:
            self.w_znw = -self.gamma/self.gamma1 + self.omega0
            self.znw   = 2*np.pi* 299792458 * (1e9)/(1e12) /self.w_znw
        else:
            self.w_znw = None
            self.znw   = None
    
    #---Algunos métodos útiles---
    #Método para pasar de omega a lambda
    def omega_to_lambda(self, w):  #Función para pasar de Omega a lambda.
        return 2*np.pi* 299792458 * (1e9)/(1e12)/(self.omega0+w)
    def lambda_to_omega(self,lam): #Función para pasar de lambda a Omega.
        return 2*np.pi*299792458 * (1e9)/(1e12) * (1/lam - 1/self.lambda0)
    #Método para calcular gamma en función de omega
    def gamma_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        return self.gamma + self.gamma1 * w
    #Método para calcular beta2 en función de omega
    def beta2_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        if self.betas != 0:
            beta2 = 0
            for i, beta in enumerate(self.betas):
                beta2 += beta * w**i / np.math.factorial(i)
        else:
            beta2 = self.beta2 + self.beta3 * w
        return beta2


#Ver como meter esto en la clase Fibra directamente.
def beta2w(freq, fib:Fibra):
    return fib.beta2 + fib.beta3  * (2*np.pi*freq)

#%% Funciones para guardar y cargar datos


# Guardando (BAJO PRUEBA!)
def saver(AW, AT, sim:Sim, fib:Fibra, filename, other_par = None):
    # Guardando los parametros de simulación y de la fibra en un diccionario.
    metadata = {'Sim': sim.__dict__, 'Fibra': fib.__dict__} #sim.__dict__ = {'puntos'=N, 'Tmax'=70, ...}

    # Guardando los datos en filename-data.txt con pickle para cargar después.
    with open(f"{filename}-data.txt", 'wb') as f:
        pickle.dump((AW, AT, metadata), f)
        
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

# Cargando datos
def loader(filename, resim = None):
    with open(f"{filename}-data.txt", 'rb') as f:
        AW, AT, metadata = pickle.load(f)
    if resim:
        sim, fibra = ReSim(metadata)
        return AW, AT, sim, fibra
    else:
        return AW, AT, metadata

# Cargando metadata en las clases
# Devuelve objetos sim:Sim and fib:Fibra objects, ya cargados con los parámetros.

def ReSim(metadata):
    # Define the parameters that Sim class's __init__ method accepts
    sim_params = ['puntos', 'Tmax']
    
    # Filter the metadata to only include the parameters that Sim class's __init__ method accepts
    sim_m = {k: v for k, v in metadata['Sim'].items() if k in sim_params}
    
    # Load the saved parameters in metadata to the Sim and Fibra classes, returning sim and fibra objects.
    sim = Sim(**sim_m)
    
    # Define the parameters that Fibra class's __init__ method accepts
    fibra_params = ['L', 'beta2', 'beta3', 'gamma', 'gamma1', 'alpha', 'lambda0', 'TR', 'fR', 'betas']
    
    # Filter the metadata to only include the parameters that Fibra class's __init__ method accepts
    fib_m = {k: v for k, v in metadata['Fibra'].items() if k in fibra_params}
    
    fibra = Fibra(**fib_m)
    return sim, fibra



#%% Saver antiguo

#Guardado
# =============================================================================
# def saver(AW, AT, sim:Sim, fib:Fibra, filename, other_par = None):
#     
#     #Guardamos los parámetros de la fibra y simulación en un diccionario
#     metadata = {}
#     metadata['Sim']   = {}
#     metadata['Sim']['puntos']    = sim.puntos
#     metadata['Sim']['Tmax']      = sim.Tmax
#     metadata['Fibra'] = {}
#     metadata['Fibra']['L']       = fib.L
#     metadata['Fibra']['beta2']   = fib.beta2
#     metadata['Fibra']['beta3']   = fib.beta3
#     metadata['Fibra']['gamma']   = fib.gamma
#     metadata['Fibra']['gamma1']  = fib.gamma1
#     metadata['Fibra']['alpha']   = fib.alpha
#     metadata['Fibra']['lambda0'] = fib.lambda0
#     metadata['Fibra']['TR']      = fib.TR
#     metadata['Fibra']['fR']      = fib.fR
#     metadata['Fibra']['ZNW']     = fib.znw
#     metadata['Fibra']['ZDW']     = fib.zdw
#     
#     #Se guardan los datos a filename-data.txt con pickle para cargar más adelante.
#     with open(filename+"-data.txt", 'wb') as f:
#         pickle.dump(((AW, AT, metadata)), f)
#         
#     #Se guardan los parámetros a filename-param.txt para leer de ser necesario.
#     with open(filename+'-param.txt', 'w') as f:
#         lines = ['-------------Parameters-------------\n']
#         lines.append(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')+"\n\n")
#         lines.append('-Sim: \n \n')
#         for i in metadata['Sim']:
#             lines.append(i+' = '+str(metadata['Sim'][i])+"\n")
#         lines.append('\n\n-Fibra: \n \n')
#         for i in metadata['Fibra']:
#             lines.append(i+' = '+str(metadata['Fibra'][i])+"\n")
#         if other_par:
#             lines.append("\n\n-Other Parameters:\n\n")
#             for i in other_par:
#                 if type(i) == str:
#                     lines.append(i)
#                 else:
#                     lines.append(str(i)+'\n')
#         f.writelines(lines)
# 
# #Cargado de datos
# def loader(filename, resim = None):
#     with open(filename+"-data.txt", 'rb') as f:
#         AW, AT, metadata = pickle.load(f)
#     if resim:
#         sim, fibra = ReSim(metadata)
#         return AW, AT, sim, fibra
#     else:
#         return AW, AT, metadata
# 
# #Cargado de metadata en clases para simular con parámetros viejos
# #Devuelve objetos sim:Sim y fib:Fibra, ya cargado con datos.
# def ReSim(metadata):
#     
#     #Definimos variables auxiliares para comprimir un poco la notación
#     sim_m = metadata['Sim']
#     fib_m = metadata['Fibra']
#     
#     #Cargamos los parámetros guardados en metadata a la clases Sim y Fibra, devolviendo objetos sim y fibra.
#     sim   = Sim(sim_m['puntos'], sim_m['Tmax'])
#     fibra = Fibra(L=fib_m['L'], beta2=fib_m['beta2'], beta3=fib_m['beta3'],
#                   gamma=fib_m['gamma'], gamma1=fib_m['gamma1'], alpha=fib_m['alpha'],
#                   lambda0=fib_m['lambda0'], TR=fib_m['TR'], fR=fib_m['fR'])
#     return sim, fibra
# 
# =============================================================================

