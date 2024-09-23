# -*- coding: utf-8 -*-
"""
Created on Thr Aug 29 09:41:32 2024

@author: d/dt Lucas
"""

#%% Imports

import numpy as np
from common.commonfunc import ReSim, FT, IFT, fftshift, Pot, Fibra, Sim, Adapt_Vector
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pickle

#Imports temporarios, BORRAR!!!
from common.plotter import plotcmap, plotinst, plt

#%% Soliton number 

# Mod loader (solo carga espectro y metadata)
def modloader(filename, resim = None):
    with open(f"{filename}-data.txt", 'rb') as f:
        AW, metadata = pickle.load(f)
    if resim:
        sim, fibra = ReSim(metadata)
        return AW, sim, fibra
    else:
        return AW, metadata

# Funcion de ajuste
def soliton_fit(T, amplitude, center, width, offset):
    if amplitude <= 0: #Si toda la amplitud es cero o menor a cero por el ajuste, tomarla cero
        return np.zeros_like(T)
    carrier = np.sqrt(amplitude) * (1 / np.cosh((T - center) / width)) + offset * 0
    return np.abs(carrier) ** 2

#Conteo de solitones
def soliton_number(fib:Fibra, sim:Sim, AW,
                   z_index = -1, plot_signal=False, plot_fits=False, prominence = 50, window_size = 100):
    
    prominence  = 50  #Prominencia, para hallar picos
    window_size = 50 #Número de puntos alrededor de cada pico
    z_index     = z_index  #A que z analizamos (se podría pasar como variable de función)
    
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
    
    if plot_signal:
        plt.figure()
        plt.plot(sim.tiempo, Pot(AT_mask))
        plt.title("Clean output")
        plt.xlabel("Time (ps)")
        plt.ylabel("Power (W)")
        plt.grid(True,alpha=.3)
        plt.show()
    
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
            
        if plot_fits:
            plt.figure()
            plt.plot(t_window, Pot(window), ".", label="Datos")
            plt.plot(t_window, soliton_fit(t_window, *popt), label="Ajuste, N: "+str(soliton_order))
            plt.legend(loc="best")
            plt.title("Fit peak"+str(peak))
            plt.xlabel("Time (ps)")
            plt.ylabel("Power (W)")
            plt.grid(True, alpha=.3)
            plt.show()
    
    return soliton_count

# Contamos la una cantidad máxima de solitones entre ciertos valores de z
def max_soliton_count(fib, sim, AW, z_indices):
    max_count = 0
    for z_index in z_indices: #Barremos en z, contando cuantos solitones hay en cada posición
        count = soliton_number(fib, sim, AW, z_index)
        if count > max_count:
            max_count = count
    return max_count #Devuelve el máximo número hallado

# Hallar parametros (Solo funciona para el barrido de 1 a 100!)
def get_parameters(task_id):
    # Definimos los rangos de parámetros
    #peak_power_values = range(0, 101, 10)  # 0, 10, 20, ..., 100
    #temporal_width_values = [i * 8/10 for i in range(1, 11)]  # 0.1, 1.1, 2.1, ..., 10.1
    peak_power_values = np.linspace(0,100,20)  # 0, 10, 20, ..., 100
    temporal_width_values = np.linspace(0.5,8, 20)  # 0.1, 1.1, 2.1, ..., 10.1

    # Calculamos todas las convinaciones
    total_combinations = len(peak_power_values) * len(temporal_width_values)

    if task_id > total_combinations:
        raise ValueError("Task ID exceeds the number of parameter combinations")

    # Determinamos indices para potencia pico y ancho temporal
    peak_power_index = (task_id - 1) // len(temporal_width_values)
    temporal_width_index = (task_id - 1) % len(temporal_width_values)

    peak_power = peak_power_values[peak_power_index]
    temporal_width = temporal_width_values[temporal_width_index]

    return peak_power, temporal_width

#Funcion que devuelve matriz de numero de solitones N, y matrices con parámetros potencia y tiempo
def generate_soliton_matrix():
    # Definimos rangos de los parámetros
    #peak_power_values = range(0, 101, 10)  # 0, 10, 20, ..., 100
    #temporal_width_values = [i * 8/10 for i in range(1, 11)]  # 0.8, 1.6, 2.4, ..., 8.0
    peak_power_values = np.linspace(0,100,20)  # 0, 10, 20, ..., 100
    temporal_width_values = np.linspace(0.5,8, 20)  # 0.1, 1.1, 2.1, ..., 10.1

    # Initialize the N matrix
    N = np.zeros((len(peak_power_values), len(temporal_width_values)))
    P = np.zeros((len(peak_power_values), len(temporal_width_values)))
    T = np.zeros((len(peak_power_values), len(temporal_width_values)))

    # Iteramos sobre los task_id
    for task_id in range(1, 401):
        # Cargamos los datos numerados entre 1 y 100
        try:
            AW, sim, fibra = modloader("soliton_gen/secondsweep/" + str(task_id), resim=True)
        except Exception as e:
            print(f"Error loading data for task_id {task_id}: {e}")
            continue

        # Obtenemos parametros
        peak_power, temporal_width = get_parameters(task_id)

        # Calculamos los índices de la matriz N
        peak_power_index = (task_id - 1) // len(temporal_width_values)
        temporal_width_index = (task_id - 1) % len(temporal_width_values)

        # Calculamos el máximo numero de solitones
        z_indices = range(50, len(AW))  # or any other range of z indices you want to test
        try:
            max_count = max_soliton_count(fibra, sim, AW, z_indices)
        except Exception as e:
            print(f"Error calculating soliton number for task_id {task_id}: {e}")
            max_count = 0

        # Guardamos los resultados en la matriz N, armamos matrices P y T
        N[peak_power_index, temporal_width_index] = max_count
        P[peak_power_index, temporal_width_index] = peak_power
        T[peak_power_index, temporal_width_index] = temporal_width

        # Printeamos el output
        print(f"Task ID: {task_id}, Peak Power: {peak_power}, Temporal Width: {temporal_width}, Max Solitons: {max_count}")
    
    N = N-1 #Restamos el solitón original (que siempre está)
    
    return N, P, T

# Plot de la matriz N

def plot_soliton_matrix(N, P, T, cmap="viridis"):
    # Extract unique peak power and temporal width values
    peak_power_values = np.unique(P)
    temporal_width_values = np.unique(T)

    # Create a colormap with discrete colors
    cmap = plt.cm.get_cmap(cmap, int(np.max(N)) + 1)

    # Create the plot
    plt.figure(figsize=(10, 8))
    
    
    dT = (np.unique(T)[1] - np.unique(T)[0])/2
    dP = (np.unique(P)[1] - np.unique(P)[0])/2
    
    plt.imshow(N, aspect='auto', origin='lower',
               extent=[temporal_width_values[0] - dT, temporal_width_values[-1] + dT, peak_power_values[0] - dP,
                       peak_power_values[-1] + dP], interpolation='None', cmap=cmap)

    
    # Set axis labels
    plt.xlabel('Temporal width (ps)')
    plt.ylabel('Peak power (W)')
    
    # Add color bar
    cbar = plt.colorbar()
    cbar.set_label('Number of solitons')
    
    # Set color bar ticks at the middle of each color
    tick_locs = (np.arange(0, np.max(N) +1) + 0.5) * (np.max(N) / (np.max(N) + 1))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(np.arange(0, np.max(N) + 1, dtype=int))
    
    # Set axis ticks
    plt.xticks(temporal_width_values, np.round(temporal_width_values,1))
    plt.yticks(peak_power_values)
    
    # Show the plot
    plt.title('Soliton Count Matrix (prominence = 50, window= 50)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
#%% Raman analysis

def Raman_reflection(fib:Fibra, sim:Sim, AT, AW, zindex=-1, window_size = 100, prominence = 80):
    
    AT_end = AT[zindex]
    peaks, _ = find_peaks( Pot(AT_end), prominence = prominence )
    last_peak_index = peaks[-1]
    
    reflected_window = AT_end[:last_peak_index-window_size]
    transmitted_window = AT_end[last_peak_index+window_size:]
    
    AT_start = AT[0]
    peaks_start, _ = find_peaks( Pot(AT_start), prominence = prominence )
    last_peak_s_index = peaks_start[-1]
    incident_window = AT_start[:last_peak_s_index-window_size]
    
    incident_energy = np.sum( Pot(incident_window) )
    reflected_energy = np.sum( Pot(reflected_window) )
    transmitted_energy = np.sum( Pot(transmitted_window) )
    
    end_soliton_energy = np.sum( Pot(AT_end[last_peak_index-window_size:last_peak_index+window_size] ) )
    org_soliton_energy = np.sum( Pot(AT_start[last_peak_s_index-window_size:last_peak_s_index+window_size]) )
    soliton_energy_delta = end_soliton_energy - org_soliton_energy
    
    whole_energy_loss = np.sum(Pot(AT[0])) - np.sum(Pot(AT[-1]))
    
    loss = soliton_energy_delta - whole_energy_loss
        
    
    reflection_ratio = reflected_energy/incident_energy
    transmission_ratio = transmitted_energy/incident_energy
    soliton_gain_ratio = (end_soliton_energy - org_soliton_energy)/incident_energy
    
    n_solitons = soliton_number(fib, sim, AW)

    print(reflection_ratio+transmission_ratio+soliton_gain_ratio)
    print(reflected_energy/transmitted_energy)
    
    return reflected_energy, transmitted_energy, n_solitons

def lambda_gain_raman(lam0, tau1):
    c = 299792458 * (1e9)/(1e12)
    delta_lambda = lam0**2/(2*np.pi*tau1*c)
    return delta_lambda

tau1_vec = lambda_gain_raman(1550, np.linspace(1,250,50))
freq_vec = 1/(tau1_vec*2*np.pi)

#%% Testeo individual

savedic = "soliton_gen/ramansweep/"

AW, sim, fibra = modloader(savedic+"20", resim=True)
AT = IFT(AW)
zlocs = np.linspace(0,300,100)

plotcmap(sim, fibra, zlocs, AT, AW, legacy=False, dB=True, wavelength=True,
         vlims=[-20,70,0,120], Tlim=[-25,25], Wlim=[1400,1700], zeros=True)

plt.figure()
plt.plot(sim.tiempo, Pot(AT)[-1])
plt.grid(True,alpha=.3)
plt.xlabel("Time (ps)")
plt.ylabel("Peak power (W)")
plt.xlim([-25,25])
plt.show()

Rf, Tr, N_s = Raman_reflection(fibra, sim, AT, AW)

#%% Uso sobre todos los datos

savedic="soliton_gen/ramansweep2/"

Rf_v = np.zeros(50)
Tr_v = np.zeros(50)
N_sv  = np.zeros(50)
zlocs = np.linspace(0,300,100)
for i in range(1,51):
    AW, sim, fibra = modloader(savedic+str(i), resim=True)
    AT = IFT(AW)
    Rf, Tr, N_s = Raman_reflection(fibra, sim, AT, AW)
    Rf_v[i-1] = Rf
    Tr_v[i-1] = Tr
    N_sv[i-1]  = N_s
N_sv = N_sv - 1
    
Ratio = Rf_v/Tr_v

#%% Ploteo (Con número de solitones)

tick_size = 14
label_size = 14

fig, ax1 = plt.subplots()

color = 'blue'
ax1.set_xlabel("Maximum gain frequency (THz)", size=label_size)
ax1.set_ylabel("$E_R/E_T$", color=color, size=label_size)
ax1.plot(freq_vec, Ratio, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=tick_size)
ax1.tick_params(axis="x", labelsize=tick_size)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'red'
ax2.set_ylabel("Number of solitons", color=color, size=label_size)  # we already handled the x-label with ax1
ax2.set_yticks( np.unique(N_sv) )
ax2.plot(freq_vec, N_sv, ".--", color=color, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color, labelsize=tick_size)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%% Ploteo sin solitones


def Raman_graph(T,tau1,tau2):
    hR = np.zeros( len(T) )
    hR[T>=0] = (tau1**2+tau2**2)/(tau1*tau2**2) * np.exp(-T[T>=0]/tau2) * np.sin(T[T>=0]/tau1) #Definimos el hR(T)
    hR[T<0]  = 0
    
    hR = fftshift(hR)  #Shifteamos para que la respuesta empiece al principio de la ventana temporal
    hR = hR/np.sum(hR) #Normalizamos, tal que int(hR) = 1    
    hR_W = FT(hR)      #Pasamos el hR_W a frecuencia
    
    return hR_W

taus = [1, 0.014, 0.008]
gR_tau = np.zeros(len(taus), dtype=object)
for i in range(len(taus)):
    gR_tau[i] = np.imag( Raman_graph(sim.tiempo, taus[i], 32e-3) )
    gR_tau[i][sim.freq<=0] = 0
    


#%%

tick_size = 14
label_size=14

fig, ax1 = plt.subplots()

color = 'blue'
ax1.set_xlabel("Maximum gain frequency (THz)", size=label_size)
ax1.set_ylabel("$E_R/E_T$", size=label_size)
ax1.plot(freq_vec, Ratio, color=color)
ax1.tick_params(axis='y', labelsize=tick_size)
ax1.tick_params(axis="x", labelsize=tick_size)
for i in gR_tau:
    plt.plot( fftshift(sim.freq), fftshift( i )/(np.max(i))*3, color="gray" )
plt.xlim([0,30])
fig.tight_layout()
plt.show()


#%%

hR_W = Raman_graph(sim.tiempo, tau1=1.3, tau2=32e-3)

plt.plot(fftshift(sim.freq), fftshift(np.imag(hR_W)))
plt.xlim([0,30])

