# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:14:05 2024

@author: d/dt Lucas
"""

import numpy as np
import matplotlib.pyplot as plt
from common.commonfunc import Energia, num_fotones, saver, loader, ReSim, FT, IFT, fftshift, Pot, Fibra, Adapt_Vector, Sim
from common.plotter import plotenergia, plotevol, plotinst, plotspecgram, plot_time, plotcmap
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import cmasher as cmr

AW, AT, sim, fibra = loader("soliton_gen/sgm", resim = True)
AW = np.stack(AW)
AT = np.stack(AT)
zlocs = np.linspace(0, 300, len(AT))

#%%

plotcmap(sim, fibra, zlocs, AT, AW, legacy=False, dB=True, wavelength=True,
         vlims=[-30,0,-60,0], Tlim=[-25,25], Wlim=[1400,1700], zeros=True, cmap="turbo")

plotinst(sim, fibra, AT, AW, wavelength=True)

#%% Masking Test

lam, AL = Adapt_Vector(sim.freq, fibra.omega0, AW[0])

masklam_start = fibra.zdw
masklam_end   = fibra.znw

maskw_start = fibra.lambda_to_omega( masklam_start )/(2*np.pi)
maskw_end = fibra.lambda_to_omega( masklam_end )/(2*np.pi)

mask_start_idx = np.argmin( np.abs(sim.freq - maskw_start)  )
mask_end_idx = np.argmin( np.abs(sim.freq - maskw_end)  )

#Esto corta al AW entre los índices que cumplan lo de arriba
AW_cut = AW[-1][:mask_start_idx]
AW_cut = np.append(AW_cut, np.zeros_like(sim.freq[mask_start_idx:mask_end_idx]) )
AW_cut = np.append(AW_cut, AW[-1][mask_end_idx:])


plt.figure()
plt.plot( sim.tiempo, Pot(IFT(AW_cut)), label = "Masked spectrum")
plt.plot( sim.tiempo, Pot(IFT(AW[-1])), label = "Full signal", alpha = .5)
plt.legend(loc="best")
plt.title("Filter comparison")
plt.grid(True,alpha=.3)
plt.show()


peaks, _ = find_peaks( Pot(IFT(AW_cut)), prominence = 50, height=0)

plt.figure()
plt.plot(sim.tiempo, Pot(IFT(AW_cut)))
plt.plot(sim.tiempo[peaks], Pot(IFT(AW_cut))[peaks], "x")
plt.title("Peaks")
plt.grid(True,alpha=.3)
plt.show()

#%% Análisis de pulsos a la salida (Ajuste con secante hiperbólica)

def soliton_fit(T, amplitude, center, width, offset):
    carrier = np.sqrt(amplitude)*( 1/np.cosh( (T - center)/width) ) + offset*0
    return np.abs( carrier )**2


def soliton_number(fib:Fibra, sim:Sim, AW):
    
    prominence  = 50  #Prominencia, para hallar picos
    window_size = 100 #Número de puntos alrededor de cada pico
    z_index     = -1  #A que z analizamos (se podría pasar como variable de función)
    
    #Buscamos freq. donde enmascarar
    mask_i     = fibra.lambda_to_omega( fibra.zdw )/(2*np.pi)
    mask_f     = fibra.lambda_to_omega( fibra.znw )/(2*np.pi)
    
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
        
        #------TESTING GROUND--------
        #Estudiamos el espectro
        
        #Armamos un vector que sea = 0 en todos lados, menos en el pulso de estudio
        pulse_window = np.zeros_like(sim.tiempo, dtype=complex)
        pulse_window[peak-window_size:peak+window_size] = window
        
        #Transformamos a espectro
        spectral_window = IFT(pulse_window)
        
        #Buscamos el máximo espectral
        max_peak_index = np.argmax( Pot(spectral_window) )
        
        #Frecuencia central del pulso
        pulse_centerfreq = sim.freq[max_peak_index]*-1
        
        print(fibra.omega_to_lambda(pulse_centerfreq))
        plt.figure()
        plt.plot(fftshift(sim.freq), fftshift(Pot(spectral_window)))
        plt.plot(sim.freq[max_peak_index], Pot(spectral_window)[max_peak_index], "x")
        plt.show()
        
        pulse_gamma = fibra.gamma_w(pulse_centerfreq)
        pulse_beta2 = fibra.beta2_w(pulse_centerfreq)
        
        #Parámetros para hallar el N del solitón
        pulse_amp   = Pot(AT_mask[peak])
        pulse_width = popt[2]
        soliton_order = np.sqrt(pulse_amp * pulse_width**2 * pulse_gamma / np.abs(pulse_beta2) )
        
        print(popt)
        plt.figure()
        plt.plot(t_window, Pot(window), ".", label="Datos")
        plt.plot(t_window, soliton_fit(t_window, *popt), label="Ajuste, N: "+str(soliton_order))
        plt.legend(loc="best")
        plt.grid(True, alpha=.3)
        plt.show()
        

        #Si el orden está entre 0.5 y 1.5, contamos
        if 0.5 <= soliton_order <= 1.5*2:
            soliton_count += 1

        #if rss < 1e3:
        #    soliton_count += 1
    
    return soliton_count

n_solitons = soliton_number(fibra, sim, AW)

#Solucionar la función de arriba
#find_solitons(AT, AW, 100, mask_start, mask_end)

#%% Plots generales

'''
plt.figure()
plt.plot(sim.tiempo, np.abs(AT[0])**2)
plt.show()
'''

#plotinst(sim, AT, AW)
plotevol(sim, fibra, zlocs, AT, AW, Tlim=[-25,25], Wlim=[1400,1700],wavelength=True, dB=False, cmap="turbo")
#plot_time(sim, fibra, zlocs, AT, Tlim=[-25,25], dB=False)
#plotspecgram(sim, fibra, AT[-1], Tlim = [-20,20], Wlim = [-20,20], zeros=True,save="specgram.svg")


from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
def format_func(value, tick_number):
    return f'$10^{{{int(value/10)}}}$'


P_T = Pot(np.stack(AT))
P_W = Pot(fftshift(np.stack(AW),axes=1))
toplot_T = 20*np.log10(P_T)
toplot_W = 20*np.log10(P_W)


fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(toplot_T, cmap="turbo", aspect="auto", interpolation='bilinear', origin="lower",
                 extent=[sim.tiempo[0],sim.tiempo[-1],zlocs[0],zlocs[-1]], vmin=-20, vmax=60)
#ax.set_aspect(2)
cbar = fig.colorbar(im, ax=ax, label='Interactive colorbar' )
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

#Multiplot test

fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8.76,5))

im1 = ax1.imshow(toplot_T, cmap="turbo", aspect="auto", interpolation='bilinear', origin="lower",
                 extent=[sim.tiempo[0],sim.tiempo[-1],zlocs[0],zlocs[-1]], vmin=-20, vmax=60)

cbar1 = fig.colorbar(im1, ax=ax1, label='Interactive colorbar', location="bottom", aspect=50 )
cbar1.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

im2 = ax2.imshow(toplot_W, cmap="turbo", aspect="auto", interpolation='bilinear', origin="lower",
                 extent=[sim.tiempo[0],sim.tiempo[-1],zlocs[0],zlocs[-1]], vmin=-20, vmax=60)

cbar2 = fig.colorbar(im2, ax=ax2, label='Interactive colorbar', location="bottom", aspect=50 )
cbar2.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))


plt.figure()
plt.imshow(np.abs(np.stack(AT))**2, aspect="equal")
plt.colorbar()
plt.show()


#%% FFT recortado

AT_s1 = np.zeros(len(AT[0]), dtype=complex)
AT_s0 = np.zeros(len(AT[0]), dtype=complex)

#Original soliton
AT_s0[9280:9400] = AT[-1][9280:9400]

#Nova soliton
AT_s1[8380:8520] = AT[-1][8380:8520]

#Without Nova  
#AT_s1[0:8380] = AT[-1][0:8380]        
#AT_s1[8520:-1] = AT[-1][8520:-1]

AW_s1 = FT(AT_s1)
AW_s0 = FT(AT_s0)

lamvec_full , Alam_full =Adapt_Vector(sim.freq, fibra.omega0, FT(AT[-1]))
lamvec_s1 , Alam_s1 = Adapt_Vector(sim.freq, fibra.omega0, AW_s1)
lamvec_s0 , Alam_s0 = Adapt_Vector(sim.freq, fibra.omega0, AW_s0)

lambda_s0 = lamvec_s0[ np.argmax( Pot(Alam_s0) ) ]
lambda_s1 = lamvec_s0[ np.argmax( Pot(Alam_s1) ) ]

def gammaw(freq, fib:Fibra):
    return fib.gamma + fib.gamma1 * (2*np.pi*freq)

def beta1w(freq, fib:Fibra):
    return fib.beta2 * (2*np.pi*freq) + fib.beta3  * (2*np.pi*freq)**2 / 2

def beta2w(freq, fib:Fibra):
    return fib.beta2 + fib.beta3  * (2*np.pi*freq)

def N2(gamma, beta2, To, Po):
    return np.sqrt( gamma * Po * To**2 / np.abs(beta2) )

c_n = 299792458 * (1e9)/(1e12)
nu_0  = c_n / 1600
nu_s0 = c_n / lambda_s0
nu_s1 = c_n / lambda_s1
dnu_s0 = -nu_0 + nu_s0
dnu_s1 = -nu_0 + nu_s1

print("s1 beta2: "+str(beta2w(dnu_s1, fibra)) )
print("s1 gamma: "+str(gammaw(dnu_s1, fibra)) )
print("s1 N: "+str( N2( gammaw(dnu_s1, fibra), beta2w(dnu_s1, fibra), 85e-3, np.max( Pot(AT_s1) )   ) ) )
print("------------------------------------")
print("s0 beta2: "+str(beta2w(dnu_s0, fibra)) )
print("s0 gamma: "+str(gammaw(dnu_s0, fibra)) )
print("s0 N: "+str( N2( gammaw(dnu_s0, fibra), beta2w(dnu_s0, fibra) , 85e-3, np.max( Pot(AT_s0) )   ) ) )


plt.figure()
plt.plot(lamvec_s0, Pot(Alam_s0) )
plt.plot(lamvec_s1, Pot(Alam_s1) )
plt.grid(True, alpha=.3)
plt.xlabel("Wavelength (nm)")
plt.ylabel("$|A(\omega)|^2$")
plt.show()

plt.figure()
plt.plot(lamvec_full, Pot(Alam_full) )
plt.grid(True, alpha=.3)
plt.xlabel("Wavelength (nm)")
plt.ylabel("$|A(\omega)|^2$")
plt.show()

plt.figure()
plt.plot(sim.tiempo, Pot(AT[-1]) )
plt.grid(True, alpha=.3)
plt.xlabel("Time (ps)")
plt.ylabel("$|A(T)|^2$")
plt.show()

#%% Código ajuste

def soliton_fit(T, amplitude, center, width, offset):
    carrier = np.sqrt(amplitude)*( 1/np.cosh( (T - center)/width) ) + offset
    return np.abs( carrier )**2

#-----Soliton original----
s0_guess = [330, 9.8, 0.3, 0]
s0_par, s0_cov = curve_fit(soliton_fit, xdata=sim.tiempo, ydata=Pot(AT_s0), p0=s0_guess)
N_s0 = N2( gammaw(dnu_s0, fibra), beta2w(dnu_s0, fibra), s0_par[2], s0_par[0] )

plt.figure()
plt.plot(sim.tiempo,Pot(AT_s0),".")
plt.plot(sim.tiempo, soliton_fit(sim.tiempo, s0_par[0], s0_par[1], s0_par[2], s0_par[3] ))
plt.title("")
plt.xlabel("Time (ps)")
plt.ylabel("Power (W)")
plt.xlim([s0_par[1]-5*s0_par[2] , s0_par[1]+5*s0_par[2]])
plt.grid(True, alpha=.3)
plt.show()


#------Soliton nuevo-----
s1_guess = [330, 2.3, 0.3, 0]
s1_par, s1_cov = curve_fit(soliton_fit, xdata=sim.tiempo, ydata=Pot(AT_s1), p0=s1_guess)
N_s1 = N2( gammaw(dnu_s1, fibra), beta2w(dnu_s1, fibra), s1_par[2], s1_par[0] )

plt.figure()
plt.plot(sim.tiempo,Pot(AT_s1),".")
plt.plot(sim.tiempo, soliton_fit(sim.tiempo, s1_par[0], s1_par[1], s1_par[2], s1_par[3] ))
plt.title("New soliton fit")
plt.xlabel("Time (ps)")
plt.ylabel("Power (W)")
plt.xlim([s1_par[1]-5*s1_par[2] , s1_par[1]+5*s1_par[2]])
plt.grid(True, alpha=.3)
plt.show()

print("s1 beta2: "+str(beta2w(dnu_s1, fibra)) )
print("s1 gamma: "+str(gammaw(dnu_s1, fibra)) )
print("s1 N: "+str( N_s1 ) )
print("------------------------------------")
print("s0 beta2: "+str(beta2w(dnu_s0, fibra)) )
print("s0 gamma: "+str(gammaw(dnu_s0, fibra)) )
print("s0 N: "+str( N_s0 ) )

#%% Análisis colisión

AT_sc = np.zeros( len(AT[0]), dtype=complex )
AT_sc[8380:8480] = AT[38][8380:8480] 

plt.figure()
plt.plot(sim.tiempo, Pot(AT_sc) )
plt.show()

AW_sc = FT(AT_sc)

lamvec_sc , Alam_sc = Adapt_Vector(sim.freq, fibra.omega0, AW_sc)

lambda_sc = lamvec_sc[ np.argmax( Pot(Alam_sc) ) ]
nu_sc = c_n / lambda_sc
dnu_sc = -nu_0 + nu_sc

#------Soliton colisión-----
sc_guess = [600, 2.3, 0.3, 0]
sc_par, sc_cov = curve_fit(soliton_fit, xdata=sim.tiempo, ydata=Pot(AT_sc), p0=sc_guess)
N_sc = N2( gammaw(dnu_sc, fibra), beta2w(dnu_sc, fibra), sc_par[2], sc_par[0] )

plt.figure()
plt.plot(sim.tiempo,Pot(AT_sc),".")
plt.plot(sim.tiempo, soliton_fit(sim.tiempo, sc_par[0], sc_par[1], sc_par[2], sc_par[3] ))
plt.title("Fit during collision")
plt.xlabel("Time (ps)")
plt.ylabel("Power (W)")
plt.xlim([sc_par[1]+5*sc_par[2] , sc_par[1]-5*sc_par[2]])
plt.grid(True, alpha=.3)
plt.show()

#%% Análisis N signal

lambda_signal = 1560
nu_signal = c_n/lambda_signal
dnu_signal = nu_signal - nu_0

N_signal = N2( gammaw(-dnu_signal,fibra), beta2w(-dnu_signal,fibra), 2, 30 )

lambda_signal_vec = np.linspace(1555,1580,50)
dnu_signal_vec    = c_n / lambda_signal_vec - nu_0
N_signal_vec      = N2( gammaw(-dnu_signal_vec,fibra), beta2w(-dnu_signal_vec,fibra), 2, 30 )

'''
plt.figure()
plt.plot(lambda_signal_vec, N_signal_vec)
plt.xlabel("Wavelength (nm)")
plt.ylabel("N")
plt.title("Signal N")
plt.grid(True, alpha=.3)
plt.show()
'''

to_vec = [1,2,3,4,5,6]
colors = plt.cm.Blues(np.linspace(0,1,len(to_vec)+1))

plt.figure()
for i in to_vec:
    N_signal_vec = N2( gammaw(-dnu_signal_vec,fibra), beta2w(-dnu_signal_vec,fibra), i, 30 )
    plt.plot(lambda_signal_vec, N_signal_vec, color=colors[i], label="$t_0$ = "+str(i)+" ps")
plt.xlabel("Longitud de onda (nm)", size=20)
plt.ylabel("$N$", size=20)
#plt.title("Signal N (width)")
plt.grid(True, alpha=.3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("N_lambda.svg")
plt.show()

Po_vec = [10,20,30,40,50,60]
colors = plt.cm.Reds(np.linspace(0,1,len(Po_vec)+1))

plt.figure()
for j, i in enumerate(Po_vec):
    N_signal_vec = N2( gammaw(-dnu_signal_vec,fibra), beta2w(-dnu_signal_vec,fibra), 2, i )
    plt.plot(lambda_signal_vec, N_signal_vec, color=colors[j], label="Po = "+str(i)+" W")
plt.xlabel("Wavelength (nm)")
plt.ylabel("N")
plt.title("Signal N (power)")
plt.grid(True, alpha=.3)
#plt.yticks(np.arange(0,13+1,1))
plt.legend(loc="best")
plt.show()

znw_vec = [2000,1700,1650,1555,1450,1200,400]
colors = plt.cm.RdBu(np.linspace(0,1,len(znw_vec)+1))

plt.figure()
for j,i in enumerate(znw_vec):
    
    c = 299792458 * (1e9)/(1e12)
    omega0 = 2*np.pi*c/fibra.lambda0
    lambda_znw = i
    w_znw = 2*np.pi*c/lambda_znw
    gam1_znw = -fibra.gamma/(w_znw - omega0)
    fibra_znw = Fibra(fibra.L, fibra.beta2, fibra.beta3, fibra.gamma, gam1_znw, fibra.alpha, fibra.lambda0)
    
    N_signal_vec = N2( gammaw(-dnu_signal_vec, fibra_znw), beta2w(-dnu_signal_vec,fibra_znw), 2, 30 )
    #print("ZNW: "+str(i)+", N = "+str(N_signal_vec[0]))
    print("ZNW: "+str(i)+", gamma1: "+str(gam1_znw))
    plt.plot(lambda_signal_vec, N_signal_vec, color=colors[j], label="ZNW = "+str(i)+" nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("N")
plt.title("Signal N (znw)")
plt.grid(True, alpha=.3)
#plt.yticks(np.arange(0,13+1,1))
plt.legend(loc="best")
plt.show()

#%% Reflection and transmission analysis (Temporal Reflection of an optical pulse... JOSAB 2022, J. Zhang)

def Omega_p(Tr, beta21, z, T1):
    return -8 * Tr * np.abs(beta21) * z / (15 * T1**4) 

def delta_r(omega_p0, beta21, beta22, delta_i):
    return 2*omega_p0*beta21/beta22 - delta_i    

lambda_signal_0 = 1480
nu_signal_0 = c_n/lambda_signal_0
dnu_signal_0 = (nu_signal_0 - nu_0)

Tr = 3e-3
beta21 = beta2w(0, fibra)
beta22 = beta2w(dnu_signal_0,fibra)

z_aprox  = 120   #Aprox position of collision
T1       = 85e-3 #Soliton width
omega_p0 = Omega_p(Tr, beta21, z_aprox, 85e-3)
d_r      = delta_r(omega_p0, beta21, beta22, dnu_signal_0)

lambda_reflection = c_n / (d_r + nu_0)

print("Estimated reflection wavelength: "+str(np.trunc(lambda_reflection))+" nm")

#RIFS N. Linale JQE 2021 "Revisiting soliton dynamics"

def Omega_pc(Tr, gamma0, Po, z, T1):
    return -8 * Tr * gamma0 * Po * z / (15 * T1**2)

omega_pc0= Omega_pc(Tr, fibra.gamma, 243, z_aprox, 85e-3)
d_rc      = delta_r(omega_pc0, beta21, beta22, dnu_signal_0)
lambda_reflection_pc = c_n / (d_rc + nu_0)

print("Estimated reflection wavelength pcGNLSE: "+str(np.trunc(lambda_reflection_pc))+" nm")

#%% Transmission coefficient for ZNW (Temporal reflection and refraction with ZNW... OL 2023, A. Sparapani)

def T_coef(beta1, beta2, gamma_t, Po, DeltaT):
    w_r = -2*beta1/beta2
    w_p = (-beta1 + np.sqrt(beta1**2 - 2*np.abs(beta2)*gamma_t*Po) ) / beta2
    w_m = (-beta1 - np.sqrt(beta1**2 - 2*np.abs(beta2)*gamma_t*Po) ) / beta2
    Gamma_p = w_p * np.exp(1j*w_p*DeltaT) * (1 + w_m / w_r) / (w_p - w_m)
    Gamma_m = w_m * np.exp(1j*w_m*DeltaT) * (1 + w_p / w_r) / (w_m - w_p)
    T_c = np.abs( Gamma_p + Gamma_m )**(-2)
    print("w_p: "+str(w_p))
    print("w_m: "+str(w_m))
    print("w_r: "+str(w_r))
    print("Gamma_p = "+str(Gamma_p))
    print("Gamma_m = "+str(Gamma_m))
    return T_c

lambda_znw = 1650
w_znw = 2*np.pi*c/lambda_znw
gam1_n = -fibra.gamma/(w_znw - omega0)
fibra_n = Fibra(fibra.L, fibra.beta2, fibra.beta3, fibra.gamma, gam1_n, fibra.alpha, fibra.lambda0)

lambda_signal_0 = 1480
nu_signal_0 = c_n/lambda_signal_0
dnu_signal_0 = nu_signal_0 - nu_0

gamma_t = 2*np.real( np.sqrt(fibra.gamma * gammaw(dnu_signal_0, fibra_n) * (2*np.pi*nu_0 + 2*np.pi*nu_signal_0)/(2*np.pi*nu_0),dtype=complex))
beta1_t = beta1w(nu_signal_0, fibra_n)
beta2_t = beta2w(nu_signal_0, fibra_n)

print("gamma_t: "+str(gamma_t))
print("beta1_t: "+str(beta1_t))
print("beta2_t: "+str(beta2_t))

T_c = T_coef(beta1_t, beta2_t, gamma_t, 243, 85e-3)

print("Transmission Coefficient: "+str(T_c) )

#%% Energy analysis without Raman

AW_nr, AT_nr, sim_nr, fibra_nr = loader("soliton_gen/noraman", resim = True)
zlocs_nr = np.linspace(0, 300, len(AT_nr))


'''
plt.figure()
plt.plot(np.abs(AT_nr[-1][7900:8100])**2)
plt.show()

plt.figure()
plt.plot(np.abs(AT[-1])**2)
plt.show()
'''

R_energy_nr = np.sum(  np.abs(AT_nr[-1][0:7900])**2   ) 
T_energy_nr = np.sum(  np.abs(AT_nr[-1][8100:-1])**2  )

R_energy = np.sum(  np.abs(AT[-1][0:9280])**2   ) 
T_energy = np.sum(  np.abs(AT[-1][9400:-1])**2  )


print("----Without Raman----")
print("Reflected energy: "+str(R_energy_nr))
print("Transmitted energy: "+str(T_energy_nr))
print("T/R: "+str(T_energy_nr/R_energy_nr))

print("-----With Raman-----")
print("Reflected energy: "+str(R_energy))
print("Transmitted energy: "+str(T_energy))
print("T/R: "+str(T_energy/R_energy))

#%% Soliton analysis without Raman

AW_nr2, AT_nr2, sim_nr2, fibra_nr2 = loader("soliton_gen/sgnr", resim = True)
zlocs_nr2 = np.linspace(0, 500, len(AT_nr2))

AT_nrsc = np.zeros( len(AT_nr[0]), dtype=complex )
AT_nrsc[6100:6300] = AT_nr2[-1][6100:6300] 

AW_nrsc = FT(AT_nrsc)

lamvec_nrsc , Alam_nrsc = Adapt_Vector(sim_nr2.freq, fibra_nr2.omega0, AW_nrsc)

lambda_nrsc = lamvec_nrsc[ np.argmax( Pot(Alam_nrsc) ) ]
nu_nrsc = c_n / lambda_nrsc
dnu_nrsc = -nu_0 + nu_nrsc

#------Sech Fit-----
nrsc_guess = [23, -17, 0.3, 0]
nrsc_par, nrsc_cov = curve_fit(soliton_fit, xdata=sim_nr2.tiempo, ydata=Pot(AT_nrsc), p0=nrsc_guess)
N_nrsc = N2( gammaw(dnu_nrsc, fibra_nr2), beta2w(dnu_nrsc, fibra_nr2), nrsc_par[2], nrsc_par[0] )


plt.figure()
plt.plot(sim_nr2.tiempo, Pot(AT_nr2[-1]) )
plt.xlabel("Time (ps)")
plt.ylabel("Power (W)")
plt.title("Output without Raman - $\lambda_s = 1485~$nm, $L = 500~$m")
plt.grid(True, alpha=.3)
plt.show


plt.figure()
plt.plot(sim_nr2.tiempo,Pot(AT_nrsc),".")
plt.plot(sim_nr2.tiempo, soliton_fit(sim_nr2.tiempo, nrsc_par[0], nrsc_par[1], nrsc_par[2], nrsc_par[3] ))
plt.plot(0,0,"w.", label="N = "+str(round(N_nrsc,3)))
plt.title("Fit Without Raman")
plt.xlabel("Time (ps)")
plt.ylabel("Power (W)")
plt.xlim([nrsc_par[1]+5*nrsc_par[2] , nrsc_par[1]-5*nrsc_par[2]])
plt.grid(True, alpha=.3)
plt.legend(loc="best")
plt.show()




