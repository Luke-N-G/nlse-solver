# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:00:57 2023

@author: d/dt Lucas
"""


import numpy as np
from common.commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from common.commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, find_k
from common.plotter import plotinst, plotcmap, plotspecgram, plotspecgram2, plotenergia, plotfotones, plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from functools import partial
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

#Time
import time

AW, AT, sim, fibra = loader("soliton_gen/recoil", resim = True)
zlocs = np.linspace(0, 300, len(AT))

#%% Plotting

plotcmap(sim, fibra, zlocs, AT, AW, legacy=False, dB=True, wavelength=True,cmap="turbo",
         vlims=[-30,0,-30,0], Tlim=[-50,50], Wlim=[1400,1700], zeros=True, plot_type="both")

plotspecgram2(sim, fibra, AT, Tlim=[-25,25], Wlim=[-25,25], zeros=True)

#%% Busqueda de máximos, derivadas, etc

c = 299792458 * (1e9)/(1e12)

#Buscar posición y tiempo del máximo en AT^2
def find_max(AT, tiempo):
    # Calculate the absolute square of AT
    AT_abs_square = np.abs(AT)**2

    # Find the index of the maximum value for each position
    max_time_indices = np.argmax(AT_abs_square, axis=1)

    # Get the actual time values
    max_times = tiempo[max_time_indices]

    return max_times

#Ajuste con un spline polinómico
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
def compute_derivative(max_times, zlocs):
    # Fit a spline to the data
    spline = UnivariateSpline(zlocs, max_times, k=2)  # k=3 for cubic spline
    spline.set_smoothing_factor(1)
    # Compute the derivative of the spline
    derivative = spline.derivative()
    # Evaluate the derivative at each position
    derivative_values = derivative(zlocs)
    return derivative_values

def compute_derivative_zd(max_times, zlocs, window_length, polyorder):
    # Apply Savitzky-Golay filter to the data
    smoothed_max_times = savgol_filter(max_times, window_length, polyorder)
    # Compute the differences between consecutive times and positions
    dt = np.diff(smoothed_max_times)
    dz = np.diff(zlocs)
    # Compute the derivative as the ratio of differences
    derivative_values = dt / dz
    return derivative_values



maxs = find_max(AT, sim.tiempo)
vgs  = compute_derivative(maxs, zlocs)
vgs_zd = compute_derivative_zd(maxs, zlocs, 3, 2)


def omega_to_lambda(w, w0): #Función para pasar de Omega a lambda.
    return 2*np.pi*c/(w0+w)

#Reflexión con beta1 del solitón manual (encontrado con la trayectoria de la simulación)
def find_reflection_manual(fib:Fibra, lambda_i, beta1_s): 
    w_i     = 2*np.pi*c/lambda_i - fib.omega0
    #dw_s    = 2*np.pi*c/lambda_s - 2*np.pi*c/fib.lambda0
    shift_c = beta1_s
    c_coef  = -fib.beta3/6 * w_i**3 - fib.beta2/2 * w_i**2 + shift_c * w_i
    coefs   = [fib.beta3/6, fib.beta2/2, -shift_c, c_coef]
    raices  = np.roots(coefs)
    #print(omega_to_lambda(raices,omega0))
    return omega_to_lambda(raices, fib.omega0)

reflection = np.zeros_like(zlocs)
reflection_zd = np.zeros_like(zlocs)
for i in range(len(zlocs)-1):
    reflection[i] = find_reflection_manual(fibra, 1500, vgs[i])[1]
    reflection_zd[i] = find_reflection_manual(fibra, 1500, vgs_zd[i])[1]

spl = UnivariateSpline(zlocs,maxs, k =3)
spl.set_smoothing_factor(0.02)

reflection[-1] = reflection[-2]

#%% dB Scale

# Labels y tamaños
cbar_tick_size = 14
tick_size      = 14
m_label_size   = 14
M_label_size   = 15
cmap = "magma"

Tlim = [-20,10]
Wlim = [1470,1680]

P_T = Pot(AT)

# Escala dB
P_T = 10*np.log10(P_T) - np.max(10*np.log10(P_T))

# Limites del colorbar
vmin_t = -30
vmax_t = 0
vmin_s = -50
vmax_s = 0
textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5), constrained_layout=True, sharey=True)

# Imshow 1
im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                 extent=textent, vmin=vmin_t, vmax=vmax_t)
ax1.plot(spl(zlocs), zlocs, "--w", linewidth=3)
ax1.tick_params(labelsize=tick_size)
ax1.set_ylabel("Distance (m)", size=m_label_size)
ax1.set_xlabel("Time (ps)", size=m_label_size)
ax1.set_xlim(Tlim)

cbar1 = fig.colorbar(im1, ax=ax1, label='Normalized power (dB)', location="bottom", aspect=20 )
cbar1.set_label('Normalized power (dB)', size=m_label_size)
cbar1.ax.tick_params(labelsize=cbar_tick_size)


#Espectro 

lamvec, AL = Adapt_Vector(sim.freq, fibra.omega0, AW)
# Armamos un vector de lambdas lineal (lamvec es no lineal, ya que viene de la freq.)
lamvec_lin = np.linspace(lamvec.min(), lamvec.max(), len(lamvec))
# Interpolamos los datos a nuestra nueva "grilla" lineal
AWs = np.empty_like(AL)
for i in range(AL.shape[0]):
    interp_func = interp1d(lamvec, AL[i, :], kind='next')
    AWs[i, :] = interp_func(lamvec_lin)
P_W = Pot(AWs)
wextent = [lamvec_lin[0], lamvec_lin[-1], zlocs[0], zlocs[-1]]
P_W = 10*np.log10(P_W) - np.max( 10*np.log10(P_W) )

im2 = ax2.imshow(P_W, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                 extent=wextent, vmin=vmin_s, vmax=vmax_s)
ax2.plot(reflection, zlocs, "--w", linewidth=3)
ax2.set_xticks([1500,1550,1600,1650])
ax2.tick_params(labelsize=tick_size)
#Colorbar 2
cbar2 = fig.colorbar(im2, ax=ax2, label='PSD (a.u. dB)', location="bottom", aspect=20 )
cbar2.set_label('PSD (a.u., dB)', size=m_label_size)
cbar2.ax.tick_params(labelsize=cbar_tick_size)

freq_zdw = (fibra.omega0 - fibra.w_zdw)/(2*np.pi)
freq_znw = (fibra.omega0 - fibra.w_znw)/(2*np.pi)

ax2.axvline(x = fibra.zdw, linestyle="-.", color="dodgerblue",  label="ZDW", linewidth=2)
ax2.axvline(x = fibra.znw, linestyle="--", color="crimson", label="ZNW", linewidth=2)

plt.legend(loc="best")

ax2.set_xlim(Wlim)

ax2.set_xlabel("Wavelength (nm)", size=m_label_size)

#plt.subplots_adjust(wspace=.05)
#plt.savefig("recoil.pdf")
plt.show()

