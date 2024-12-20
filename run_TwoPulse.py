# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:00:57 2023

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

Lambda0= 1600                    #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0


#Parámetros pulso 1:
Lambda1  = Lambda0
amp_1    = 243                   #Amplitud:  sqrt(W), Po = amp**2
ancho_1  = 85e-3                 #Ancho T0:  ps
offset_1 = 0


#Parámetros pulso 2:
Lambda2  = 1480
amp_2    = 20#30
ancho_2  = 5750e-3
offset_2 = 20 


#Parametros para la fibra
L     = 300                   #Lfib:   m
b2    = -4.4e-3*1                  #Beta2:  ps^2/km
b3    = 0.13e-3*1                  #Beta3:  ps^3/km
gam   = 2.5e-3*1                   #Gamma:  1/Wkm

lambda_znw = 1650
w_znw = 2*np.pi*c/lambda_znw
gam1 = -gam/(w_znw - omega0)*1

alph  = 0                        #alpha:  dB/m
TR    = 3e-3*0                   #TR:     fs
fR    = 0.18*0                   #fR:     adimensional (0.18)

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
zlocs, AW, AT = Solve_pcGNLSE(sim, fibra, pulso, z_locs=300)
t1 = time.time()

total_n = t1 - t0 #Implementar en Solve_pcGNLSE
print("Time",np.round(total_n/60,2),"(min)")
chime.success()

#saver(AW, AT, sim, fibra, "1470", f'{[Lambda1, amp_1, ancho_1, offset_1, Lambda2, amp_2, ancho_2, offset_2] = }')


#%%

plotinst(sim, fibra, AT, AW, dB=False , wavelength=True, zeros=True, end=-1)

plotinst(sim, fibra, AT, AW, dB=False, wavelength=True, zeros=True, end=210)

plotspecgram(sim, fibra, AT, zeros=True)

plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=True, Tlim=[-30,30], Wlim=[1400,1700],
          vlims=[-30,0,-30,0], zeros=True,plot_type="both", cmap="magma")

#%% Extra1

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
    spline = UnivariateSpline(zlocs, max_times, k=3)  # k=3 for cubic spline
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


plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=True, Tlim=[-30,30], Wlim=[1400,1700],
          vlims=[-20,50,0,120], zeros=True, plot_type="time")
#plt.plot(maxs, zlocs, "--w")
plt.plot(spl(zlocs), zlocs, "--w")
plt.show()

plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=True, Tlim=[-30,30], Wlim=[1400,1700],
          vlims=[-20,50,0,120], zeros=True, plot_type="freq")
plt.plot(reflection, zlocs, "--b")
plt.show()

#%% Extra2


#Para guardar en loop con i el parametro variado
#j = str(i).replace(".", "," )
# =============================================================================
#     
#     plotcmap(sim, fibra, zlocs, AT, AW, Tlim=[-50,50], dB=True, wavelength=True, save=savedic+str(j), zeros=True, noshow=True)
#     plotspecgram(sim, fibra, AT[-1], save=savedic+str(j)+"_specgram", dB=False, zeros=True, noshow=True)
#     
# ============================================================================
#------- NO OLVIDARSE DE CAMBIAR LOS OTROS PARAMETROS! -------
#saver(AW,AT,sim,fibra,savedic+str(j), f'{[Lambda1, amp_1, ancho_1, offset_1, Lambda2, amp_2, ancho_2, offset_2] = }')

#Po = 10W, To = 2.1ps, lambda_i = 1500 nm
# =============================================================================
# reflect_wave_t = [1525.95643739, 1528.68813412, 1530.99614865, 1535.59401619,
#        1537.67139725, 1538.91257145, 1538.91257145, 1540.35459015,
#        1540.97034682, 1541.38002821, 1541.38002821, 1542.40112176,
#        1542.60477423, 1543.41735052]
# 
# z_s = [267.2,280.6,294.8,309.8,322.5,338.2,352.4,368.1,390.5,
#        408.4,433.1,456.3,472.0,492.2]
# =============================================================================

#Po = 20W, To = 3 ps, lambda_i = 1500 nm
# =============================================================================
# reflect_wave_t = [1527.63785688, 1531.41528825, 1535.38575916, 1542.19727487,
#        1546.03250148, 1548.01155099, 1551.66612959, 1555.30091614,
#        1559.21611957, 1561.90664954]
# 
# z_s = [233,239,242.5,248.9,256.2,268.2,277.2,289.2,302.9,315.8]
# =============================================================================


soliton_wave_t = [1598.17938852, 1598.09266555, 1597.66182899, 1597.15103508,
       1596.64717061, 1596.52449369, 1596.48078477]

z_s = [286.3,304.4,323.6,344.4,374.4,425.6,494.9]

plotcmap(sim, fibra, zlocs, AT, AW, wavelength=True, dB=True, Tlim=[-30,30], Wlim=[1400,1700],
          vlims=[-20,50,0,120], zeros=True)
plt.plot(soliton_wave_t,z_s, color="white")