# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:00:57 2023

@author: d/dt Lucas
"""

import numpy as np
from solvegnlse import SolveNLS, Raman
from solvepcnlse import Solve_pcNLSE
from solvepcGNLSE import Solve_pcGNLSE
from commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, Adapt_Vector_1D
from plotter import plotinst, plotevol, plotspecgram, plotenergia, plotfotones, plt, plot_time
from scipy.signal import find_peaks

#Time
import time

import chime
chime.theme("mario")


#Parametros para la simulación
N = int(2**14)
Tmax = 70 #ps

sim   = Sim(N, Tmax) #Objeto SIM

c = 299792458 * (1e9)/(1e12)

Lambda0= 1542                       #Longitud de onda central (nm)
omega0 = 2*np.pi*c/Lambda0

#Parametros para la fibra
L     = 4000.300#300                      #Lfib:   m
b2    = 0.68e-3                 #Beta2:  ps^2/km
b3    = 0.115e-3                  #Beta3:  ps^3/km
gam   = 2e-3#2.5e-3                   #Gamma:  1/Wkm
gam1  = 0
alph  = 0                        #alpha:  dB/m
TR    = 3e-3*0                   #TR:     fs
fR    = 0.18*0                   #fR:     adimensional (0.18)

fibra = Fibra(L, b2, b3, gam, gam1, alph, Lambda0) #Objeto FIBRA


#Parámetros pulso 1: (Pump)
Lambda1  = Lambda0
amp_1    = 0.94        #Amplitud:  sqrt(W), Po = amp**2
ancho_1  = 600e-3         #Ancho T0:  ps
offset_1 = 0

#Parámetros pulso 2: (Signal)
Lambda2  = 1560#1554
amp_2    = 0.1*amp_1
ancho_2  = 2000e-3
offset_2 = -5*1 


#Diferencia de frecuencias: (En caso de tener dos pulsos con velocidad de grupo distintas)
nu2     = c/Lambda2
nu1     = c/Lambda1
dnu2    = nu2 - nu1

def dark_soliton(time,amplitude,dip,duration,offset,chirp):
  assert 0.0<=dip and dip <= 1.0, f"ERROR: The value of dip is {dip}, but it must be between 0 and 1!!!"
  t_char = dip*((time-offset)/duration-dip*np.sqrt(1-dip**2))

  return (1+0j)*amplitude*(dip*np.tanh(t_char)-1j*np.sqrt(1-dip**2))*SuperGauss(time,1,duration*50,offset,chirp,25)

def simple_ds(time, beta2, gamma, width, maxwindow):
    u = 1/np.sqrt(gamma*width**2/np.abs(beta2))
    squarewell_factor = maxwindow/width - 5
    return u * np.tanh(time/width) * SuperGauss(time,1,width*squarewell_factor,0,0,50)
    

dark_amplitude = np.sqrt(np.abs(b2)/gam/1**2)

darksol = simple_ds(sim.tiempo, fibra.beta2, fibra.gamma, width=ancho_1, maxwindow=Tmax)*1

#Calculamos el pulso inicial
signal = np.sqrt(amp_2)*np.exp( - ( (sim.tiempo - offset_2)/(np.sqrt(2)*ancho_2) )**2  )*np.exp(-2j*np.pi*dnu2*sim.tiempo)
pulso = darksol + signal


# =============================================================================
# signal = np.sqrt(amp_2)*(1/np.cosh(sim.tiempo/ancho_2 + offset_2/ancho_2))*np.exp(-2j*np.pi*dnu2*sim.tiempo)*1
# pump = dark_soliton(sim.tiempo, amplitude=dark_amplitude, dip=1, duration=1, offset=0, chirp=0)*0
# pulso = pump + signal
# pulso = darksol + signal
# #pulso = Two_Pulse(sim.tiempo, amp_1*0, amp_2*1, ancho_1, ancho_2, offset_1, offset_2, dnu2, pulses = "p+s")
# espectro = FT(pulso)
# =============================================================================


N = np.sqrt( gam * amp_1 * ancho_1**2 / np.abs(b2)  )
print(N)

# =============================================================================
# plt.figure()
# plt.plot( fftshift(sim.freq),  fftshift(Pot(espectro)) )
# plt.xlim([-.2,.2])
# plt.show()
# =============================================================================

#%%
b_w = b2 * (2*np.pi*sim.freq) + b3/2 * (2*np.pi*sim.freq)**2
lambda_vec = 299792458 * (1e9)/(1e12) / (sim.freq + omega0/(2*np.pi))
lam = lambda_vec[lambda_vec.argsort()]
b_lam = b_w[lambda_vec.argsort()]
plt.plot(lam,b_lam)

#%% Corriendo la simulación

#---pcgNLSE---
t0 = time.time()
zlocs, A_w, A_t = Solve_pcGNLSE(sim, fibra, pulso, z_locs=100)
#SolveNLS(sim, fibra, pulso, raman=False, z_locs=100)  #
t1 = time.time()
total_n = t1 - t0 #Implementar en Solve_pcGNLSE
print("Time",np.round(total_n/60,2),"(min)")
chime.success()

#%% Plotting

plotevol(sim, fibra, zlocs, A_t, A_w, Tlim=[-50,50], dB=False, wavelength=False, noshow=False, cmap="RdBu")
plotinst(sim, A_t, A_w)

#%% Extra

Tlim = [-20,20]

fig, ax1 = plt.subplots(1,1,figsize=(16, 12), dpi=100, subplot_kw={'aspect': 'equal'})

ax1.set_yticks([int(j) for j in range(int(zlocs[0]),int(zlocs[-1]))])
ax1.set_xticks([int(j) for j in range(int(sim.tiempo[0]),int(sim.tiempo[-1]))])

for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontsize(15)
for tick in ax1.get_xticklines() + ax1.get_yticklines():
    tick.set_markeredgewidth(2)
    tick.set_markersize(6)

im = ax1.imshow( Pot(A_t), cmap="turbo",     interpolation='nearest',origin='lower', extent=[int(sim.tiempo[0]),int(sim.tiempo[-1]),0,int(sim.tiempo[-1])])
ax1.set_xlim(-50,50)

#%% Extra: Adimensional test

#Parametros para la simulación
N_a = int(2**14)
Tmax_a = 200 #ps

sim_a   = Sim(N_a, Tmax_a) #Objeto SIM

c = 299792458 * (1e9)/(1e12)

Lambda0_a= 1600#1550                       #Longitud de onda central (nm)
omega0_a = 2*np.pi*c/Lambda0

#Parametros para la fibra
L_a     = 10#300                      #Lfib:   m
b2_a    = 1                 #Beta2:  ps^2/km
b3_a    = 1                  #Beta3:  ps^3/km
gam_a   = 1                   #Gamma:  1/Wkm
gam1_a  = 0
alph_a  = 0                        #alpha:  dB/m
TR_a    = 3e-3*0                   #TR:     fs
fR_a    = 0.18*0                   #fR:     adimensional (0.18)

fibra_a = Fibra(L_a, b2_a, b3_a, gam_a, gam1_a, alph_a, Lambda0_a) #Objeto FIBRA


#Parámetros pulso 1: (Pump)
Lambda1_a  = Lambda0_a
amp_1a    = 243        #Amplitud:  sqrt(W), Po = amp**2
ancho_1a  = 85e-3         #Ancho T0:  ps
offset_1a = 0
amp_1a    = np.abs(b2)/(gam * ancho_1**2)

#Parámetros pulso 2: (Signal)
Lambda2_a  = 1565#1480
amp_2a    = 0.1
ancho_2a  = 10
offset_2a = -30 


#Diferencia de frecuencias: (En caso de tener dos pulsos con velocidad de grupo distintas)
nu2_a     = c/Lambda2_a
nu1_a     = c/Lambda1_a
dnu2_a    = nu2_a - nu1_a

dnu2_a = -2.1

darksol_a = simple_ds(sim.tiempo, fibra.beta2, fibra.gamma, width=1, maxwindow=Tmax)*1

#Calculamos el pulso inicial
signal_a = np.sqrt(amp_2a)*(1/np.cosh(sim_a.tiempo/ancho_2a + offset_2a/ancho_2a))*np.exp(-2j*np.pi*dnu2_a*sim_a.tiempo)*1
pulso_a = darksol_a + signal_a

plt.figure()
plt.plot(sim_a.tiempo, Pot(pulso_a))
plt.show()

#%% Corriendo la simulación adimensional

#---pcgNLSE---
t0 = time.time()
zlocs_a, A_wa, A_ta = Solve_pcGNLSE(sim_a, fibra_a, pulso_a, z_locs=100)
#SolveNLS(sim, fibra, pulso, raman=False, z_locs=100)  #
t1 = time.time()
total_n = t1 - t0 #Implementar en Solve_pcGNLSE
print("Time",np.round(total_n/60,2),"(min)")
chime.success()

#%% Plotting

plotevol(sim_a, fibra_a, zlocs_a, A_ta, A_wa, Tlim=[-50,50], dB=False, wavelength=False, noshow=False, cmap="turbo")
plotinst(sim_a, A_ta, A_wa)
