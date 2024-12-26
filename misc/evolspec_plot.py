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

#Time
import time

AW, AT, sim, fibra = loader("soliton_gen/sgm", resim = True)
zlocs = np.linspace(0, 300, len(AT))


AT = np.stack(AT)
AW = np.stack(AW)
#%% Plotting

plotcmap(sim, fibra, zlocs, AT, AW, legacy=False, dB=True, wavelength=True,cmap="turbo",
         vlims=[-30,0,-60,0], Tlim=[-50,50], Wlim=[1400,1700], zeros=True, plot_type="both")

plotspecgram2(sim, fibra, AT, Tlim=[-25,25], Wlim=[-25,25], zeros=True)


#%% dB Scale

# Labels y tamaños
cbar_tick_size = 14
tick_size      = 14
m_label_size   = 14
M_label_size   = 15
cmap = "turbo"

Tlim = [-25,25]
Wlim_s = [-25,25]

P_T = Pot(AT)

# Escala dB
P_T = 10*np.log10(P_T) - np.max(10*np.log10(P_T))

# Limites del colorbar
vmin_t = -30
vmax_t = 0
textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)

# Imshow 1
im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                 extent=textent, vmin=vmin_t, vmax=vmax_t)
ax1.tick_params(labelsize=tick_size)
ax1.set_ylabel("Distance (m)", size=m_label_size)
ax1.set_xlabel("Time (ps)", size=m_label_size)
ax1.set_xlim(Tlim)

AT_spec = AT[-1, :]

# x-axis limits
xextent = [sim.tiempo[0], sim.tiempo[-1]]
# Time span
t_span = sim.tiempo[-1] - sim.tiempo[0]
# Sampling rate
t_sampling = len(sim.tiempo) / t_span

# Compute the spectrogram
Pxx, freqs, bins, im2 = ax2.specgram(AT_spec, NFFT=700, noverlap=650, Fs=t_sampling, scale="dB", xextent=xextent, cmap=cmap)

# Adjust the vmin and vmax
vmax_s = 13.7 #np.max(Pxx)
vmin_s = vmax_s + vmin_t
im2.set_clim(vmin=vmin_s, vmax=vmax_s)

freq_zdw = (fibra.omega0 - fibra.w_zdw) / (2 * np.pi)
ax2.plot(xextent, [freq_zdw, freq_zdw], "-.", color="dodgerblue", linewidth=2, label="ZDW = " + str(round(fibra.zdw)) + " nm")

freq_znw = (fibra.omega0 - fibra.w_znw) / (2 * np.pi)
ax2.plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2, label="ZNW = " + str(round(fibra.znw)) + " nm")
ax2.legend(loc="best", prop={'size': 12})

ax2.set_xlabel("Time (ps)", size=m_label_size)
ax2.set_ylabel("Frequency (THz)", size=m_label_size)
ax2.tick_params(labelsize=tick_size)
ax2.set_xlim(Tlim)
ax2.set_ylim(Wlim_s)

# Create a single colorbar for both plots
cbar = fig.colorbar(im1, ax=[ax1, ax2], label='dB', location="right", aspect=20)
cbar.set_label('dB', size=m_label_size)
cbar.ax.tick_params(labelsize=cbar_tick_size)

#plt.savefig("evolspec2.svg")
plt.show()

#%%

# Labels y tamaños
cbar_tick_size = 16
tick_size      = 16
m_label_size   = 16
M_label_size   = 15
cmap = "cmr.ember"

Tlim = [-40,40]
Wlim_s = [-25,25]

P_T = Pot(AT)

# Escala dB
P_T = 10*np.log10(P_T) - np.max(10*np.log10(P_T[0]))

# Limites del colorbar
vmin_t = -30
vmax_t = 0
textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True, sharex=True)

# Imshow 1
im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                 extent=textent, vmin=vmin_t, vmax=vmax_t)
ax1.tick_params(labelsize=tick_size)
ax1.set_ylabel("Distance (m)", size=m_label_size)
#ax1.set_xlabel("Time (ps)", size=m_label_size)
ax1.set_xlim(Tlim)

AT_spec = AT[-1, :]

# x-axis limits
xextent = [sim.tiempo[0], sim.tiempo[-1]]
# Time span
t_span = sim.tiempo[-1] - sim.tiempo[0]
# Sampling rate
t_sampling = len(sim.tiempo) / t_span

# Compute the spectrogram
Pxx, freqs, bins, im2 = ax2.specgram(AT_spec, NFFT=700, noverlap=650, Fs=t_sampling, scale="dB", xextent=xextent, cmap=cmap)

# Adjust the vmin and vmax
vmax_s = 13.7 #np.max(Pxx)
vmin_s = vmax_s + vmin_t
im2.set_clim(vmin=vmin_s, vmax=vmax_s)

freq_zdw = (fibra.omega0 - fibra.w_zdw) / (2 * np.pi)
ax2.plot(xextent, [freq_zdw, freq_zdw], "-.", color="dodgerblue", linewidth=2.5, label="ZDW @ " + str(round(fibra.zdw)) + " nm")

freq_znw = (fibra.omega0 - fibra.w_znw) / (2 * np.pi)
ax2.plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2.5, label="ZNW @ " + str(round(fibra.znw)) + " nm")
ax2.legend(loc="upper left", prop={'size': 12})

ax2.set_xlabel("Time (ps)", size=m_label_size)
ax2.set_ylabel("Frequency (THz)", size=m_label_size)
ax2.tick_params(labelsize=tick_size)
ax2.set_xlim(Tlim)
ax2.set_ylim(Wlim_s)
# Invert the y-axis
ax2.invert_yaxis()


# Create a single colorbar for both plots
cbar = fig.colorbar(im1, ax=[ax1, ax2], label='dB', location="top", aspect=30, pad=0.01)
cbar.set_label('dB', size=m_label_size, labelpad=10)
cbar.ax.tick_params(labelsize=cbar_tick_size)

#plt.savefig("evolspec2.svg")
plt.show()

#%% FLIPPED: Revision according to Reviewer 1, the y-axis in the specgram was inverted.

# Labels y tamaños
cbar_tick_size = 16
tick_size      = 16
m_label_size   = 16
M_label_size   = 15
cmap = "magma"

Tlim = [-25,25]
Wlim_s = [-25,25]

P_T = Pot(AT)

# Escala dB
P_T = 10*np.log10(P_T) - np.max(10*np.log10(P_T[0]))

# Limites del colorbar
vmin_t = -30
vmax_t = 0
textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True, sharex=True)

# Imshow 1
im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                 extent=textent, vmin=vmin_t, vmax=vmax_t)
ax1.tick_params(labelsize=tick_size)
ax1.set_ylabel("Distance (m)", size=m_label_size)
#ax1.set_xlabel("Time (ps)", size=m_label_size)
ax1.set_xlim(Tlim)

AT_spec = AT[-1, :]

# x-axis limits
xextent = [sim.tiempo[0], sim.tiempo[-1]]
# Time span
t_span = sim.tiempo[-1] - sim.tiempo[0]
# Sampling rate
t_sampling = len(sim.tiempo) / t_span

# Compute the spectrogram
Pxx, freqs, bins, im2 = ax2.specgram(AT_spec, NFFT=700, noverlap=650, Fs=t_sampling, scale="dB", xextent=xextent, cmap=cmap)

# Adjust the vmin and vmax
vmax_s = 13.7 #np.max(Pxx)
vmin_s = vmax_s + vmin_t
im2.set_clim(vmin=vmin_s, vmax=vmax_s)


# Manually set the y-ticks and y-tick labels
y_ticks = [20, 10, 0, -10, -20]
y_tick_labels = [-20, -10, 0, 10, 20]
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(y_tick_labels)

freq_zdw = (fibra.omega0 - fibra.w_zdw) / (2 * np.pi)
ax2.plot(xextent, [freq_zdw, freq_zdw], "-.", color="dodgerblue", linewidth=2.5, label="ZDW @ " + str(round(fibra.zdw)) + " nm")

freq_znw = (fibra.omega0 - fibra.w_znw) / (2 * np.pi)
ax2.plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2.5, label="ZNW @ " + str(round(fibra.znw)) + " nm")
ax2.legend(loc="upper left", prop={'size': 12})

ax2.set_xlabel("Time (ps)", size=m_label_size)
ax2.set_ylabel("Frequency (THz)", size=m_label_size)
ax2.tick_params(labelsize=tick_size)
ax2.set_xlim(Tlim)
ax2.set_ylim(Wlim_s)
# Invert the y-axis
ax2.invert_yaxis()

# Create a single colorbar for both plots
cbar = fig.colorbar(im1, ax=[ax1, ax2], label='dB', location="top", aspect=30, pad=0.01)
cbar.set_label('dB', size=m_label_size, labelpad=10)
cbar.ax.tick_params(labelsize=cbar_tick_size)

#plt.savefig("evolspec2.svg")
plt.show()