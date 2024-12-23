# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:09:24 2024

@author: d/dt Lucas
"""

import numpy as np
from common.commonfunc import SuperGauss, Soliton, Two_Pulse, Sim, Fibra, Pot, fftshift, FT, IFT
from common.commonfunc import find_shift, find_chirp, Adapt_Vector, saver, loader, ReSim, find_k
from common.plotter import plotinst, plotcmap, plotspecgram, plotspecgram2, plotenergia, plotfotones, plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from functools import partial

cmap = "cmr.ember"

#%% Individual cases: OK

# Load data for sgm
AW_sgm, AT_sgm, sim, fibra_sgm = loader("soliton_gen/sgm", resim=True)
zlocs_sgm = np.linspace(0, 300, len(AT_sgm))
AT_sgm = np.stack(AT_sgm)
AW_sgm = np.stack(AW_sgm)

# Load data for sg1
AW_sg1, AT_sg1, sim, fibra_sg1 = loader("soliton_gen/sg1", resim=True)
zlocs_sg1 = np.linspace(0, 300, len(AT_sg1))
AT_sg1 = np.stack(AT_sg1)
AW_sg1 = np.stack(AW_sg1)

# Load data for znw
AW_znw, AT_znw, sim, fibra_znw = loader("soliton_gen/IPC/raman-znw/1450znw", resim=True)
zlocs_znw = np.linspace(0, 300, len(AT_znw))
AT_znw = np.stack(AT_znw)
AW_znw = np.stack(AW_znw)

# Load data for nr
AW_nr, AT_nr, sim, fibra_nr = loader("soliton_gen/IPC/raman-znw/1480nr", resim=True)
zlocs_nr = np.linspace(0, 300, len(AT_nr))
AT_nr = np.stack(AT_nr)
AW_nr = np.stack(AW_nr)


# Common plotting parameters
cbar_tick_size = 16
tick_size = 16
m_label_size = 16
M_label_size = 15
#cmap = "cmr.ember"
Tlim = [-30, 30]
Wlim_s = [-25, 25]

def plot_data(AT, zlocs, fibra, title, Tlim = [-30,30], Wlim=[-25,25]):
    P_T = Pot(AT)
    P_T = 10 * np.log10(P_T) - np.max(10 * np.log10(P_T[0]))
    vmin_t = -30
    vmax_t = 0
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True, sharex=True)

    # Imshow 1
    im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                     extent=textent, vmin=vmin_t, vmax=vmax_t)
    ax1.tick_params(labelsize=tick_size)
    ax1.set_ylabel("Distance (m)", size=m_label_size)
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
    vmax_s = 13.7
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
    ax2.set_ylim(Wlim)

    # Create a single colorbar for both plots
    cbar = fig.colorbar(im1, ax=[ax1, ax2], label='dB', location="top", aspect=30, pad=0.01)
    cbar.set_label('dB', size=m_label_size, labelpad=10)
    cbar.ax.tick_params(labelsize=cbar_tick_size)
    #plt.savefig(str(title)+".png", dpi=800)
    plt.show()
    
def plot_data2(AT, zlocs, fibra, title, Tlim=[-30, 30], Wlim=[-25, 25], side_by_side=False):
    P_T = Pot(AT)
    P_T = 10 * np.log10(P_T) - np.max(10 * np.log10(P_T[0]))
    vmin_t = -30
    vmax_t = 0
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

    if side_by_side:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True, sharex=True)

    # Imshow 1
    im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                     extent=textent, vmin=vmin_t, vmax=vmax_t)
    ax1.tick_params(labelsize=tick_size)
    ax1.set_ylabel("Distance (m)", size=m_label_size)
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
    vmax_s = 13.7
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
    ax2.set_ylim(Wlim)

    # Create a colorbar for each plot
    cbar1 = fig.colorbar(im1, ax=ax1, label='dB', location="top", aspect=30, pad=0.01)
    cbar1.set_label('dB', size=m_label_size, labelpad=10)
    cbar1.ax.tick_params(labelsize=cbar_tick_size)

    cbar2 = fig.colorbar(im2, ax=ax2, label='dB', location="top", aspect=30, pad=0.01)
    cbar2.set_label('dB', size=m_label_size, labelpad=10)
    cbar2.ax.tick_params(labelsize=cbar_tick_size)

    plt.savefig(str(title)+"_ZNW.png", dpi=800)

    plt.show()

# Plot for sgm
#plot_data(AT_sgm, zlocs_sgm, fibra_sgm, "sgm")

# Plot for sg1
plot_data2(AT_sg1, zlocs_sg1, fibra_sg1, "sg1", Wlim = [-25,20], side_by_side=True)

# Plot for znw
plot_data2(AT_znw, zlocs_znw, fibra_znw, "znw", Tlim=[-30,45], Wlim = [-25,20], side_by_side=True)

# Plot for nr
#plot_data(AT_nr, zlocs_nr, fibra_nr, "nr")

#%%

#%% Lambda sweep: OK

# Load data from the specified directory
AW_1455, AT1, sim, fibra_1455 = loader("soliton_gen/IPC/lambda_sweep/1455", resim=True)
AW_1470, AT2, sim_1470, fibra_1470 = loader("soliton_gen/IPC/lambda_sweep/1470", resim=True)
AW_1480, AT3, sim_1480, fibra_1480 = loader("soliton_gen/IPC/lambda_sweep/1480", resim=True)
AW_1490, AT4, sim_1490, fibra_1490 = loader("soliton_gen/IPC/lambda_sweep/1490", resim=True)

zlocs = np.linspace(0, 300, len(AT1))

# Labels y tamaños
cbar_tick_size = 16
tick_size      = 16
m_label_size   = 16
M_label_size   = 18
#cmap = "cmr.ember"

Tlim = [-20, 20]

# Stack AT and AW
AT1 = np.stack(AT1)
AT2 = np.stack(AT2)
AT3 = np.stack(AT3)
AT4 = np.stack(AT4)
AT_cases = [AT1, AT2, AT3, AT4]
AW_cases = np.stack([AW_1455, AW_1470, AW_1480, AW_1490])
zlocs_cases = [zlocs, zlocs, zlocs, zlocs]

lambda_d_values = [1455, 1470, 1480, 1490]

fig, axs = plt.subplots(1, 4, figsize=(50, 6), constrained_layout=True, sharey=True)

for i, (AT, zlocs, lambda_d) in enumerate(zip(AT_cases, zlocs_cases, lambda_d_values)):
    P_T = Pot(AT)
    P_T = np.array(P_T)  # Ensure P_T is a numpy array
    P_T = 10 * np.log10(P_T) - np.max(10 * np.log10(P_T[0]))
    vmin_t = -30
    vmax_t = 0
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

    im = axs[i].imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                       extent=textent, vmin=vmin_t, vmax=vmax_t)
    axs[i].tick_params(labelsize=tick_size)
    axs[i].set_xlabel("Time (ps)", size=m_label_size)
    axs[i].set_xlim(Tlim)
    axs[i].set_title(f"$\lambda_d$ = {lambda_d} nm", size=M_label_size)

axs[0].set_ylabel("Distance (m)", size=m_label_size)

# Create a single colorbar for all plots
cbar = fig.colorbar(im, ax=axs, label='dB', location="right", aspect=30, pad=0.01)
cbar.set_label('dB', size=m_label_size, labelpad=10)
cbar.ax.tick_params(labelsize=cbar_tick_size)
#plt.savefig("lambdasweep_nospec.png", dpi=800)
plt.show()

#%% Lambda Sweep + specgram: OK

# Load data from the specified directory
AW_1455, AT1, sim, fibra_1455 = loader("soliton_gen/IPC/lambda_sweep/1455", resim=True)
AW_1470, AT2, sim_1470, fibra_1470 = loader("soliton_gen/IPC/lambda_sweep/1470", resim=True)
AW_1480, AT3, sim_1480, fibra_1480 = loader("soliton_gen/IPC/lambda_sweep/1480", resim=True)
AW_1490, AT4, sim_1490, fibra_1490 = loader("soliton_gen/IPC/lambda_sweep/1490", resim=True)

zlocs = np.linspace(0, 300, len(AT1))

# Labels y tamaños
cbar_tick_size = 16
tick_size = 16
m_label_size = 16
M_label_size = 15
#cmap = "cmr.ember"

Tlim = [-25, 25]
Wlim = [-25, 25]

# Stack AT and AW
AT1 = np.stack(AT1)
AT2 = np.stack(AT2)
AT3 = np.stack(AT3)
AT4 = np.stack(AT4)
AT_cases = [AT1, AT2, AT3, AT4]
AW_cases = np.stack([AW_1455, AW_1470, AW_1480, AW_1490])
zlocs_cases = [zlocs, zlocs, zlocs, zlocs]

lambda_d_values = [1455, 1470, 1480, 1490]

fig, axs = plt.subplots(2, 4, figsize=(50, 12), constrained_layout=True, sharey='row', sharex='col')

for i, (AT, zlocs, lambda_d) in enumerate(zip(AT_cases, zlocs_cases, lambda_d_values)):
    P_T = Pot(AT)
    P_T = np.array(P_T)  # Ensure P_T is a numpy array
    P_T = 10 * np.log10(P_T) - np.max(10 * np.log10(P_T[0]))
    vmin_t = -30
    vmax_t = 0
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

    im = axs[0, i].imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                          extent=textent, vmin=vmin_t, vmax=vmax_t)
    axs[0, i].tick_params(labelsize=tick_size)
    axs[0, i].set_xlim(Tlim)
    axs[0, i].set_title(f"$\lambda_d$ = {lambda_d} nm", size=M_label_size)

    AT_spec = AT[-1, :]
    xextent = [sim.tiempo[0], sim.tiempo[-1]]
    t_span = sim.tiempo[-1] - sim.tiempo[0]
    t_sampling = len(sim.tiempo) / t_span

    Pxx, freqs, bins, im2 = axs[1, i].specgram(AT_spec, NFFT=700, noverlap=650, Fs=t_sampling, scale="dB", xextent=xextent, cmap=cmap)
    vmax_s = 13.7
    vmin_s = vmax_s + vmin_t
    im2.set_clim(vmin=vmin_s, vmax=vmax_s)

    freq_zdw = (fibra_1455.omega0 - fibra_1455.w_zdw) / (2 * np.pi)
    axs[1, i].plot(xextent, [freq_zdw, freq_zdw], "-.", color="dodgerblue", linewidth=2.5, label="ZDW @ " + str(round(fibra_1455.zdw)) + " nm")

    freq_znw = (fibra_1455.omega0 - fibra_1455.w_znw) / (2 * np.pi)
    axs[1, i].plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2.5, label="ZNW @ " + str(round(fibra_1455.znw)) + " nm")
    if i == 0:
        axs[1, i].legend(loc="upper left", prop={'size': 12})
        axs[1, i].set_ylabel("Frequency (THz)", size=m_label_size)

    axs[1, i].set_xlabel("Time (ps)", size=m_label_size)
    axs[1, i].tick_params(labelsize=tick_size)
    axs[1, i].set_xlim(Tlim)
    axs[1, i].set_ylim(Wlim)

axs[0, 0].set_ylabel("Distance (m)", size=m_label_size)

# Create a single colorbar for all plots
cbar = fig.colorbar(im, ax=axs, label='dB', location="right", aspect=30, pad=0.01)
cbar.set_label('dB', size=m_label_size, labelpad=10)
cbar.ax.tick_params(labelsize=cbar_tick_size)
#plt.savefig("lambdasweep.png", dpi=800)
plt.show()

#%% to sweep: OK

# Load data from the specified directory
AW_0_5, AT1, sim, fibra_0_5 = loader("soliton_gen/IPC/t0_sweep/0,5", resim=True)
AW_2, AT2, sim_2, fibra_2 = loader("soliton_gen/IPC/t0_sweep/2", resim=True)
AW_4_5, AT3, sim_4_5, fibra_4_5 = loader("soliton_gen/IPC/t0_sweep/4,5", resim=True)
AW_5_75, AT4, sim_5_75, fibra_5_75 = loader("soliton_gen/IPC/t0_sweep/5,75", resim=True)

zlocs = np.linspace(0, 300, len(AT1))

# Labels y tamaños
cbar_tick_size = 16
tick_size = 16
m_label_size = 16
M_label_size = 15
#cmap = "cmr.ember"

Tlim = [-25, 25]
Wlim = [-25, 25]

# Stack AT and AW
AT1 = np.stack(AT1)
AT2 = np.stack(AT2)
AT3 = np.stack(AT3)
AT4 = np.stack(AT4)
AT_cases = [AT1, AT2, AT3, AT4]
AW_cases = np.stack([AW_0_5, AW_2, AW_4_5, AW_5_75])
zlocs_cases = [zlocs, zlocs, zlocs, zlocs]

lambda_d_values = ["0.5", "2.0", "4.5", "5.7"]

fig, axs = plt.subplots(2, 4, figsize=(50, 12), constrained_layout=True, sharey='row', sharex='col')

for i, (AT, zlocs, lambda_d) in enumerate(zip(AT_cases, zlocs_cases, lambda_d_values)):
    P_T = Pot(AT)
    P_T = np.array(P_T)  # Ensure P_T is a numpy array
    P_T = 10 * np.log10(P_T) - np.max(10 * np.log10(P_T[0]))
    vmin_t = -30
    vmax_t = 0
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

    im = axs[0, i].imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                          extent=textent, vmin=vmin_t, vmax=vmax_t)
    axs[0, i].tick_params(labelsize=tick_size)
    axs[0, i].set_xlim(Tlim)
    axs[0, i].set_title(f"$\\tau_d$ = {lambda_d} ps", size=M_label_size)

    AT_spec = AT[-1, :]
    xextent = [sim.tiempo[0], sim.tiempo[-1]]
    t_span = sim.tiempo[-1] - sim.tiempo[0]
    t_sampling = len(sim.tiempo) / t_span

    Pxx, freqs, bins, im2 = axs[1, i].specgram(AT_spec, NFFT=700, noverlap=650, Fs=t_sampling, scale="dB", xextent=xextent, cmap=cmap)
    vmax_s = 13.7
    vmin_s = vmax_s + vmin_t
    im2.set_clim(vmin=vmin_s, vmax=vmax_s)

    freq_zdw = (fibra_0_5.omega0 - fibra_0_5.w_zdw) / (2 * np.pi)
    axs[1, i].plot(xextent, [freq_zdw, freq_zdw], "-.", color="dodgerblue", linewidth=2.5, label="ZDW @ " + str(round(fibra_0_5.zdw)) + " nm")

    freq_znw = (fibra_0_5.omega0 - fibra_0_5.w_znw) / (2 * np.pi)
    axs[1, i].plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2.5, label="ZNW @ " + str(round(fibra_0_5.znw)) + " nm")
    if i == 0:
        axs[1, i].legend(loc="upper left", prop={'size': 12})
        axs[1, i].set_ylabel("Frequency (THz)", size=m_label_size)

    axs[1, i].set_xlabel("Time (ps)", size=m_label_size)
    axs[1, i].tick_params(labelsize=tick_size)
    axs[1, i].set_xlim(Tlim)
    axs[1, i].set_ylim(Wlim)

axs[0, 0].set_ylabel("Distance (m)", size=m_label_size)

# Create a single colorbar for all plots
cbar = fig.colorbar(im, ax=axs, label='dB', location="right", aspect=30, pad=0.01)
cbar.set_label('dB', size=m_label_size, labelpad=10)
cbar.ax.tick_params(labelsize=cbar_tick_size)
#plt.savefig("t0sweep.png", dpi=800)
plt.show()

#%% Po sweep: OK

# Load data from the specified directory
AW_0_5, AT1, sim, fibra_0_5 = loader("soliton_gen/IPC/p0_sweep/0-5", resim=True)
AW_2, AT2, sim_2, fibra_2 = loader("soliton_gen/IPC/p0_sweep/1-5", resim=True)
AW_4_5, AT3, sim_4_5, fibra_4_5 = loader("soliton_gen/IPC/p0_sweep/3", resim=True)
AW_5_75, AT4, sim_5_75, fibra_5_75 = loader("soliton_gen/IPC/p0_sweep/4", resim=True)

zlocs = np.linspace(0, 300, len(AT1))

# Labels y tamaños
cbar_tick_size = 16
tick_size = 16
m_label_size = 16
M_label_size = 15
#cmap = "cmr.ember"

Tlim = [-25, 25]
Wlim = [-25, 25]

# Stack AT and AW
AT1 = np.stack(AT1)
AT2 = np.stack(AT2)
AT3 = np.stack(AT3)
AT4 = np.stack(AT4)
AT_cases = [AT1, AT2, AT3, AT4]
AW_cases = np.stack([AW_0_5, AW_2, AW_4_5, AW_5_75])
zlocs_cases = [zlocs, zlocs, zlocs, zlocs]

lambda_d_values = ["5", "15", "30", "40"]

fig, axs = plt.subplots(2, 4, figsize=(50, 12), constrained_layout=True, sharey='row', sharex='col')

for i, (AT, zlocs, lambda_d) in enumerate(zip(AT_cases, zlocs_cases, lambda_d_values)):
    P_T = Pot(AT)
    P_T = np.array(P_T)  # Ensure P_T is a numpy array
    P_T = 10 * np.log10(P_T) - np.max(10 * np.log10(P_T[0]))
    vmin_t = -30
    vmax_t = 0
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]

    im = axs[0, i].imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                          extent=textent, vmin=vmin_t, vmax=vmax_t)
    axs[0, i].tick_params(labelsize=tick_size)
    axs[0, i].set_xlim(Tlim)
    axs[0, i].set_title(f"$P_d$ = {lambda_d} ps", size=M_label_size)

    AT_spec = AT[-1, :]
    xextent = [sim.tiempo[0], sim.tiempo[-1]]
    t_span = sim.tiempo[-1] - sim.tiempo[0]
    t_sampling = len(sim.tiempo) / t_span

    Pxx, freqs, bins, im2 = axs[1, i].specgram(AT_spec, NFFT=700, noverlap=650, Fs=t_sampling, scale="dB", xextent=xextent, cmap=cmap)
    vmax_s = 13.7
    vmin_s = vmax_s + vmin_t
    im2.set_clim(vmin=vmin_s, vmax=vmax_s)

    freq_zdw = (fibra_0_5.omega0 - fibra_0_5.w_zdw) / (2 * np.pi)
    axs[1, i].plot(xextent, [freq_zdw, freq_zdw], "-.", color="dodgerblue", linewidth=2.5, label="ZDW @ " + str(round(fibra_0_5.zdw)) + " nm")

    freq_znw = (fibra_0_5.omega0 - fibra_0_5.w_znw) / (2 * np.pi)
    axs[1, i].plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2.5, label="ZNW @ " + str(round(fibra_0_5.znw)) + " nm")
    if i == 0:
        axs[1, i].legend(loc="upper left", prop={'size': 12})
        axs[1, i].set_ylabel("Frequency (THz)", size=m_label_size)

    axs[1, i].set_xlabel("Time (ps)", size=m_label_size)
    axs[1, i].tick_params(labelsize=tick_size)
    axs[1, i].set_xlim(Tlim)
    axs[1, i].set_ylim(Wlim)

axs[0, 0].set_ylabel("Distance (m)", size=m_label_size)

# Create a single colorbar for all plots
cbar = fig.colorbar(im, ax=axs, label='dB', location="right", aspect=30, pad=0.01)
cbar.set_label('dB', size=m_label_size, labelpad=10)
cbar.ax.tick_params(labelsize=cbar_tick_size)
plt.savefig("p0sweep.png", dpi=800)
plt.show()