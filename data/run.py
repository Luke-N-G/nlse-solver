# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:43:33 2023

@author: d/dt Lucas
"""


import numpy as np
from solvers.solvegnlse import SolveNLS, Raman
from solvers.solvepcGNLSE import Solve_pcGNLSE
from common.commonfunc import SuperGauss, Soliton, Sim, Fibra, Pot, fftshift, FT, IFT, find_shift, find_chirp
from common.plotter import plotinst, plotevol, plotcmap, plotenergia, plotfotones, plt
from scipy.signal import find_peaks




#Parametros para la simulación
ventana = 200 #ps
N_r = int(2**14)
dt_r = ventana/N_r #1e-3
Tmax_r = 200 #ps

c = 299792458 * (1e9)/(1e12)

lambda_r= 1550                       #Longitud de onda central (nm)
omega0_r = 2*np.pi*c/lambda_r

#Parametros para la fibra
L_r     = 500*0.01                       #Lfib:   m
b2_r    = -21 *1e-3                 #Beta2:  ps^2/km
b3_r    = 9.39*1e-5                  #Beta3:  ps^3/km
gam_r   = 0.01560*1                    #Gamma:  1/Wkm
gam1_r  = gam_r/omega0_r*0
alph_r  = 0                          #alpha:  dB/*m
TR_r    = 3e-3*0                       #TR:     fs
fR_r    = 0.18                       #fR:     adimensional (0.18)

#Parámetros pulso gaussiano:
amp_r    = 10                        #Amplitud:  sqrt(W), Po = amp**2
ancho_r  = .5                        #Ancho T0:  ps
offset_r = 0
chirp_r  = 1
orden_r  = 1

#Cargo objetos con los parámetros:
#sim_r   = Sim(N_r,dt_r)
sim_r   = Sim(N_r, Tmax_r)
fibra_r = Fibra(L_r, b2_r, b3_r, gam_r, gam1_r, alph_r, lambda_r,TR_r, fR_r)

#Calculamos el pulso inicial
pulso_r = SuperGauss(sim_r.tiempo, amp_r, ancho_r, offset_r, chirp_r, orden_r)

#soliton_r = Soliton(sim_r.tiempo, ancho_r, fibra_r.beta2, fibra_r.gamma, orden = 1)


#%% Corriendo la simulación

zlocs_r, A_wr, A_tr = Solve_pcGNLSE(sim_r, fibra_r, pulso_r, z_locs=100)
#zlocs_nr, A_wnr, A_tnr = SolveNLS(sim_r, fibra_r, soliton_r, raman=False, z_locs=500)


#%% Plotting

plotinst(sim_r, A_tr, A_wr, Tlim=[-2.5,2.5] ,Wlim=[-50,50])

plot


#%% Extra

'''
#Función Delta W: Expresión analítica del frequency shift dado por Raman en un solitón fundamental.
def delta_w(z, b2, To, Tr):
    return -8*np.abs(b2)*Tr*z/(15*To**4)
'''
'''
plt.plot(zlocs_r, delta_w(zlocs_r, b2_r, ancho_r, TR_r))
plt.title("$\Delta \omega$")
plt.show()
'''

'''
#Comparación entre el corrimiento analítico, aproximado y completo
dw_raman   = find_shift(zlocs_r, sim_r.freq, A_wr) 
dw_noraman = find_shift(zlocs_nr, sim_r.freq, A_wnr)

plt.plot( zlocs_r, dw_raman,"-", label="Respuesta Raman completa", color="grey",markersize=2 )
plt.plot( zlocs_nr, dw_noraman,"-", label="Aproximación con $T_R$", color="orangered")
plt.plot(zlocs_r, delta_w(zlocs_r, b2_r, ancho_r, TR_r), label="Solución analítica")
plt.title("$\Delta \omega$")
plt.xlabel("$z$ (m)")
plt.grid(alpha=.2)
plt.legend(loc="best",handlelength=1)
plt.show()
'''


'''
#Chirp para distintos z (Los guarda para hacer un GIF)
prev1 = 0
for i in range( int(len(zlocs_r)/10) ):
    chirp1 = i*10
    #chirp2 = -i
    #plt.plot(sim_r.tiempo, find_chirp(sim_r.tiempo, A_tr[20]) , label="20")
    #plt.plot(sim_r.tiempo, find_chirp(sim_r.tiempo, A_tr[200]), label="130")
    plt.plot(sim_r.tiempo, find_chirp(sim_r.tiempo, A_tr[chirp1]), label=str(chirp1))
    #plt.plot(sim_r.tiempo, find_chirp(sim_r.tiempo, A_tr[chirp2]),  label=str(chirp2))
    plt.xlim([-3,3])
    plt.ylim([-10,10])
    plt.legend(loc="upper right")
    plt.savefig("chirpgif/"+str(prev1)+".png")
    plt.show()
    prev1=prev1+1
'''
'''
#plt.plot(sim_r.tiempo, find_chirp(sim_r.tiempo, pulso_r))
plt.plot(sim_r.tiempo, find_chirp(sim_r.tiempo, A_tr[50]))
plt.grid(alpha=.3)
plt.xlim([-2,2])
plt.show()
'''

'''
#Corrimiento para distintos valores de N
N_val = [1e4, 2e4, 5e4, 7e4, 1e5, 2**14]
dw = np.zeros(len(N_val), dtype=object)

for index_N, N_d in enumerate(N_val):
    print(index_N)
    Tmax_d      = 100
    sim_d     = Sim(int(N_d), Tmax_d)
    soliton_d = Soliton(sim_d.tiempo, ancho_r, fibra_r.beta2, fibra_r.gamma, orden = 1)
    zlocs_d, A_wd, A_td = SolveNLS(sim_d, fibra_r, soliton_d, raman=True, z_locs=100)
    peaks = np.zeros( len(zlocs_d) , dtype=int)
    for index_z, z_d in enumerate(zlocs_d):
        #peaks[index_z] = find_peaks( Pot( fftshift(A_wd[index_z]) ) , height=1000, prominence=1000)[0]
        peaks[index_z] = np.argmax( Pot( fftshift(A_wd[index_z]) ) )
    dw[index_N] = fftshift(2*np.pi*sim_d.freq)[peaks]
    plt.plot( fftshift(2*np.pi*sim_d.freq), Pot(fftshift(A_wd[-1]))/max(Pot(fftshift(A_wd[-1]))) )
plt.xlim([-10,10])
plt.show()
    
plt.plot(zlocs_d, delta_w(zlocs_d, b2_r, ancho_r, TR_r) )
for i in range( len(dw) ):
    plt.plot(zlocs_d, dw[i])
plt.show()
'''
'''
N_val = [1e4,2e4] #5e4,7e4,1e5]


for index_N, N_d in enumerate(N_val):
    print(index_N)
    Tmax_d = 100
    sim_d = Sim(int(N_d),Tmax_d)
    soliton_d = Soliton(sim_d.tiempo, ancho_r, fibra_r.beta2, fibra_r.gamma, orden = 1)
    zlocs_dr, A_wdr, A_tdr = SolveNLS(sim_d, fibra_r, soliton_d, raman=True, z_locs=100)
    zlocs_dnr, A_wdnr, A_tdnr = SolveNLS(sim_d, fibra_r, soliton_d, raman=False, z_locs=100)
    dw_dr   = find_shift(zlocs_dr, sim_r.freq, A_wdr) 
    dw_dnr = find_shift(zlocs_dnr, sim_r.freq, A_wdnr)
    plt.plot( zlocs_dr, dw_dr,"-", label="Respuesta Raman completa", color="grey",markersize=2 )
    plt.plot( zlocs_dnr, dw_dnr,"-", label="Aproximación con $T_R$", color="orangered")
    plt.title("$\Delta \omega$")
    plt.xlabel("$z$ (m)")
    plt.grid(alpha=.2)
    plt.legend(loc="best",handlelength=1)
    plt.show()
'''

'''
dw_center = find_shift(zlocs_r, sim_r.freq, A_wr)
for i in range( int(len(zlocs_r)) ):
    plt.plot(fftshift(sim_r.freq), Pot(fftshift(A_wr[i])))
    plt.xlim( [-1, 1])
    plt.savefig("chirpgif/"+str(i)+".png")
    plt.show()
'''