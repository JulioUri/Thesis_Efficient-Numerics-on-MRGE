#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from progressbar import progressbar
from a03_FIELD0_QSCNT import velocity_field_Quiescent
from a09_PRTCLE_QSCNT import maxey_riley_relaxing

"""
Created on Tue Jan 30 17:56:36 2024
@author: Julio Urizarna-Carasa
"""

#
###############################################################################
############################## Import flow field ##############################
###############################################################################
#
vel          = velocity_field_Quiescent()   # Quiescent (static) flow field



#
###############################################################################
#################### Define particle and fluid parameters #####################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
# - Left plot
rho_p        = np.array([0.01,   1., 5.]) # - Particle's density
rho_f        = 1.                         # - Fluid's density (water)

# Define scales of the flow field
T_scale      = np.copy(vel.T)  # Timescale of the flow field
L_scale      = np.copy(vel.L)  # Lengthscale of the flow field
U_scale      = np.copy(vel.U)  # Velocityscale of the flow field

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*T_scale)):
nu_f         = 8.917e-7 # Kinematic viscosity of water at 25ºC
rad_p        = np.sqrt(1.) * np.sqrt(3. * nu_f * T_scale)  # Particle's radius



#
###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                   # Folder where to save data
y0           = np.array([0., 0.]) / L_scale  # Initial position (nondimensional)
v0           = np.array([1., 0.]) / U_scale  # Initial velocity (nondimensional)
tini         = 0.   / T_scale                # Initial time
tend         = 15.  / T_scale                # Final time
L            = 1501                          # Time nodes
taxis        = np.linspace(tini, tend, L)    # Time axis
dt           = taxis[1] - taxis[0]           # time step



#
###############################################################################
############ Create classes instances and calculate trajectories ##############
###############################################################################
#
particle_dic = dict()
for j in range(0, len(rho_p)):
    particle_dic[j]  = maxey_riley_relaxing(j+1, y0, v0, tini,
                                            particle_density    = rho_p[j],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p,
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    # Calculate trajectories!
    for tt in progressbar(range(1, len(taxis))):
        particle_dic[j].solve(taxis[tt])



#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs = 6
lw = 1.2



###################
# Trajectory plot #
###################

fig1 = plt.figure(1, layout='tight', figsize=(2.5, 2.15))
for j in range(0, len(rho_p)):
    pos_vec = np.linalg.norm(particle_dic[j].pos_vec, 2, axis=1)
    plt.plot(taxis, pos_vec, linewidth=lw,
             label=chr(946) + " = " + str(round(particle_dic[j].p.beta, 2)))
plt.xlabel('time', fontsize=fs, labelpad=0.25)
plt.ylabel('distance', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, fontsize=fs)
plt.grid()

plt.savefig(save_plot_to + '06010101-QSCNT-TRJCT.pdf', format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + '06010101-QSCNT-TRJCT.pdf', save_plot_to + '06010101-QSCNT-TRJCT.pdf'])



#################
# Velocity plot #
#################

fig2 = plt.figure(2, layout='tight', figsize=(2.5, 2.15))
for j in range(0, len(rho_p)):
    vel_vec = np.linalg.norm(particle_dic[j].vel_vec, 2, axis=1)
    plt.plot(taxis, vel_vec, linewidth=lw,
             label=str(chr(946)) + " = " + str(round(particle_dic[j].p.beta, 2)))   
plt.xlabel('time', fontsize=fs, labelpad=0.25)
plt.ylabel('velocity', fontsize=fs, labelpad=0.25)
plt.yscale("log")
plt.ylim([1e-5, 1e1])
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, fontsize=fs)
plt.grid()

plt.savefig(save_plot_to + '06010101-QSCNT-VLCTY.pdf', format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + '06010101-QSCNT-VLCTY.pdf', save_plot_to + '06010101-QSCNT-VLCTY.pdf'])


plt.show()
