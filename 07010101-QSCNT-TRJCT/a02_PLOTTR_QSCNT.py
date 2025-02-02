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
rho_p        = 2. / 3.   # - Particle's density
rho_f        = 1.        # - Fluid's density (water)

# Define scales of the flow field
T_scale      = np.copy(vel.T)  # Timescale of the flow field
L_scale      = np.copy(vel.L)  # Lengthscale of the flow field
U_scale      = np.copy(vel.U)  # Velocityscale of the flow field

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*T_scale)):
nu_f         = 8.917e-7 # Kinematic viscosity of water at 25ºC
rad_p        = np.sqrt(np.array([0.01, 0.1, 1., 10.])) * \
               np.sqrt(3. * nu_f * T_scale)  # Particle's radius

S_v          = rad_p**2. / ( 3. * nu_f * T_scale )



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
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Control constant (Koleva 2005)
c           = 20

# Define number of time Chebyshev nodes in each time interval
nodes_dt    = 21 # Advised by Prasath et al. (2019)



#
###############################################################################
############ Create classes instances and calculate trajectories ##############
###############################################################################
#
particle_dic = dict()
for j in range(0, len(S_v)):
    particle_dic[j]  = maxey_riley_relaxing(j+1, y0, v0, tini,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    # Calculate trajectories!
    for tt in progressbar(range(1, len(taxis))):
        particle_dic[j].solve(taxis[tt])
        particle_dic[j].solve_nohistory(taxis[tt])



#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs   = 6
lw   = 1.2

for j in range(0, len(rad_p)):
    fig = plt.figure(j+1, layout='tight', figsize=(2.25, 2.))
    # Normalised trajectory with full history
    pos_vec = np.linalg.norm(particle_dic[j].pos_vec, 2, axis=1) / \
               np.linalg.norm(particle_dic[j].pos_vec[-1], 2)
    plt.plot(taxis, pos_vec, 'r-',
             linewidth=lw, label="with History Term")

    # Normalised trajectory without Basset History Term
    pos_vec_nohistory = np.linalg.norm(particle_dic[j].pos_vec_nohistory, 2, axis=1) / \
               np.linalg.norm(particle_dic[j].pos_vec[-1], 2)
    plt.plot(taxis, pos_vec_nohistory, 'b--',
             linewidth=lw, label="without History Term")

    plt.xlabel('time', fontsize=fs, labelpad=0.25)
    plt.ylabel('Relative distance', fontsize=fs, labelpad=0.25)
    # plt.yscale("log")
    # plt.ylim([1e-5, 1e1])
    plt.tick_params(axis='both', labelsize=fs)
    plt.legend(loc='lower right', fontsize=fs)#bbox_to_anchor=(0.5, -0.21), ncol=2, fontsize=fs)
    # plt.xlim([x_left, x_right])
    plt.ylim([-0.2, 2.2])
    plt.grid()
    
    filename = '07010101-QSCNT-TRJCT-R=' + str(round(particle_dic[j].p.R, 2)) +\
        '-St=' + str(round(particle_dic[j].p.S, 2)) + '.pdf'
    
    plt.savefig(save_plot_to + filename, format='pdf',
                dpi=400, bbox_inches='tight')
    call(["pdfcrop", save_plot_to + filename, save_plot_to + filename])

plt.show()
