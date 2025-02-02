#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from progressbar import progressbar
from a03_FIELD0_VORTX import velocity_field_Analytical
from a09_PRTCLE_VORTX import maxey_riley_analytic
from a09_PRTCLE_VORTX_NoHistory import maxey_riley_analytic_nohistory

"""
Created on Tue Jan 30 11:17:01 2024
@author: Julio Urizarna-Carasa
"""
#
###############################################################################
############################## Import flow field ##############################
###############################################################################
#
omega_value  = 1.                         # Angular velocity of the field 
vel          = velocity_field_Analytical(omega=omega_value) # Flow field



#
###############################################################################
#################### Define particle and fluid parameters #####################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
rho_p        = 2./3.    # - Particle's density
rho_f        = 1.       # - Fluid's density (water)

# Define scales of the flow field
T_scale      = np.copy(vel.T)  # Timescale of the flow field
L_scale      = np.copy(vel.L)  # Lengthscale of the flow field
U_scale      = np.copy(vel.U)  # Velocityscale of the flow field

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*T_scale)):
nu_f         = 8.917e-7 # Kinematic viscosity of water at 25ºC
rad_p        = np.sqrt(np.array([0.1, 1., 10.])) * \
               np.sqrt(3. * nu_f * T_scale)  # Particle's radius

S_v          = rad_p**2. / ( 3. * nu_f * T_scale )



#
###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                   # Folder where to save data
tini         = 0.   / T_scale                # Initial time
tend         = 10.  / T_scale                # Final time
L            = 101                           # Time nodes
taxis        = np.linspace(tini, tend, L)    # Time axis
dt           = taxis[1] - taxis[0]           # time step


y0           = np.array([1., 0.]) / L_scale  # Initial position (nondimensional)
v0           = vel.get_velocity(y0[0], y0[1], tini)  # Initial velocity (nondimensional)



#
###############################################################################
############ Create classes instances and calculate trajectories###############
###############################################################################
#
particle_dic = dict()
particle_nohistory_dic = dict()
for j in range(0, len(S_v)):
    # Create particle class instances
    particle_dic[j]  = maxey_riley_analytic(j+1, y0, v0, tini, vel,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    particle_nohistory_dic[j] = maxey_riley_analytic_nohistory(j+1, y0, v0,
                                            tini, vel,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    # Calculate trajectories!
    for tt in progressbar(range(1, len(taxis))):
        particle_dic[j].solve(taxis[tt])
        particle_nohistory_dic[j].solve_nohistory(taxis[tt])


#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs   = 6
lw   = 1.2


for j in range(0, len(S_v)):
    fig = plt.figure(j+1, layout='tight', figsize=(2.25, 2.))
    # Normalised trajectory with full history
    pos_vec = np.linalg.norm(particle_dic[j].pos_vec, 2, axis=1)
    plt.plot(taxis, pos_vec, 'r-',
             linewidth=lw, label="with History Term")

    # Normalised trajectory without Basset History Term
    pos_vec_nohistory = np.linalg.norm(particle_nohistory_dic[j].pos_vec, 2, axis=1)
    plt.plot(taxis, pos_vec_nohistory, 'b--',
             linewidth=lw, label="without History Term")

    plt.xlabel('time', fontsize=fs, labelpad=0.25)
    plt.ylabel('Radial distance', fontsize=fs, labelpad=0.25)
    # plt.yscale("log")
    # plt.ylim([1e-5, 1e1])
    plt.tick_params(axis='both', labelsize=fs)
    # plt.xlim([x_left, x_right])
    
    if particle_dic[j].p.R > 1.:
        code = '07010204'
        plt.ylim([0.97, 1.55])
        plt.legend(loc='upper left', fontsize=fs)#bbox_to_anchor=(0.5, -0.21), ncol=2, fontsize=fs)
    elif particle_dic[j].p.R < 1:
        code = '07010203'
        plt.ylim([0.35, 1.03])
        plt.legend(loc='lower left', fontsize=fs)#bbox_to_anchor=(0.5, -0.21), ncol=2, fontsize=fs)
    plt.grid()
    
    filename = code + '-VORTX-RADIST-R=' + str(round(particle_dic[j].p.R, 2)) +\
        '-St=' + str(round(particle_dic[j].p.S, 2)) + '.pdf'
    
    plt.savefig(save_plot_to + filename, format='pdf',
                dpi=400, bbox_inches='tight')
    call(["pdfcrop", save_plot_to + filename, save_plot_to + filename])

plt.show()
