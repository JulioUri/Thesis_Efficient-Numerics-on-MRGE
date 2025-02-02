#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
from subprocess import call
from a03_FIELD0_DATA1 import velocity_field_Faraday1
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_STOKS import particle_stokes

"""
Created on Tue Jan 30 17:19:11 2024
@author: Julio Urizarna-Carasa
"""
#
###############################################################################
############################## Import flow field ##############################
###############################################################################
#
vel          = velocity_field_Faraday1()  # Flow field



#
###############################################################################
#################### Define particle and fluid parameters #####################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
rho_p        = 4./3.    # - Particle's density
rho_f        = 1.       # - Fluid's density (water)

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
tini         = 0.   / T_scale                # Initial time
tend         = 5.   / T_scale                # Final time
L            = 1001                          # Time nodes
taxis        = np.linspace(tini, tend, L)    # Time axis
dt           = taxis[1] - taxis[0]           # time step


y0           = np.array([0.01, 0.01]) / L_scale  # Initial position (nondimensional)
v0           = vel.get_velocity(y0[0], y0[1], tini)  # Initial velocity (nondimensional)



#
###############################################################################
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Define Uniform grid [0,1):
N           = np.copy(L)  # Number of nodes
xi_fd_v     = np.linspace(0., 1., int(N))[:-1]

# Control constant (Koleva 2005)
c           = 20

# Logarithm map to obtain QUM
x_fd_v      = -c * np.log(1.0 - xi_fd_v)

# Parameter for Prasath et al.
nodes_dt    = 21 # Advised by Prasath et al.

#
###############################################################################
######################### Create classes instances ############################
###############################################################################
#
particle_dic = dict()
particle_nohistory_dic = dict()
for j in range(0, len(S_v)):
    # Create particle class instances
    particle_dic[j]  = maxey_riley_imex(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    
    particle_nohistory_dic[j] = particle_stokes(j+1, y0, v0,
                                            tini, vel,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    
    
    # Calculate trajectories!
    for tt in progressbar(range(1, len(taxis))):
        particle_dic[j].update()
        particle_nohistory_dic[j].update(dt)



#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs   = 5
lw   = 1
ms   = 5


#########
# Plots #
#########

for k in range(0, len(S_v)):
    fig = plt.figure(k+1, layout='tight', figsize=(2.5, 2.15))
    
    pos_vec = np.linalg.norm(particle_dic[k].pos_vec - particle_dic[k].pos_vec[0], 2, axis=1) * L_scale #/ \
                # np.linalg.norm(particle_dic[k].pos_vec[-1], 2)
    plt.plot(taxis, pos_vec,
             color='red', linewidth=lw, label="with History Term")
    
    pos_nohistory_vec = np.linalg.norm(particle_nohistory_dic[k].pos_vec - particle_nohistory_dic[k].pos_vec[0], 2, axis=1) * L_scale #/ \
               # np.linalg.norm(particle_dic[k].pos_vec[-1], 2)
    plt.plot(taxis, pos_nohistory_vec,
             color='blue', linewidth=lw, label="without History Term")
    
    plt.xlabel('Time', fontsize=fs, labelpad=0.25)
    plt.ylabel('Radial distance from initial position', fontsize=fs, labelpad=0.25)
    plt.tick_params(axis='both', labelsize=fs)
    plt.legend(loc='upper left', fontsize=fs)
    plt.ylim([-0.002, 0.035]) #1.2])
    plt.xlim([-0.02, 0.5])
    plt.grid()
    
    if particle_dic[k].p.R < 1.:
        filename = '07010603-DATA1-RDIST-R=' + str(round(particle_dic[k].p.R, 2)) +\
                   '-S=' + str(round(particle_dic[k].p.S, 2)) + '.pdf'
    elif particle_dic[k].p.R > 1.:
        filename = '07010604-DATA1-RDIST-R=' + str(round(particle_dic[k].p.R, 2)) +\
                   '-S=' + str(round(particle_dic[k].p.S, 2)) + '.pdf'
               
    plt.savefig(save_plot_to + filename,
                format='pdf', dpi=400, bbox_inches='tight')
    
    call(["pdfcrop", save_plot_to + filename, save_plot_to + filename])
    
    # plt.close()
plt.show()