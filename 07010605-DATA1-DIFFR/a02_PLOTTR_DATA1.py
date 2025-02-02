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
rho_p        = np.array([0., 2./3., 4./3., 2.])    # - Particle's density
rho_f        = 1.       # - Fluid's density (water)

# Define scales of the flow field
T_scale      = np.copy(vel.T)  # Timescale of the flow field
L_scale      = np.copy(vel.L)  # Lengthscale of the flow field
U_scale      = np.copy(vel.U)  # Velocityscale of the flow field

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*T_scale)):
nu_f         = 8.917e-7 # Kinematic viscosity of water at 25ºC
rad_p        = np.sqrt(np.geomspace(0.01, 100, 101)) * \
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


y0           = np.array([0.02, 0.01]) / L_scale  # Initial position (nondimensional)
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
traj1_diff     = np.array([])
traj2_diff     = np.array([])
traj3_diff     = np.array([])
traj4_diff     = np.array([])
particle1_dic = dict()
particle1_nohistory_dic = dict()
particle2_dic = dict()
particle2_nohistory_dic = dict()
particle3_dic = dict()
particle3_nohistory_dic = dict()
particle4_dic = dict()
particle4_nohistory_dic = dict()
for j in progressbar(range(0, len(S_v))):
    # Create particle class instances
    particle1_dic[j]  = maxey_riley_imex(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p[0],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    
    particle1_nohistory_dic[j] = particle_stokes(j+1, y0, v0,
                                            tini, vel,
                                            particle_density    = rho_p[0],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    particle2_dic[j]  = maxey_riley_imex(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p[1],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    
    particle2_nohistory_dic[j] = particle_stokes(j+1, y0, v0,
                                            tini, vel,
                                            particle_density    = rho_p[1],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    # Create particle class instances
    particle3_dic[j]  = maxey_riley_imex(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p[2],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    
    particle3_nohistory_dic[j] = particle_stokes(j+1, y0, v0,
                                            tini, vel,
                                            particle_density    = rho_p[2],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    # Create particle class instances
    particle4_dic[j]  = maxey_riley_imex(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p[3],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    
    particle4_nohistory_dic[j] = particle_stokes(j+1, y0, v0,
                                            tini, vel,
                                            particle_density    = rho_p[3],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    
    # Calculate trajectories!
    for tt in range(1, len(taxis)):
        particle1_dic[j].update()
        particle1_nohistory_dic[j].update(dt)
        
        particle2_dic[j].update()
        particle2_nohistory_dic[j].update(dt)
        
        particle3_dic[j].update()
        particle3_nohistory_dic[j].update(dt)
        
        particle4_dic[j].update()
        particle4_nohistory_dic[j].update(dt)
        
    diff1 = np.linalg.norm(particle1_dic[j].pos_vec - particle1_nohistory_dic[j].pos_vec, axis=1)
    traj1_diff = np.append(traj1_diff, max(diff1[1:] / np.linalg.norm(particle1_dic[j].pos_vec[1:] - particle1_dic[j].pos_vec[0], axis=1)))
    
    diff2 = np.linalg.norm(particle2_dic[j].pos_vec - particle2_nohistory_dic[j].pos_vec, axis=1)
    traj2_diff = np.append(traj2_diff, max(diff2[1:] / np.linalg.norm(particle2_dic[j].pos_vec[1:] - particle2_dic[j].pos_vec[0], axis=1)))
    
    diff3 = np.linalg.norm(particle3_dic[j].pos_vec - particle3_nohistory_dic[j].pos_vec, axis=1)
    traj3_diff = np.append(traj3_diff, max(diff3[1:] / np.linalg.norm(particle3_dic[j].pos_vec[1:] - particle3_dic[j].pos_vec[0], axis=1)))
    
    diff4 = np.linalg.norm(particle4_dic[j].pos_vec - particle4_nohistory_dic[j].pos_vec, axis=1)
    traj4_diff = np.append(traj4_diff, max(diff4[1:] / np.linalg.norm(particle4_dic[j].pos_vec[1:] - particle4_dic[j].pos_vec[0], axis=1)))



#
###############################################################################
############################## Plot figure plots ##############################
###############################################################################
#
fs = 6
lw = 1.2

fig = plt.figure(1, layout='tight', figsize=(2.5, 2.15))

plt.plot(S_v, traj1_diff, linewidth=lw, label='R=' + str(round(particle1_dic[0].p.R, 2)))
plt.plot(S_v, traj2_diff, linewidth=lw, label='R=' + str(round(particle2_dic[0].p.R, 2)))
plt.plot(S_v, traj3_diff, linewidth=lw, label='R=' + str(round(particle3_dic[0].p.R, 2)))
plt.plot(S_v, traj4_diff, linewidth=lw, label='R=' + str(round(particle4_dic[0].p.R, 2)))
filename = '07010605-DATA1-DIFFR.pdf'

plt.xlabel('Stokes number', fontsize=fs, labelpad=0.25)
plt.ylabel('Maximum normalised difference', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc='lower right', fontsize=fs, ncol=2)
plt.xscale('log')
plt.yscale('log')
# plt.ylim([1e-3, 1e1])
plt.grid()

plt.savefig(save_plot_to + filename,
            format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + filename, save_plot_to + filename])

plt.close()