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
rho_p        = np.array([0., 2./3., 4./3, 2.])  # - Particle's density
rho_f        = 1.                               # - Fluid's density (water)

# Define scales of the flow field
T_scale      = np.copy(vel.T)  # Timescale of the flow field
L_scale      = np.copy(vel.L)  # Lengthscale of the flow field
U_scale      = np.copy(vel.U)  # Velocityscale of the flow field

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*T_scale)):
nu_f         = 8.917e-7 # Kinematic viscosity of water at 25ºC
rad_p        = np.sqrt(np.geomspace(0.001, 100, 101)) * \
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
tend         = 10.  / T_scale                # Final time
L            = 101                           # Time nodes
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
traj1_diff     = np.array([])
traj2_diff     = np.array([])
particle1_dic = dict()
particle2_dic = dict()
traj3_diff     = np.array([])
traj4_diff     = np.array([])
particle3_dic = dict()
particle4_dic = dict()
for j in progressbar(range(0, len(S_v))):
    particle1_dic[j]  = maxey_riley_relaxing(j+1, y0, v0, tini,
                                            particle_density    = rho_p[0],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    particle2_dic[j]  = maxey_riley_relaxing(j+1, y0, v0, tini,
                                            particle_density    = rho_p[1],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    particle3_dic[j]  = maxey_riley_relaxing(j+1, y0, v0, tini,
                                            particle_density    = rho_p[2],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    particle4_dic[j]  = maxey_riley_relaxing(j+1, y0, v0, tini,
                                            particle_density    = rho_p[3],
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale)
    
    # Calculate trajectories!
    for tt in range(1, len(taxis)):
        particle1_dic[j].solve(taxis[tt])
        particle1_dic[j].solve_nohistory(taxis[tt])
        
        particle2_dic[j].solve(taxis[tt])
        particle2_dic[j].solve_nohistory(taxis[tt])
        
        particle3_dic[j].solve(taxis[tt])
        particle3_dic[j].solve_nohistory(taxis[tt])
        
        particle4_dic[j].solve(taxis[tt])
        particle4_dic[j].solve_nohistory(taxis[tt])
    
    diff1 = np.linalg.norm(particle1_dic[j].pos_vec - particle1_dic[j].pos_vec_nohistory, axis=1)
    traj1_diff = np.append(traj1_diff, max(diff1[1:] / np.linalg.norm(particle1_dic[j].pos_vec[1:] - particle1_dic[j].pos_vec[0], axis=1)))
    
    diff2 = np.linalg.norm(particle2_dic[j].pos_vec - particle2_dic[j].pos_vec_nohistory, axis=1)
    traj2_diff = np.append(traj2_diff, max(diff2[1:] / np.linalg.norm(particle2_dic[j].pos_vec[1:] - particle2_dic[j].pos_vec[0], axis=1)))
    
    diff3 = np.linalg.norm(particle3_dic[j].pos_vec - particle3_dic[j].pos_vec_nohistory, axis=1)
    traj3_diff = np.append(traj3_diff, max(diff3[1:] / np.linalg.norm(particle3_dic[j].pos_vec[1:] - particle3_dic[j].pos_vec[0], axis=1)))
    
    diff4 = np.linalg.norm(particle4_dic[j].pos_vec - particle4_dic[j].pos_vec_nohistory, axis=1)
    traj4_diff = np.append(traj4_diff, max(diff4[1:] / np.linalg.norm(particle4_dic[j].pos_vec[1:] - particle4_dic[j].pos_vec[0], axis=1)))
    
    

#
###############################################################################
############################## Plot figure plots ##############################
###############################################################################
#
fs = 6
lw = 1.2



#####################
# Trajectory plot 1 #
#####################

# for k in range(0, len(rho_p)):
fig = plt.figure(1, layout='tight', figsize=(2.5, 2.15))

plt.plot(S_v, traj1_diff, linewidth=lw, label="R=" + str(round(particle1_dic[0].p.R, 2)))
plt.plot(S_v, traj2_diff, linewidth=lw, label="R=" + str(round(particle2_dic[0].p.R, 2)))
plt.plot(S_v, traj3_diff, linewidth=lw, label="R=" + str(round(particle3_dic[0].p.R, 2)))
plt.plot(S_v, traj4_diff, linewidth=lw, label="R=" + str(round(particle4_dic[0].p.R, 2)))
filename = '07010102-QSCNT-DIFFR.pdf'

plt.xlabel('Stokes number', fontsize=fs, labelpad=0.25)
plt.ylabel('Maximum normalised difference', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc='lower right', fontsize=fs)
plt.xscale('log')
plt.yscale('log')
plt.ylim([3e-2, 3.])
plt.grid()

plt.savefig(save_plot_to + filename,
            format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + filename, save_plot_to + filename])

plt.show()
