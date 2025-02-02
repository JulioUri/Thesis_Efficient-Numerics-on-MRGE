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
############# Define limits of the plot and import velocity field #############
###############################################################################
#
# Bounds for Convergence velocity Field
x_left  = 0.005  / L_scale
x_right = 0.03   / L_scale
y_down  =-0.002  / L_scale
y_up    = 0.025  / L_scale



#
###############################################################################
########### Define spatial grid for the plotting of the Flow Field ############
###############################################################################
#
# Define points where to show the flow field
nx = 20
ny = 21

xaxis = np.linspace(x_left, x_right, nx)
yaxis = np.linspace(y_down, y_up, ny)
X, Y  = np.meshgrid(xaxis, yaxis)


#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs   = 5
lw   = 1
ms   = 5

u, v = vel.get_velocity(X, Y, taxis[-1])

markers_on = np.arange(0, L, int((L-1)/5))


#########
# Plots #
#########

for j in range(0, len(S_v)):
    fig = plt.figure(j+1, layout='tight', figsize=(2.25, 2.))
    plt.quiver(X * L_scale, Y * L_scale,
               u * U_scale, v * U_scale)
    
    plt.plot(particle_dic[j].pos_vec[:,0] * L_scale,
             particle_dic[j].pos_vec[:,1] * L_scale,
             color='red', linewidth=lw, label="with History Term")
    
    plt.plot(particle_nohistory_dic[j].pos_vec[:,0] * L_scale,
             particle_nohistory_dic[j].pos_vec[:,1] * L_scale,
             color='blue', linewidth=lw, label="without History Term")
    
    plt.xlabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
    plt.ylabel('$y^{(2)}$', fontsize=fs, labelpad=0.25)
    plt.tick_params(axis='both', labelsize=fs)
    plt.legend(loc="lower center", fontsize=fs, prop={'size':fs})
    plt.xlim([x_left  * L_scale,
              x_right * L_scale])
    plt.ylim([y_down * L_scale,
              y_up * L_scale])
    
    if particle_dic[j].p.R < 1.:
        code = '07010601'
    elif particle_dic[j].p.R > 1.:
        code = '07010602'
        
    filename = code + '-DATA1-TRJCT-R=' + str(round(particle_dic[j].p.R, 2)) +\
               '-S=' + str(round(particle_dic[j].p.S, 2)) + '.pdf'
    
    plt.savefig(save_plot_to + filename, format='pdf', dpi=400, bbox_inches='tight')
    call(["pdfcrop", save_plot_to + filename, save_plot_to + filename])

plt.show()