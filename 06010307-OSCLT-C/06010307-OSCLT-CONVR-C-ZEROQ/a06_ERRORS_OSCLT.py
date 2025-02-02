#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
# import pandas as pd
from tabulate import tabulate
from subprocess import call
from a03_FIELD0_OSCLT import velocity_field_Oscillatory
from a09_PRTCLE_OSCLT import maxey_riley_oscillatory
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_DIRK4 import maxey_riley_dirk
from a09_PRTCLE_TRAPZ import maxey_riley_trapezoidal


"""
Created on Thu Feb 01 10:48:56 2024
@author: Julio Urizarna-Carasa
"""

#
###############################################################################
############################## Import flow field ##############################
###############################################################################
#
vel          = velocity_field_Oscillatory() # Flow field



#
###############################################################################
#################### Define particle and fluid parameters #####################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
# - Left plot
rho_p        = 2./3.   # - Particle's density
rho_f        = 1.      # - Fluid's density (water)

R = ( 1. + 2. * rho_p / rho_f ) / 3.

# Define scales of the flow field
T_scale      = np.copy(vel.T)  # Timescale of the flow field
L_scale      = np.copy(vel.L)  # Lengthscale of the flow field
U_scale      = np.copy(vel.U)  # Velocityscale of the flow field

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*T_scale)):
nu_f         = 8.917e-7 # Kinematic viscosity of water at 25ºC
rad_p        = np.sqrt(0.1) * np.sqrt(3. * nu_f * T_scale)  # Particle's radius



#
###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                    # Folder where to save data
tini         = 0.   / T_scale                 # Initial time
tend         = 1.   / T_scale                 # Final time
L            = 101                            # Time nodes
taxis        = np.linspace(tini, tend, L)     # Time axis
dt           = taxis[1] - taxis[0]            # time step

y0           = np.array([0., 0.])  / L_scale  # Initial position (nondimensional)
v0           = vel.get_velocity(y0[0], y0[1], tini)  # Initial velocity (nondimensional)

# Define vector of time and spatial nodes
N          = 101



#
###############################################################################
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Control constant (Koleva 2005)
c_v          = np.array([2., 5., 10., 15., 20., 30., 50., 100., 200.])

#
#########################################
# Start loop over the densities vectors #
#########################################
#

# initialise count variable
count         = 0

# Start loop
IMEX2_err_v   = np.array([])
IMEX4_err_v   = np.array([])
DIRK4_err_v   = np.array([])
Trap_err_v    = np.array([])
for j in range(0, len(c_v)):    
    taxis     = np.linspace(tini, tend, N) # Time axis
    dt        = taxis[1] - taxis[0]        # time step
    count    += 1
    
    t0        = time.time()
        
    #
    ################################
    # Calculate reference solution #
    ################################
    #
    Analytic_particle  = maxey_riley_oscillatory(1, y0, v0, tini, vel,
                                           particle_density    = rho_p,
                                           fluid_density       = rho_f,
                                           particle_radius     = rad_p,
                                           kinematic_viscosity = nu_f,
                                           time_scale          = T_scale)
    
    
    
    #
    #######################################################################
    ############# Define parameters for the numerical schemes #############
    #######################################################################
    #
    # Define Uniform grid [0,1]:
    xi_fd_v     = np.linspace(0., 1., N)[:-1]

    # Logarithm map to obtain QUM
    x_fd_v      = -c_v[j] * np.log(1.0 - xi_fd_v)

    Trap_particle    = maxey_riley_trapezoidal(1, y0, v0, vel,
                                              x_fd_v, c_v[j], dt, tini,
                                              particle_density    = rho_p,
                                              fluid_density       = rho_f,
                                              particle_radius     = rad_p,
                                              kinematic_viscosity = nu_f,
                                              time_scale          = T_scale)
    
    IMEX2_particle   = maxey_riley_imex(1, y0, v0, vel,
                                        x_fd_v, c_v[j], dt, tini,
                                        particle_density    = rho_p,
                                        fluid_density       = rho_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = T_scale,
                                        IMEXOrder           = 2,
                                        FDOrder             = 2,
                                        parallel_flag       = False)
    
    IMEX4_particle   = maxey_riley_imex(1, y0, v0, vel,
                                        x_fd_v, c_v[j], dt, tini,
                                        particle_density    = rho_p,
                                        fluid_density       = rho_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = T_scale,
                                        IMEXOrder           = 4,
                                        FDOrder             = 4,
                                        parallel_flag       = False)

    DIRK4_particle   = maxey_riley_dirk(1, y0, v0, vel,
                                       x_fd_v, c_v[j], dt, tini,
                                       particle_density    = rho_p,
                                       fluid_density       = rho_f,
                                       particle_radius     = rad_p,
                                       kinematic_viscosity = nu_f,
                                       time_scale          = T_scale,
                                       parallel_flag       = False)

    # Prasath_pos      = np.array([y0])
    for tt in range(1, len(taxis)):
        Analytic_particle.solve(taxis[tt])
    
        Trap_particle.update()
        IMEX2_particle.update()
        IMEX4_particle.update()
        DIRK4_particle.update()

    Trap_err      = Trap_particle.pos_vec[:,1] - Oscillatory_particle.pos_vec[:,1]
    Trap_err_max  = np.linalg.norm(Trap_err, ord=2)
    Trap_err_v    = np.append(Trap_err_v, Trap_err_max)

    IMEX2_err     = IMEX2_particle.pos_vec[:,1] - Oscillatory_particle.pos_vec[:,1]
    IMEX2_err_max = np.linalg.norm(IMEX2_err, ord=2)
    IMEX2_err_v   = np.append(IMEX2_err_v, IMEX2_err_max)

    IMEX4_err     = IMEX4_particle.pos_vec[:,1] - Oscillatory_particle.pos_vec[:,1]
    IMEX4_err_max = np.linalg.norm(IMEX4_err)
    IMEX4_err_v   = np.append(IMEX4_err_v, IMEX4_err_max)

    DIRK4_err     = DIRK4_particle.pos_vec[:,1] - Oscillatory_particle.pos_vec[:,1]
    DIRK4_err_max = np.linalg.norm(DIRK4_err, ord=2)
    DIRK4_err_v   = np.append(DIRK4_err_v, DIRK4_err_max)
    
    tf              = time.time()
    
    print("\n   Round number " + str(count) + " finished in " + str(round(tf - t0, 2)) + " seconds.\n")

#
###############################################################################
##################### Create Table with convergence orders ####################
###############################################################################
#
# # Create convergence table
mydata = [
          [c_v[0], c_v[1], c_v[2],
            c_v[3], c_v[4], c_v[5],
            c_v[6], c_v[7], c_v[8]],
          [IMEX2_err_v[0],   IMEX2_err_v[1],   IMEX2_err_v[2],
           IMEX2_err_v[3],   IMEX2_err_v[4],   IMEX2_err_v[5],
           IMEX2_err_v[6],   IMEX2_err_v[7],   IMEX2_err_v[8]],
          [Trap_err_v[0],    Trap_err_v[1],    Trap_err_v[2],
           Trap_err_v[3],    Trap_err_v[4],    Trap_err_v[5],
           Trap_err_v[6],    Trap_err_v[7],    Trap_err_v[8]],
          [IMEX4_err_v[0],   IMEX4_err_v[1],   IMEX4_err_v[2],
           IMEX4_err_v[3],   IMEX4_err_v[4],   IMEX4_err_v[5],
           IMEX4_err_v[6],   IMEX4_err_v[7],   IMEX4_err_v[8]],
          [DIRK4_err_v[0],   DIRK4_err_v[1],   DIRK4_err_v[2],
           DIRK4_err_v[3],   DIRK4_err_v[4],   DIRK4_err_v[5],
           DIRK4_err_v[6],   DIRK4_err_v[7],   DIRK4_err_v[8]]]

# create header
head = ["c", "FD2 + IMEX2:", "FD2 + Trap.:", "FD4 + IMEX4:", "FD4 + DIRK4:"]

# Transpose the data using zip
transposed_data = list(zip(*mydata))

print("\nErrors")
print("\n" + tabulate(transposed_data, headers=head, tablefmt="grid"))

with open(save_plot_to + 'Errors.txt', 'w') as file:
    file.write("Errors\n")
    file.write( str(tabulate(transposed_data, headers=head, tablefmt="grid") ))


#
###############################################################################
############################ Create Convergen plots ###########################
###############################################################################
#

fs   = 7
N_fs = 7
lw   = 1
ms   = 4

fgsz = (3.84, 2.88) # (2.5, 2.15)
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

plt.figure(1, layout='tight', figsize=fgsz)

plt.plot(c_v, IMEX2_err_v,   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(c_v, Trap_err_v,    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(c_v, IMEX4_err_v,   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(c_v, DIRK4_err_v,   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-7,1e-1)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=fs-1)
plt.xlabel('c values', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06010307-OSCLT-CNVRG-C-ZEROQ.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06010307-OSCLT-CNVRG-C-ZEROQ.pdf', save_plot_to + '06010307-OSCLT-CNVRG-C-ZEROQ.pdf'])

plt.close()

print("\007")
