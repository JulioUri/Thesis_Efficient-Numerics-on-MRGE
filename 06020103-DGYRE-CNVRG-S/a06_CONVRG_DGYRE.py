#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
from subprocess import call
from tabulate import tabulate
from a03_FIELD0_DGYRE import velocity_field_DoubleGyre
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_DIRK4 import maxey_riley_dirk
from a09_PRTCLE_TRAPZ import maxey_riley_trapezoidal
from a09_PRTCLE_PRSTH import maxey_riley_Prasath
from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Thu Feb 01 10:48:56 2024
@author: Julio Urizarna-Carasa
"""
#
###############################################################################
############################## Import flow field ##############################
###############################################################################
#
vel          = velocity_field_DoubleGyre()  # Flow field



#
###############################################################################
#################### Define particle and fluid parameters #####################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
rho_p        = 2. / 3. # - Particle's density
rho_f        = 1.   # - Fluid's density

R_v          = (1. + 2.* (rho_p/rho_f)) / 3.

# Define scales of the flow field
T_scale      = np.copy(vel.T)  # Timescale of the flow field
L_scale      = np.copy(vel.L)  # Lengthscale of the flow field
U_scale      = np.copy(vel.U)  # Velocityscale of the flow field

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*T_scale)):
nu_f         = 8.917e-7 # Kinematic viscosity of water at 25ºC
rad_p        = np.sqrt(np.array([0.01, 0.1, 0.5, 1., 2., 4., 10.])) * \
               np.sqrt(3. * nu_f * T_scale)  # Particle's radius

S_v          = rad_p**2. / (3. * nu_f * T_scale)



#
###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                   # Folder where to save data
tini         = 0.   / T_scale                # Initial time
tend         = 1.   / T_scale                # Final time

y0           = np.array([1., 0.5]) / L_scale # Initial position (nondimensional)
v0           = vel.get_velocity(y0[0], y0[1], tini) # Initial nondimensional velocity

# Define vector of time nodes (The last node is used for the calculation of the reference solution)
L_v          = np.array([ 26, 51, 101, 251, 501, 1001])



#
###############################################################################
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Control constant (Koleva 2005)
c           = 20

# Define the order for Daitche's method
order_Daitche = 3

# Define number of time Chebyshev nodes in each time interval
nodes_dt    = 21 # Advised by Prasath et al. (2019)



#
#########################################
# Start loop over the densities vectors #
#########################################
#
# Warning
print("\nRunning this script could take up to an hour. \n")

# Start loop
Trap_err_dic    = dict()
Daitche_err_dic = dict()
Prasath_err_dic = dict()
IMEX2_err_dic   = dict()
IMEX4_err_dic   = dict()
DIRK4_err_dic   = dict()
ConvTrap        = dict()
ConvDIRK4       = dict()
ConvIMEX2       = dict()
ConvIMEX4       = dict()
ConvDaitche     = dict()
ConvPrasath     = dict()
for j in range(0, len(S_v)):
    count         = 0
    
    Trap_err_v    = np.array([])
    Daitche_err_v = np.array([])
    Prasath_err_v = np.array([])
    IMEX2_err_v   = np.array([])
    IMEX4_err_v   = np.array([])
    DIRK4_err_v   = np.array([])
    
    
    #
    ###########################################################################
    ######################## Calculate reference solution #####################
    ###########################################################################
    #
    print("-> Starting calculating values of the plot Nº" + str(j+1) + "\n")
    print("    *Calculating reference solution with Prasath et al's method.")
    N            = np.copy(L_v[-1])  # Number of nodes
    taxis        = np.linspace(tini, tend, L_v[-1]) # Time axis
    dt           = taxis[1] - taxis[0]        # time step    
    Reference_particle  = maxey_riley_Prasath(0, y0, v0, vel, N, tini, dt,
                                           nodes_dt,
                                           particle_density    = rho_p,
                                           fluid_density       = rho_f,
                                           particle_radius     = rad_p[j],
                                           kinematic_viscosity = nu_f,
                                           time_scale          = T_scale)
    
    Reference_pos      = np.array([y0])
    for tt in range(1, len(taxis)):        
        Reference_particle.update()
        Reference_pos = np.vstack((Reference_pos,
                                   Reference_particle.pos_vec[tt * (nodes_dt-1)]))
            
    print("    *Starting the calculation of the numerical solutions.\n")
    for ll in range(0, len(L_v[:-1])):
        taxis     = np.linspace(tini, tend, L_v[ll]) # Time axis
        dt        = taxis[1] - taxis[0]        # time step
        count    += 1
        
        t0        = time.time()
        
        
        #
        #######################################################################
        ########### Create particle classes of the numerical schemes ##########
        #######################################################################
        #
        # Define Uniform grid [0,1):
        N           = np.copy(L_v[ll])  # Number of nodes
        xi_fd_v     = np.linspace(0., 1., int(N))[:-1]

        # Logarithm map to obtain QUM
        x_fd_v      = -c * np.log(1.0 - xi_fd_v)
    
        Trap_particle    = maxey_riley_trapezoidal(j+1, y0, v0, vel,
                                                   x_fd_v, c, dt, tini,
                                                   particle_density    = rho_p,
                                                   fluid_density       = rho_f,
                                                   particle_radius     = rad_p[j],
                                                   kinematic_viscosity = nu_f,
                                                   time_scale          = T_scale)
    
        IMEX2_particle   = maxey_riley_imex(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale,
                                            IMEXOrder           = 2,
                                            FDOrder             = 2,
                                            parallel_flag       = False)
    
        IMEX4_particle   = maxey_riley_imex(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale,
                                            IMEXOrder           = 4,
                                            FDOrder             = 4,
                                            parallel_flag       = False)

        DIRK4_particle   = maxey_riley_dirk(j+1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p,
                                            fluid_density       = rho_f,
                                            particle_radius     = rad_p[j],
                                            kinematic_viscosity = nu_f,
                                            time_scale          = T_scale,
                                            parallel_flag       = False)

        Daitche_particle = maxey_riley_Daitche(j+1, y0, v0, vel, L_v[ll],
                                               order_Daitche,
                                               particle_density    = rho_p,
                                               fluid_density       = rho_f,
                                               particle_radius     = rad_p[j],
                                               kinematic_viscosity = nu_f,
                                               time_scale          = T_scale)
    
        Prasath_particle = maxey_riley_Prasath(j+1, y0, v0, vel, N, tini, dt,
                                               nodes_dt,
                                               particle_density    = rho_p,
                                               fluid_density       = rho_f,
                                               particle_radius     = rad_p[j],
                                               kinematic_viscosity = nu_f,
                                               time_scale          = T_scale)
        
        
        #
        #######################################################################
        #################### Calculate numerical solutions ####################
        #######################################################################
        #
        
        Prasath_pos      = np.array([y0])
        for tt in range(1, len(taxis)):        
            try:
                Prasath_particle.update()
                Prasath_pos = np.vstack((Prasath_pos,
                                         Prasath_particle.pos_vec[tt * (nodes_dt-1)]))
            except:
                None
        
            Trap_particle.update()
            IMEX2_particle.update()
            IMEX4_particle.update()
            DIRK4_particle.update()
        
        try:
            Prasath_err     = Prasath_pos[-1] - Reference_pos[-1]
            Prasath_err_max = np.linalg.norm(Prasath_err, ord=2, axis=0)
            Prasath_err_v   = np.append(Prasath_err_v, Prasath_err_max)
        except:
            None

        Trap_err      = Trap_particle.pos_vec[-1] - Reference_pos[-1]
        Trap_err_max  = np.linalg.norm(Trap_err, ord=2, axis=0)
        Trap_err_v    = np.append(Trap_err_v, Trap_err_max)

        IMEX2_err     = IMEX2_particle.pos_vec[-1] - Reference_pos[-1]
        IMEX2_err_max = np.linalg.norm(IMEX2_err, ord=2, axis=0)
        IMEX2_err_v   = np.append(IMEX2_err_v, IMEX2_err_max)

        IMEX4_err     = IMEX4_particle.pos_vec[-1] - Reference_pos[-1]
        IMEX4_err_max = np.linalg.norm(IMEX4_err, ord=2, axis=0)
        IMEX4_err_v   = np.append(IMEX4_err_v, IMEX4_err_max)

        DIRK4_err     = DIRK4_particle.pos_vec[-1] - Reference_pos[-1]
        DIRK4_err_max = np.linalg.norm(DIRK4_err, ord=2, axis=0)
        DIRK4_err_v   = np.append(DIRK4_err_v, DIRK4_err_max)

        if order_Daitche == 1:
            Daitche_particle.Euler(taxis, flag=True)  # First order method
        elif order_Daitche == 2:
            Daitche_particle.AdamBashf2(taxis, flag=True) # Second order method
        elif order_Daitche == 3:
            Daitche_particle.AdamBashf3(taxis, flag=True) # Third order method

        Daitche_err     = Daitche_particle.pos_vec[-1] - Reference_pos[-1]
        Daitche_err_max = np.linalg.norm(Daitche_err, ord=2, axis=0)
        Daitche_err_v   = np.append(Daitche_err_v, Daitche_err_max)
        
        tf              = time.time()
        
        print("   Round number " + str(count) + " finished in " + str(round(tf - t0, 2)) + " seconds.\n")
        
    Trap_err_dic[j]    = Trap_err_v
    Daitche_err_dic[j] = Daitche_err_v
    Prasath_err_dic[j] = Prasath_err_v
    IMEX2_err_dic[j]   = IMEX2_err_v
    IMEX4_err_dic[j]   = IMEX4_err_v
    DIRK4_err_dic[j]   = DIRK4_err_v

    ConvTrap[j]    = str(round(np.polyfit(np.log(L_v[:-1]), np.log(Trap_err_v),   1)[0], 2))
    ConvDIRK4[j]   = str(round(np.polyfit(np.log(L_v[:-1]), np.log(DIRK4_err_v),  1)[0], 2))
    ConvIMEX2[j]   = str(round(np.polyfit(np.log(L_v[:-1]), np.log(IMEX2_err_v),  1)[0], 2))
    ConvIMEX4[j]   = str(round(np.polyfit(np.log(L_v[:-1]), np.log(IMEX4_err_v),  1)[0], 2))
    ConvDaitche[j] = str(round(np.polyfit(np.log(L_v[:-1]), np.log(Daitche_err_v),1)[0], 2))
    ConvPrasath[j] = str(round(np.polyfit(np.log(L_v[:-1]), np.log(Prasath_err_v),1)[0], 2))



#
###############################################################################
################################## Save data ##################################
###############################################################################
#
for j in range(0, len(S_v)):
    with open(save_plot_to + 'Data_0'  + str(j+1) +'.txt', 'w') as file:
        file.write( "Parameters:\n" )
        file.write( " - t_0: " + str(tini) + "\n" )
        file.write( " - t_f: " + str(tend) + "\n" )
        file.write( " - y_0: " + str(y0) + "\n" )
        file.write( " - v_0: " + str(v0) + "\n" )
        file.write( " - R: " + str((1.+ 2.*rho_p/rho_f) /3.) + "\n" )
        file.write( " - S: " + str(S_v[j]) + "\n\n" )
        file.write( "Nodes: " + str(L_v) + "\n\n" )
        file.write( "Errors:\n")
        file.write( " - Prasath_err_v = " + str(Prasath_err_dic[j]) + "\n" )
        file.write( " - Trap_err_v = " + str(Trap_err_dic[j]) + "\n" )
        file.write( " - IMEX2_err_v = " + str(IMEX2_err_dic[j]) + "\n" )
        file.write( " - Daitche_err_v = " + str(Daitche_err_dic[j]) + "\n" )
        file.write( " - IMEX4_err_v = " + str(IMEX4_err_dic[j]) + "\n" )
        file.write( " - DIRK4_err_v = " + str(DIRK4_err_dic[j]) + "\n" )



#
###############################################################################
##################### Create Table with convergence orders ####################
###############################################################################
#
# Create convergence table
mydata = [
         ["Prasath et al.:",
          ConvPrasath[0], ConvPrasath[1], ConvPrasath[2],
          ConvPrasath[3], ConvPrasath[4], ConvPrasath[5],
          ConvPrasath[6]],
         ["FD2 + Trap.:",
          ConvTrap[0],    ConvTrap[1],    ConvTrap[2],
          ConvTrap[3],    ConvTrap[4],    ConvTrap[5],
          ConvTrap[6]],
         ["FD2 + IMEX2:",
          ConvIMEX2[0],   ConvIMEX2[1],   ConvIMEX2[2],
          ConvIMEX2[3],   ConvIMEX2[4],   ConvIMEX2[5],
          ConvIMEX2[6]],
         ["Datiche, " + str(order_Daitche) + " order:",
          ConvDaitche[0], ConvDaitche[1], ConvDaitche[2],
          ConvDaitche[3], ConvDaitche[4], ConvDaitche[5],
          ConvDaitche[6]],
         ["FD2 + IMEX4:",
          ConvIMEX4[0],   ConvIMEX4[1],   ConvIMEX4[2],
          ConvIMEX4[3],   ConvIMEX4[4],   ConvIMEX4[5],
          ConvIMEX4[6]],
         ["FD2 + DIRK4:",
          ConvDIRK4[0],   ConvDIRK4[1],   ConvDIRK4[2],
          ConvDIRK4[3],   ConvDIRK4[4],   ConvDIRK4[5],
          ConvDIRK4[6]]
         ]

# create header
head = ["S:", str(round(S_v[0], 2)), str(round(S_v[1], 2)),
              str(round(S_v[2], 2)), str(round(S_v[3], 2)),
              str(round(S_v[4], 2)), str(round(S_v[5], 2)),
              str(round(S_v[6], 2))]

print("\nConvergence rates")
print("\n" + tabulate(mydata, headers=head, tablefmt="grid"))

with open(save_plot_to + 'Convergence_rates.txt', 'w') as file:
    file.write("Convergence rates\n")
    file.write( str(tabulate(mydata, headers=head, tablefmt="grid") ))



#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs   = 5
N_fs = 7
lw   = 1
ms   = 5

fgsz = (1.8, 1.62) # (2.5, 2.15)



#
############
# Plot nº1 #
############
#

plt.figure(1, layout='tight', figsize=fgsz)

# plt.plot(L_v[2:4], 20.*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 2e-3, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5e-1*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-9, "$N^{-3}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[0], 'P-', color='darkgoldenrod', label="Prasath et al. (2019)", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Trap_err_dic[0],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX2_err_dic[0],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Daitche_err_dic[0], 'd-', color='darkturquoise',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX4_err_dic[0],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], DIRK4_err_dic[0],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-14,1e-1)
plt.xlabel('Nodes N', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[0], 2))  + '.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[0], 2))  + '.pdf', save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[0], 2))  + '.pdf'])

plt.close()



#
############
# Plot nº2 #
############
#
plt.figure(2, layout='tight', figsize=fgsz)

# plt.plot(L_v[2:4], 6.*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-8, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 1.*L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 4e-9, "$N^{-4}$", color='grey')


plt.plot(L_v[:-1], Prasath_err_dic[1], 'P-', color='darkgoldenrod', label="Prasath et al. (2019)", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Trap_err_dic[1],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX2_err_dic[1],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Daitche_err_dic[1], 'd-', color='darkturquoise',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX4_err_dic[1],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], DIRK4_err_dic[1],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-14,1e-1)
plt.xlabel('Nodes N', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[1], 2))  + '.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[1], 2))  + '.pdf', save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[1], 2))  + '.pdf'])

plt.close()



#
############
# Plot nº3 #
############
#
plt.figure(3, layout='tight', figsize=fgsz)

# plt.plot(L_v[2:4], 20.*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 2e-3, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5e-1*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-9, "$N^{-3}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[2], 'P-', color='darkgoldenrod', label="Prasath et al. (2019)", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Trap_err_dic[2],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX2_err_dic[2],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Daitche_err_dic[2], 'd-', color='darkturquoise',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX4_err_dic[2],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], DIRK4_err_dic[2],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-14,1e-1)
plt.xlabel('Nodes N', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[2], 2))  + '.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[2], 2))  + '.pdf', save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[2], 2))  + '.pdf'])

plt.close()



#
############
# Plot nº4 #
############
#
plt.figure(4, layout='tight', figsize=fgsz)

# plt.plot(L_v[2:4], 20.*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 2e-3, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5e-1*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-9, "$N^{-3}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[3], 'P-', color='darkgoldenrod', label="Prasath et al. (2019)", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Trap_err_dic[3],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX2_err_dic[3],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Daitche_err_dic[3], 'd-', color='darkturquoise',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX4_err_dic[3],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], DIRK4_err_dic[3],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-14,1e-1)
plt.xlabel('Nodes N', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[3], 2))  + '.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[3], 2))  + '.pdf', save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[3], 2))  + '.pdf'])

plt.close()



#
############
# Plot nº5 #
############
#
plt.figure(5, layout='tight', figsize=fgsz)

# plt.plot(L_v[2:4], 20.*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 2e-3, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5e-1*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-9, "$N^{-3}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[4], 'P-', color='darkgoldenrod', label="Prasath et al. (2019)", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Trap_err_dic[4],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX2_err_dic[4],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Daitche_err_dic[4], 'd-', color='darkturquoise',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX4_err_dic[4],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], DIRK4_err_dic[4],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-14,1e-1)
plt.xlabel('Nodes N', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[4], 2))  + '.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[4], 2))  + '.pdf', save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[4], 2))  + '.pdf'])

plt.close()



#
############
# Plot nº6 #
############
#
plt.figure(6, layout='tight', figsize=fgsz)

# plt.plot(L_v[2:4], 20.*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 2e-3, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5e-1*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-9, "$N^{-3}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[5], 'P-', color='darkgoldenrod', label="Prasath et al. (2019)", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Trap_err_dic[5],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX2_err_dic[5],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Daitche_err_dic[5], 'd-', color='darkturquoise',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX4_err_dic[5],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], DIRK4_err_dic[5],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-14,1e-1)
plt.xlabel('Nodes N', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[5], 2))  + '.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[5], 2))  + '.pdf', save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[5], 2))  + '.pdf'])

plt.close()



#
############
# Plot nº7 #
############
#
plt.figure(7, layout='tight', figsize=fgsz)

# plt.plot(L_v[2:4], 20.*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 2e-3, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5e-1*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 7e-9, "$N^{-3}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[6], 'P-', color='darkgoldenrod', label="Prasath et al. (2019)", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Trap_err_dic[6],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX2_err_dic[6],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], Daitche_err_dic[6], 'd-', color='darkturquoise',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], IMEX4_err_dic[6],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw, markersize=ms)
plt.plot(L_v[:-1], DIRK4_err_dic[6],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw, markersize=ms)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-14,1e-1)
plt.xlabel('Nodes N', fontsize=fs, labelpad=0.25)
plt.ylabel('$l_2$ error at last node', fontsize=fs, labelpad=0.25)
plt.grid()

plt.savefig(save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[6], 2))  + '.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[6], 2))  + '.pdf', save_plot_to + '06020103-DGYRE-CNVRG-ZEROQ-S=' + str(round(S_v[6], 2))  + '.pdf'])

plt.close()



#
##########
# LEGEND #
##########
#
plt.figure(4, layout='tight', figsize=fgsz)
f       = lambda m, c: plt.plot([], [], marker=m, color=c, linewidth=lw, markersize=ms)[0]
colors  = ['darkgoldenrod', 'green', 'violet', 'darkturquoise', 'red', 'blue']
markers = ['P', 'v', 's', 'd', 'o', '+']
handles = [f(markers[i], colors[i]) for i in range(0, len(colors))]

labels  = ['Prasath et al. (2019)',
           'FD2 + Trap. Rule',
           'FD2 + IMEX2',
           'Daitche, 3rd order',
           'FD4 + IMEX4',
           'FD4 + DIRK4']
legend  = plt.legend(handles, labels, loc='center', fontsize=fs+4, framealpha=1, frameon=True)

def export_legend(legend, filename="LEGEND.pdf", expand=[-65,-55,-15,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    legend.axes.axis("off")
    fig.savefig(save_plot_to + filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)

plt.close()

print("\007")
