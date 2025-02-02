#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
from subprocess import call
from tabulate import tabulate
from a03_FIELD0_DATA1 import velocity_field_Faraday1
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_DIRK4 import maxey_riley_dirk
from a09_PRTCLE_TRAPZ import maxey_riley_trapezoidal
from a09_PRTCLE_PRSTH import maxey_riley_Prasath
from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Wed Feb 07 12:49:22 2024
@author: Julio Urizarna-Carasa
"""
#
###############################################################################
############################## Import flow field ##############################
###############################################################################
#
vel          = velocity_field_Faraday1()   # Flow field



#
###############################################################################
#################### Define particle and fluid parameters #####################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
rho_p        = 2. / 3.  # - Particle's density
rho_f        = 1.       # - Fluid's density

R_v          = (1. + 2.* (rho_p/rho_f)) / 3.

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
save_plot_to = './OUTPUT/'                   # Folder where to save data
tini         = 0.   / T_scale                # Initial time
tend         = 1.   / T_scale                # Final time

y0           = np.array([0.02, 0.01]) / L_scale  # Initial position (nondimensional)
v0           = vel.get_velocity(y0[0], y0[1], tini) # Initial velocity  # Initial velocity (nondimensional)

# Define vector of time nodes (The last node is used for the calculation of the reference solution)
L_v          = np.array([ 26, 51, 101, 251, 501, 1001])



#
###############################################################################
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Control constant (Koleva 2005)
c              = 20

# Define the order for Daitche's method
order_Daitche  = 3

# Define number of time Chebyshev nodes in each time interval
nodes_dt       = 21 # Advised by Prasath et al. (2019)



#
#########################################
# Start loop over the densities vectors #
#########################################
#
# Warning
print("\nRunning this script could take up to two hours. \n")

# Create vectors where errors and calculation times will be calculated
Trap_err_v     = np.array([])
Trap_time_v    = np.array([])

Daitche_err_v  = np.array([])
Daitche_time_v = np.array([])

Prasath_err_v  = np.array([])
Prasath_time_v = np.array([])

# IMEX4 with 2nd order spatial discretization
IMEX2_err_v    = np.array([])
IMEX2_time_v   = np.array([])

# IMEX4 with 4th order spatial discretization
IMEX4_err_v    = np.array([])
IMEX4_time_v   = np.array([])

DIRK4_err_v    = np.array([])
DIRK4_time_v   = np.array([])



#
###########################################################################
######################## Calculate reference solution #####################
###########################################################################
#
print("Calculating reference solution with Prasath et al.\n")
N            = np.copy(L_v[-1])  # Number of nodes
taxis        = np.linspace(tini, tend, L_v[-1]) # Time axis
dt           = taxis[1] - taxis[0]        # time step
Reference_particle  = maxey_riley_Prasath(1, y0, v0, vel, N, tini, dt,
                                             nodes_dt,
                                             particle_density    = rho_p,
                                             fluid_density       = rho_f,
                                             particle_radius     = rad_p,
                                             kinematic_viscosity = nu_f,
                                             time_scale          = T_scale)

Reference_pos      = np.array([y0])
for tt in range(1, len(taxis)):        
    Reference_particle.update()
    Reference_pos = np.vstack((Reference_pos,
                               Reference_particle.pos_vec[tt * (nodes_dt-1)]))

# Start loop
count         = 0
for ll in range(0, len(L_v[:-1])):
    taxis     = np.linspace(tini, tend, L_v[ll]) # Time axis
    dt        = taxis[1] - taxis[0]              # time step
    count    += 1
    
    # Start registering running times
    t0_all    = time.time()
    
    
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

    Trap_particle    = maxey_riley_trapezoidal(1, y0, v0, vel,
                                               x_fd_v, c, dt, tini,
                                               particle_density    = rho_p,
                                               fluid_density       = rho_f,
                                               particle_radius     = rad_p,
                                               kinematic_viscosity = nu_f,
                                               time_scale          = T_scale)

    IMEX2_particle   = maxey_riley_imex(1, y0, v0, vel,
                                        x_fd_v, c, dt, tini,
                                        particle_density    = rho_p,
                                        fluid_density       = rho_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = T_scale,
                                        IMEXOrder           = 2,
                                        FDOrder             = 2,
                                        parallel_flag       = False)

    IMEX4_particle   = maxey_riley_imex(1, y0, v0, vel,
                                        x_fd_v, c, dt, tini,
                                        particle_density    = rho_p,
                                        fluid_density       = rho_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = T_scale,
                                        IMEXOrder           = 4,
                                        FDOrder             = 4,
                                        parallel_flag       = False)

    DIRK4_particle   = maxey_riley_dirk(1, y0, v0, vel,
                                        x_fd_v, c, dt, tini,
                                        particle_density    = rho_p,
                                        fluid_density       = rho_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = T_scale,
                                        parallel_flag       = False)

    Daitche_particle = maxey_riley_Daitche(1, y0, v0, vel, L_v[ll],
                                           order_Daitche,
                                           particle_density    = rho_p,
                                           fluid_density       = rho_f,
                                           particle_radius     = rad_p,
                                           kinematic_viscosity = nu_f,
                                           time_scale          = T_scale)

    Prasath_particle = maxey_riley_Prasath(1, y0, v0, vel, N, tini, dt,
                                           nodes_dt,
                                           particle_density    = rho_p,
                                           fluid_density       = rho_f,
                                           particle_radius     = rad_p,
                                           kinematic_viscosity = nu_f,
                                           time_scale          = T_scale)
    
    
    #
    #######################################################################
    #################### Calculate numerical solutions ####################
    #######################################################################
    #
    
    print("- Calculating Trapoidal Rule solution.")
    t0            = time.time()
    for tt in range(1, len(taxis)):
        Trap_particle.update()
    
    t_Trap        = time.time() - t0
    Trap_time_v   = np.append(Trap_time_v, t_Trap)
    
    Trap_err      = Trap_particle.pos_vec[-1] - Reference_particle.pos_vec[-1]
    Trap_err_max  = np.linalg.norm(Trap_err, ord=2, axis=0)
    Trap_err_v    = np.append(Trap_err_v, Trap_err_max)
    
    
    print("- Calculating IMEX (2nd order in space) solution.")
    t0            = time.time()
    for tt in range(1, len(taxis)):
        IMEX2_particle.update()
    
    t_IMEX2       = time.time() - t0
    IMEX2_time_v  = np.append(IMEX2_time_v, t_IMEX2)
    
    IMEX2_err     = IMEX2_particle.pos_vec[-1] - Reference_particle.pos_vec[-1]
    IMEX2_err_max = np.linalg.norm(IMEX2_err, ord=2, axis=0)
    IMEX2_err_v   = np.append(IMEX2_err_v, IMEX2_err_max)
    
    
    print("- Calculating IMEX (4th order in space) solution.")
    t0            = time.time()
    for tt in range(1, len(taxis)):
        IMEX4_particle.update()
    
    t_IMEX4       = time.time() - t0
    IMEX4_time_v  = np.append(IMEX4_time_v, t_IMEX4)
    
    IMEX4_err     = IMEX4_particle.pos_vec[-1] - Reference_particle.pos_vec[-1]
    IMEX4_err_max = np.linalg.norm(IMEX4_err, ord=2, axis=0)
    IMEX4_err_v   = np.append(IMEX4_err_v, IMEX4_err_max)
    
    
    print("- Calculating DIRK solution.")
    t0            = time.time()
    for tt in range(1, len(taxis)):
        DIRK4_particle.update()
    
    t_DIRK4       = time.time() - t0
    DIRK4_time_v  = np.append(DIRK4_time_v, t_DIRK4)
    
    DIRK4_err     = DIRK4_particle.pos_vec[-1] - Reference_particle.pos_vec[-1]
    DIRK4_err_max = np.linalg.norm(DIRK4_err, ord=2, axis=0)
    DIRK4_err_v   = np.append(DIRK4_err_v, DIRK4_err_max)
    
    
    print("- Calculating Daitche's solution.")
    if order_Daitche == 1:
        t0              = time.time()
        Daitche_particle.Euler(taxis, flag=True)  # First order method
        t_Daitche       = time.time() - t0
    elif order_Daitche == 2:
        t0              = time.time()
        Daitche_particle.AdamBashf2(taxis, flag=True) # Second order method
        t_Daitche       = time.time() - t0
    elif order_Daitche == 3:
        t0              = time.time()
        Daitche_particle.AdamBashf3(taxis, flag=True) # Third order method
        t_Daitche       = time.time() - t0
    Daitche_time_v  = np.append(Daitche_time_v, t_Daitche)
    
    Daitche_err      = Daitche_particle.pos_vec[-1] - Reference_particle.pos_vec[-1]
    Daitche_err_max  = np.linalg.norm(Daitche_err, ord=2, axis=0)
    Daitche_err_v    = np.append(Daitche_err_v, Daitche_err_max)
    
    try:
        print("- Calculating Prasath et al's solution (nodes_dt=" + str(nodes_dt) + ").")
        t0            = time.time()
        Prasath_pos   = np.array([y0])
        for tt in range(1, len(taxis)):
            Prasath_particle.update()
            Prasath_pos = np.vstack((Prasath_pos, Prasath_particle.pos_vec[tt * int(nodes_dt-1)]))
    
        t_Prasath       = time.time() - t0
        Prasath_time_v  = np.append(Prasath_time_v, t_Prasath)
    
        Prasath_err     = Prasath_pos[-1] - Reference_particle.pos_vec[-1]
        Prasath_err_max = np.linalg.norm(Prasath_err, ord=2, axis=0)
        Prasath_err_v   = np.append(Prasath_err_v, Prasath_err_max)
    
    except:
        print("- Prasath et al's (nodes_dt=" + str(nodes_dt) + ") did not Precerge.")
    
    tf_all          = time.time()
    print("\n   Round number " + str(count) + " finished in " + str(round(tf_all - t0_all,2)) + " seconds.\n")
    

PrecTrap    = str(round(np.polyfit(np.log(Trap_time_v),    np.log(Trap_err_v),   1)[0], 2))
PrecDIRK4   = str(round(np.polyfit(np.log(DIRK4_time_v),   np.log(DIRK4_err_v),  1)[0], 2))
PrecIMEX2   = str(round(np.polyfit(np.log(IMEX2_time_v),   np.log(IMEX2_err_v),  1)[0], 2))
PrecIMEX4   = str(round(np.polyfit(np.log(IMEX4_time_v),   np.log(IMEX4_err_v),  1)[0], 2))
PrecDaitche = str(round(np.polyfit(np.log(Daitche_time_v), np.log(Daitche_err_v),1)[0], 2))
PrecPrasath = str(round(np.polyfit(np.log(Prasath_time_v), np.log(Prasath_err_v),1)[0], 2))



#
###############################################################################
################################## Save data ##################################
###############################################################################
#
with open(save_plot_to + 'Data.txt', 'w') as file:
    file.write( "Parameters:\n" )
    file.write( " - t_0: " + str(tini) + "\n" )
    file.write( " - t_f: " + str(tend) + "\n" )
    file.write( " - y_0: " + str(y0) + "\n" )
    file.write( " - v_0: " + str(v0) + "\n" )
    file.write( " - R: "   + str((1.+ 2.*rho_p/rho_f) /3.) + "\n" )
    file.write( " - S: "   + str(rad_p**2./(3.*nu_f*T_scale)) + "\n\n" )
    file.write( "Nodes: "  + str(L_v) + "\n\n" )
    file.write( "Errors:\n")
    file.write( " - Prasath_err_v = "  + str(Prasath_err_v) + "\n" )
    file.write( " - Trap_err_v = "     + str(Trap_err_v) + "\n" )
    file.write( " - IMEX2_err_v = "    + str(IMEX2_err_v) + "\n" )
    file.write( " - Daitche_err_v = "  + str(Daitche_err_v) + "\n" )
    file.write( " - IMEX4_err_v = "    + str(IMEX4_err_v) + "\n" )
    file.write( " - DIRK4_err_v = "    + str(DIRK4_err_v) + "\n\n" )
    file.write( "Runtimes:\n")
    file.write( " - Prasath_time_v = " + str(Prasath_time_v) + "\n" )
    file.write( " - Trap_time_v = "    + str(Trap_time_v) + "\n" )
    file.write( " - IMEX2_time_v = "   + str(IMEX2_time_v) + "\n" )
    file.write( " - Daitche_time_v = " + str(Daitche_time_v) + "\n" )
    file.write( " - IMEX4_time_v = "   + str(IMEX4_time_v) + "\n" )
    file.write( " - DIRK4_time_v = "   + str(DIRK4_time_v) + "\n" )
  


#
###############################################################################
##################### Create Table with Precergence orders ####################
###############################################################################
#
# Create Precergence table
mydata = [["Prasath et al.:", PrecPrasath],
          ["FD2 + Trap.:",    PrecTrap],
          ["FD2 + IMEX2:",    PrecIMEX2],
          ["Datiche, " + str(order_Daitche) + " order:", PrecDaitche],
          ["FD2 + IMEX4:",    PrecIMEX4],
          ["FD2 + DIRK4:",    PrecDIRK4]]

# create header
head = ["R:", str( round(1.+2.*(rho_p/rho_f)/3., 2) )]

print("\nSlopes in the Work-precision plots.")
print("\n" + tabulate(mydata, headers=head, tablefmt="grid"))

with open(save_plot_to + 'Precision_rates.txt', 'w') as file:
    file.write("Slopes in the Work-precision plots\n")
    file.write( str(tabulate(mydata, headers=head, tablefmt="grid") ))



#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
# Plot Work-precision plot
fs   = 7
lw   = 1.4
ms   = 6

plt.figure(1, layout='tight', figsize=(3.84, 2.88))

plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

# Define vertical line characteristics
plt.vlines(x = tend * T_scale, ymin = 7e-11, ymax = 2e-2,
           colors     = 'black', linestyles = 'solid')

# Define text on plot
plt.text(4e-2, 5e-10, 'Faster than \n  Real Time', weight='bold', fontsize=8)
plt.annotate('', xy=(2e-2, 1e-10), xytext=(0.8, 1e-10), 
             arrowprops=dict(facecolor='black', shrink=0.,
                             width=2, headwidth=8, headlength=7))
plt.text(1.3, 5e-10, 'Slower than \nReal Time', weight='bold', fontsize=8)
plt.annotate('', xy=(48, 1e-10), xytext=(1.3, 1e-10), 
             arrowprops=dict(facecolor='black', shrink=0.,
                             width=2, headwidth=8, headlength=7))

# Plot data
plt.plot(Prasath_time_v, Prasath_err_v, 'P-', color="darkgoldenrod",
         label="Prasath et al. (2019)",   linewidth=lw, markersize=ms)
plt.plot(Trap_time_v,  Trap_err_v,  'v-', color="green",
         label='FD2 + Trap. Rule',        linewidth=lw, markersize=ms)
plt.plot(IMEX2_time_v,   IMEX2_err_v,   's-', color="violet",
         label = "FD2 + IMEX2",  linewidth=lw, markersize=ms)
plt.plot(Daitche_time_v, Daitche_err_v, 'd-', color="darkturquoise",
         label="Daitche, $3^{rd}$ order", linewidth=lw, markersize=ms)
plt.plot(IMEX4_time_v,   IMEX4_err_v,   'o-', color="red",
         label ="FD4 + IMEX4",  linewidth=lw, markersize=ms)
plt.plot(DIRK4_time_v,    DIRK4_err_v,    '+-', color="blue",
         label ="FD4 + DIRK4",  linewidth=lw, markersize=ms)

# Define plotting limits
plt.xlim(3e-3, 1e3)
plt.ylim(1e-12, 1e-1)

# Define plot characteristics
plt.xscale("log")
plt.yscale("log")

plt.tick_params(axis='both', labelsize=fs)
plt.ylabel('$l_2$ error at last node',  fontsize=fs, labelpad=0.25)
plt.xlabel('runtimes [sec]', fontsize=fs, labelpad=0.25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=fs-1)
plt.grid()

# Save plot
plt.savefig(save_plot_to + '06020306-DATA1-PRCSN-ZEROQ.pdf', format='pdf', dpi=500)

call(["pdfcrop", save_plot_to + '06020306-DATA1-PRCSN-ZEROQ.pdf', save_plot_to + '06020306-DATA1-PRCSN-ZEROQ.pdf'])

plt.close()

print("\007")
