#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:06:06 2024

@author: julio
"""

import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from subprocess import call

c      = 10
N      = 20
rho_p  = 2.
rho_f  = 1.
R      = (1. + 2.* (rho_p / rho_f) ) / 3.
S      = 0.1
alpha  = 1. / (R * S)
gamma  = (1. / R) * np.sqrt(3. / S)
d      = 1.0 / N

###############################################################################


###############################
# Second order implementation #
###############################

# Define psi_0
psi0   = c * np.log( (2.0 * N + 1.0) / (2.0 * N - 1.0) )
      
# Define omega_0
omega0 = c * np.log( (4.0 * N - 1.0) / (4.0 * N - 3.0) )
    
# Define entries a_11 and a_12
a11    = - ( gamma + 2.0 * alpha * omega0 ) / ( omega0 * (2.0 + gamma * psi0 ))
a12    = gamma / ( omega0 * (2.0 + gamma * psi0 ) )
    
# Create vector of the Main Diagonal
diag0         = np.zeros([1, 2 * (N - 1) + 2])[0]
diag0[0]      = a11
diag0[1]      = a11
    
# Create vector of the second upper diagonal
diag2up       = np.zeros([1, 2 * (N - 1)])[0]
diag2up[0]    = a12
diag2up[1]    = a12
    
# Create vector of the second lower diagonal
diag2dw       = np.zeros([1, 2 * (N - 1)])[0]
    
# Start filling up entries of the vectors of the diagonals.
for elem in range(1, (N - 1)):
    psii            = c * \
                      np.log((2.0 * N - 2.0 * elem + 1.0) / \
                      (2.0 * N - 2.0 * elem - 1.0))
  
    omegai          = c * \
                      np.log((4.0 * N - 4.0 * elem - 1.0) / \
                      (4.0 * N - 4.0 * elem - 3.0))
      
    mui             = c * \
                      np.log((4.0 * N - 4.0 * elem + 3.0) / \
                      (4.0 * N - 4.0 * elem + 1.0))
  
  
    diag0[2*elem]     = -(mui + omegai) / (2.0 * psii * omegai * mui)
    diag0[2*elem+1]   = -(mui + omegai) / (2.0 * psii * omegai * mui)

    diag2dw[2*elem-2] = 1.0 / (2.0 * psii * mui)
    diag2dw[2*elem-1] = 1.0 / (2.0 * psii * mui)


    if elem != (N - 1) - 1:
        diag2up[2*elem]   = 1.0 / (2.0 * psii * omegai)
        diag2up[2*elem+1] = 1.0 / (2.0 * psii * omegai)


# Build A from the diagonals
diagonals    = np.array([diag2dw, diag0, diag2up],dtype=object)
diag_pos     = np.array([-2, 0, 2])

A            = scp.sparse.diags(diagonals, diag_pos,
                    shape=(2*(N - 1), 2*(N - 1)),
                    dtype='float64')
'''
    
###############################
# Fourth order implementation #
###############################

# Calculate matrix of coefficients of the derivatives, called M:
Mdiag2dw    = np.array([c, c])
Mdiag0      = np.array([c, c,
                        4.0 * N * c / (N - 1),
                        4.0 * N * c / (N - 1)])
M1diag2up    = np.array([      N * c / (N - 1),
                              N * c / (N - 1)])

# Fill in interior matrix values by 
for elem in range(2, (N - 1)):
    iteration   = 0
    while iteration < 2:
        Mdiag2dw    = np.append(Mdiag2dw,
                            N * c / (N + 1.0 - elem))
        Mdiag0      = np.append(Mdiag0,
                            4.0 * N * c / (N - elem))
        M1diag2up   = np.append(M1diag2up,
                            N * c / (N - elem))
        iteration  += 1

M2diag2up    = np.copy(M1diag2up)
M2diag2up[0] = 3.0 * N * c / (N - 1)
M2diag2up[1] = 3.0 * N * c / (N - 1)

# Build M from the diagonals
M1diagonals  = np.array([Mdiag2dw, Mdiag0, M1diag2up], dtype=object)
M2diagonals  = np.array([Mdiag2dw, Mdiag0, M2diag2up], dtype=object)
Mdiag_pos    = np.array([-2, 0, 2])

M1           = scp.sparse.diags(M1diagonals, Mdiag_pos,
                               shape  = (2*(N - 1), 2*(N - 1)),
                               format = 'csc',
                               dtype  = 'float64')
M2           = scp.sparse.diags(M2diagonals, Mdiag_pos,
                               shape  = (2*(N - 1), 2*(N - 1)),
                               format = 'csc',
                               dtype  = 'float64')

invM1        = scp.sparse.linalg.inv(M1)
#invM1[abs(invM1) < 1e-14] = 0.0

# Calculate matrix of coefficients of q, called B:
Bdiag2dw     = np.ones(2 * (N - 1) - 2) * (-3.0 / d)

B1diag0      = np.zeros(2 * (N - 1))
B1diag0[0]   = ( 12.0 * alpha * d * c - 17.0 * gamma ) / ( 18.0 * gamma * d )
B1diag0[1]   = ( 12.0 * alpha * d * c - 17.0 * gamma ) / ( 18.0 * gamma * d )

B1diag2up    = np.ones(2 * (N - 1) - 2) * ( 3.0 / d)
B1diag2up[0] =   1.0 / ( 2.0 * d )
B1diag2up[1] =   1.0 / ( 2.0 * d )

B1diag4up    = np.zeros(2 * (N - 1) - 4)
B1diag4up[0] =   1.0 / ( 2.0 * d )
B1diag4up[1] =   1.0 / ( 2.0 * d )

B1diag6up    = np.zeros(2 * (N - 1) - 6)
B1diag6up[0] = - 1.0 / ( 18.0 * d )
B1diag6up[1] = - 1.0 / ( 18.0 * d )

B1diagonals  = np.array([Bdiag2dw, B1diag0,
                         B1diag2up, B1diag4up,
                         B1diag6up], dtype=object)
Bdiag_pos    = np.array([-2, 0, 2, 4, 6])

B1           = scp.sparse.diags(B1diagonals, Bdiag_pos,
                                shape  = (2*(N - 1), 2*(N - 1)),
                                format = 'csc',
                                dtype  = 'float64')

B2diag0      = np.zeros(2 * (N - 1))
B2diag0[0]   = -17.0 / ( 6.0 * d )
B2diag0[1]   = -17.0 / ( 6.0 * d )

B2diag2up    = np.ones(2 * (N - 1) - 2) * ( 3.0 / d)
B2diag2up[0] =   3.0 / ( 2.0 * d )
B2diag2up[1] =   3.0 / ( 2.0 * d )

B2diag4up    = np.zeros(2 * (N - 1) - 4)
B2diag4up[0] =   3.0 / ( 2.0 * d )
B2diag4up[1] =   3.0 / ( 2.0 * d )

B2diag6up    = np.zeros(2 * (N - 1) - 6)
B2diag6up[0] = - 1.0 / ( 6.0 * d )
B2diag6up[1] = - 1.0 / ( 6.0 * d )

B2diagonals  = np.array([Bdiag2dw, B2diag0,
                         B2diag2up, B2diag4up,
                         B2diag6up], dtype=object)

B2           = scp.sparse.diags(B2diagonals, Bdiag_pos,
                                shape  = (2*(N - 1), 2*(N - 1)),
                                format = 'csc',
                                dtype  = 'float64')

# Calculate P matrix
P            = np.zeros((2 * (N - 1), 2 * (N - 1)))
P[0,0]       = 1.0
P[1,1]       = 1.0

P            = scp.sparse.csc_matrix(P)

# Calculate LHS matrix, i.e. M + (10c/(3gamma)) * B2 @ Minv @ P
LHSmat       = M2 - ( 2.0 * c / (3.0 * gamma) ) * B2 @ invM1 @ P
#LHSmat[abs(LHSmat) < 1e-14] = 0.0

# Calculate RHS matrix, i.e. B2 @ Minv @ B1
RHSmat       = B2 @ invM1 @ B1
#RHSmat[abs(RHSmat) < 1e-14] = 0.0

# Calculate A matrix, i.e. LHSmat_inv @ RHS, including spatial equation
# x_t(t) = q(0,t) + u(y(t),t)
invLHSmat    = scp.sparse.linalg.inv(LHSmat)
#invLHSmat[abs(invLHSmat) < 1e-14] = 0.0
A            = invLHSmat @ RHSmat
'''

###############################################################################

plt.figure(figsize=(3.5, 3.5))
plt.imshow(abs(A.todense()), interpolation='none', cmap='binary')
plt.colorbar(shrink=0.7)

plt.savefig('05010301-SPARS-MATRX.pdf', format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", '05010301-SPARS-MATRX.pdf', '05010301-SPARS-MATRX.pdf'])


plt.show()