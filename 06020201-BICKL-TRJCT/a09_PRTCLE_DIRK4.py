import numpy as np
import scipy as scp

from progressbar import progressbar
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.optimize import newton, root, newton_krylov, broyden1, fsolve

from a00_PMTERS_CONST import mr_parameter

'''
The class "maxey-riley_dirk" given below calculates the trajectory and
velocity of particles whose dynamics is governed by the FULL MAXEY-RILEY
by using an approach by a 2nd order FD + 4th order DIRK methods.

The class "update_particle" is provided for the parallelization of the
calculations (since the ammount of particles to be calculated could
be huge for some cases ~10^4.
'''

'''
###############################################################################
##########################FULL MAXEY RILEY (DIRK4)#############################
###############################################################################
'''

def update_particle(particle):
  results    = particle.update()
  return results



class maxey_riley_dirk(object):

  def __init__(self, tag, x, v, velocity_field, z_v, control, dt, tini,
               particle_density=1, fluid_density=1, particle_radius=1,
               kinematic_viscosity=1, time_scale=1, FDOrder = 4,
               parallel_flag = False):
      '''
      Calculates particles trajectory and relative velocity by using a 4th
      order DIRK method (*SPECIFY METHOD*) for the time integration and
      a 2nd order FD scheme (centered differences) for the spatial
      discretization.
      
      The problem solved is as follows:
          
          q_zz (z,t) = q_t(z,t)
          q(z,t_0)   = [0, 0]^T
          q_t(0,t) + alpha * q(0,t) + gamma * q_z(0,t) = f(q(0,t), X(t), t)
          
          for z, t, alpha, gamma > 0 and q(0,t) = [q(0,t), p(0,t)]^T is
          the relative velocity of the particle and z represents a 1D
          pseudospace without any phisical meaning.
          
          For the calculation of q(0,t), we also need to know the position
          of the particle X(t) = [x(t), y(t)], which is calculated by
          using the following ODE:
          
          X_t(t)     = q(0,t) + U(X(t))
          X(t_0)     = [x_0, y_0]
          q(0,t_0)   = V_0
          
          More information could be found in the notes attached to this
          TOOLBOX.
          
      Parameters
      -------
      tag: int
           Natural number used for the identification of the particle.
                  
      x: float, sequence, or ndarray
           Initial position of the particle
                  
      v: float, sequence, or ndarray
           Initial absolute velocity of the particle
                
      velocity_field: class
           Class which includes methods that define the velocity field,
           i.e. get_velocity(), get_gradient(), get_du_dt
                      
           This is used for the definition of the vector U(X(t)) and the
           flow function f(q(0,t), X(t), t).
                      
           Check notes for further information.
                
      z_v: sequence, or ndarray
           Discretized pseudospace.
                
      control: float, int
           Parameter "c" in the logarithmic map that produces the
           Quasi-Uniform Grid from the Uniform Grid.
                      
           More information in Notes or in Koleva (2005)'s paper.
                  
      dt: float, int
           Time step in the time integration.
                     
      tini: float, int
           Initial time.
                      
      particle_density: float, int, optional
           Particles' densities.
                      
           Used for the calculation of the constant "R".
                      
           IMPORTANT: All particles have the same density.
                  
      fluid_density: float, int, optional
           Fluid's density.
                      
           Used for the calculation of the constant "R".
                  
      particle_radius: float, int, optional
           Particles' radius.
                      
           Used for the calculation of the Stokes's number "S".
                      
           IMPORTANT: All particles have the same density.
                      
      kinematic_viscosity: float, int, optional
           Fluid's kinematic viscosity.
                      
           Used for the calculation of the Stokes's number "S".
                  
      time_scale: float, int, optional
           Velocity field's time scale.
                      
           Used for the calculation of the Stokes's number "S".
      '''
      
      ## Initialize of variables that will be used in the code. ##
      
      self.tag     = tag
      self.N       = len(z_v)        # Nº of nodes in space discret.
      self.N_inf   = len(z_v) + 1    # Nº of nodes in space discret. + inf.
      self.time    = tini
      self.c       = control
      self.d       = 1.0 / self.N_inf
      
      self.parallel_flag = parallel_flag
      
      self.vel     = velocity_field
      u0, v0       = velocity_field.get_velocity(x[0], x[1], tini)
                                     # flow's velocity at initial position.
      self.q0      = np.array([v[0] - u0, v[1] - v0])
        
      # q_n = [q(x_i, tini), p(x_i, tini)] vector
      # (semidiscretized vector or rel. velocities on space grid), where
      # q(x_i, tini) is the horizontal component and p(x_i, tini) the
      # vertical one
      
      q_n          = np.zeros((1, 2*self.N+2))[0]
      q_n[0]       = v[0] - u0
      q_n[1]       = v[1] - v0
      q_n[-2]      = x[0]
      q_n[-1]      = x[1]
      self.q_n     = q_n.copy()
      
      # "mr_parameter" is a class that has all the methods that calculate
      # all the parameters needed for the MRE, s.t. alpha, gamma, R, S...
      # See Prasath et al (2019) paper for more information.
      # For example: alpha will be called as self.p.alpha
      self.p       = mr_parameter(particle_density, fluid_density,
                                  particle_radius, kinematic_viscosity,
                                  time_scale)
      
      # Create position vector
      self.pos_vec = np.copy(x)
      
      # Relative velocity vectors for first and second pseudospatial nodes.
      # This is needed for the calculation of the Forces acting on the particle.
      self.q0_vec  = np.array([q_n[0], q_n[1]])
      self.q1_vec  = np.array([q_n[2], q_n[3]])
      
      
      # The following is used in case the velocity field data is restricted
      # in space.
      # Analytical velocity fields do not usually have that problem since
      # they are usually defined for the whole R^2 or R^3 but
      # that is not the case for the Experimental velocity fields, for
      # which the velocity of the field is only available for a reduced
      # domain.
      # The Exception is then raised if the particle's initial position is
      # outside the domain.
      if self.vel.limits == True:
          if (x[0] > self.vel.x_right or x[0] < self.vel.x_left or x[1] > self.vel.y_up or x[1] < self.vel.y_down):
              raise Exception("Particle's initial position is outside the spatial domain")
      
      # Save dt as a global variable in the class.
      self.dt       = dt
      
      # Decide FD convergence order, either 2 (Koleva's) or 4 (Julio's CFD).
      self.FD_Tag   = FDOrder
      
      sqrt2 = np.sqrt(2.)
      # create matrix of coeff for 4th order method  
      self.A_mat   = np.array([[0., 0., 0., 0., 0., 0.],
                               [1./4., 1./4., 0., 0., 0., 0.],
                               [(1.-sqrt2)/8., (1.-sqrt2)/8., 1./4., 0., 0., 0.],
                               [(5.-7.*sqrt2)/64., (5.-7.*sqrt2)/64., 7.*(1.+sqrt2)/32., 1./4., 0., 0.],
                               [(-13796.-54539.*sqrt2)/125000., (-13796.-54539.*sqrt2)/125000., (506605.+132109.*sqrt2)/437500., 166.*(-97.+376.*sqrt2)/109375., 1./4., 0.],
                               [(1181.-987.*sqrt2)/13782., (1181.-987.*sqrt2)/13782., 47.*(-267.+1783.*sqrt2)/273343., -16.*(-22922.+3525.*sqrt2)/571953., -15625.*(97.+376.*sqrt2)/90749876., 1./4.]])
      
      self.b_vec   = np.array([(1181.-987.*sqrt2)/13782., (1181.-987.*sqrt2)/13782., 47.*(-267.+1783.*sqrt2)/273343., -16.*(-22922.+3525.*sqrt2)/571953., -15625.*(97.+376.*sqrt2)/90749876., 1./4.])
      self.c_vec   = np.array([0., 1./2., (2.-sqrt2)/4., 5./8., 26./25., 1.])
      
      # Calculate matrix A of the Semidiscrete system
      self.calculate_A()
      self.gammax = (3+2*np.sqrt(3)*np.cos(np.pi/18))/6 #constant for the Butcher-tableau
      
      # splu function works well in serial but not in parallel, since it is
      # written in C and cannot be pickled. Therefore, a different approach
      # has to be undertaken.
      if parallel_flag == False:
         self.LU_psi = sparse.linalg.splu(self.Psi) # LU factorisation of the linear system 

  
  def calculate_A(self):
      '''
      This method obtains the matrix A that is obtained from the
      semidiscretized system with the Quasi-Unifrom Grid from M. Koleva (2006).
      
      All entries are defined in the notes.
      
      The matrix has entries in certain diagonals (mainly main diagonal (+0) and
      the second diagonals above (+2) and below it (-2)), therefore the matrix
      A is built as a sparse matrix from the vector of its diagonals, i.e.
      we will use "scp.sparse.diags" to build A.
      
      Returns
      -------
      None, all data saved in global variables
      '''
      
      if self.FD_Tag == 2:
          # Second order implementation
          # See notes for reference
      
          # Define psi_0
          psi0          = self.c * \
              np.log( (2.0 * self.N_inf + 1.0) / (2.0 * self.N_inf - 1.0) )
        
          self.psi0     = psi0
        
          # Define omega_0
          omega0        = self.c * \
              np.log( (4.0 * self.N_inf - 1.0) / (4.0 * self.N_inf - 3.0) )
        
          self.omega0   = omega0
      
          # Define entries a_11 and a_12
          a11           = - (self.p.gamma + 2.0 * self.p.alpha * omega0) / \
                        ( omega0 * (2.0 + self.p.gamma * psi0 ))
                        
          a12           = self.p.gamma / ( omega0 * (2.0 + self.p.gamma * psi0 ) )
      
          # Create vector of the Main Diagonal
          diag0         = np.zeros([1, 2 * self.N + 2])[0]
          diag0[0]      = a11
          diag0[1]      = a11
      
          # Create vector of the second upper diagonal
          diag2up       = np.zeros([1, 2 * self.N])[0]
          diag2up[0]    = a12
          diag2up[1]    = a12
      
          # Create vector of the second lower diagonal
          diag2dw       = np.zeros([1, 2 * self.N])[0]
      
          # Start filling up entries of the vectors of the diagonals.
          for elem in range(1, self.N):
              psii            = self.c * \
                                np.log((2.0 * self.N_inf - 2.0 * elem + 1.0) / \
                                (2.0 * self.N_inf - 2.0 * elem - 1.0))
            
              omegai          = self.c * \
                                np.log((4.0 * self.N_inf - 4.0 * elem - 1.0) / \
                                (4.0 * self.N_inf - 4.0 * elem - 3.0))
                
              mui             = self.c * \
                                np.log((4.0 * self.N_inf - 4.0 * elem + 3.0) / \
                                (4.0 * self.N_inf - 4.0 * elem + 1.0))
        
        
              diag0[2*elem]     = -(mui + omegai) / (2.0 * psii * omegai * mui)
              diag0[2*elem+1]   = -(mui + omegai) / (2.0 * psii * omegai * mui)
          
              diag2dw[2*elem-2] = 1.0 / (2.0 * psii * mui)
              diag2dw[2*elem-1] = 1.0 / (2.0 * psii * mui)
          
          
              if elem != self.N - 1:
                  diag2up[2*elem]   = 1.0 / (2.0 * psii * omegai)
                  diag2up[2*elem+1] = 1.0 / (2.0 * psii * omegai)
      
          # Create diagonal that includes the space equations
          diag_ones    = np.array([1.0, 1.0])
      
          # Build A from the diagonals
          diagonals    = np.array([diag_ones, diag2dw, diag0, diag2up],dtype=object)
          diag_pos     = np.array([-2*self.N, -2, 0, 2])
      
          self.A       = scp.sparse.diags(diagonals, diag_pos,
                              shape=(2*self.N + 2, 2*self.N + 2),
                              dtype='float64')
      
      elif self.FD_Tag == 4:
          
          # Fourth order implementation
      
          # Calculate matrix of coefficients of the derivatives, called M:
          Mdiag2dw    = np.array([self.c, self.c])
          Mdiag0      = np.array([self.c, self.c,
                                  4.0 * self.N_inf * self.c / self.N,
                                  4.0 * self.N_inf * self.c / self.N])
          M1diag2up   = np.array([      self.N_inf * self.c / self.N,
                                        self.N_inf * self.c / self.N])
      
          # Fill in interior matrix values by 
          for elem in range(2, self.N):
              iteration   = 0
              while iteration < 2:
                  Mdiag2dw    = np.append(Mdiag2dw,
                                      self.N_inf * self.c / (self.N_inf + 1.0 - elem))
                  Mdiag0      = np.append(Mdiag0,
                                      4.0 * self.N_inf * self.c / (self.N_inf - elem))
                  M1diag2up   = np.append(M1diag2up,
                                      self.N_inf * self.c / (self.N_inf - elem))
                  iteration  += 1
          
          M2diag2up    = np.copy(M1diag2up)
          M2diag2up[0] = 3.0 * self.N_inf * self.c / self.N
          M2diag2up[1] = 3.0 * self.N_inf * self.c / self.N
          
          # Build M from the diagonals
          M1diagonals  = np.array([Mdiag2dw, Mdiag0, M1diag2up], dtype=object)
          M2diagonals  = np.array([Mdiag2dw, Mdiag0, M2diag2up], dtype=object)
          Mdiag_pos    = np.array([-2, 0, 2])
      
          M1           = sparse.diags(M1diagonals, Mdiag_pos,
                                         shape  = (2*self.N, 2*self.N),
                                         format = 'csc',
                                         dtype  = 'float64')
          M2           = sparse.diags(M2diagonals, Mdiag_pos,
                                         shape  = (2*self.N, 2*self.N),
                                         format = 'csc',
                                         dtype  = 'float64')
          
          invM1        = sparse.linalg.inv(M1)
          
          # Calculate matrix of coefficients of q, called B:
          Bdiag2dw     = np.ones(2 * self.N - 2) * (-3.0 / self.d)
          
          B1diag0      = np.zeros(2 * self.N)
          B1diag0[0]   = ( 12.0 * self.p.alpha * self.d * self.c - 17.0 * self.p.gamma ) / ( 18.0 * self.p.gamma * self.d )
          B1diag0[1]   = ( 12.0 * self.p.alpha * self.d * self.c - 17.0 * self.p.gamma ) / ( 18.0 * self.p.gamma * self.d )
      
          B1diag2up    = np.ones(2 * self.N - 2) * ( 3.0 / self.d)
          B1diag2up[0] =   1.0 / ( 2.0 * self.d )
          B1diag2up[1] =   1.0 / ( 2.0 * self.d )
      
          B1diag4up    = np.zeros(2 * self.N - 4)
          B1diag4up[0] =   1.0 / ( 2.0 * self.d )
          B1diag4up[1] =   1.0 / ( 2.0 * self.d )
      
          B1diag6up    = np.zeros(2 * self.N - 6)
          B1diag6up[0] = - 1.0 / ( 18.0 * self.d )
          B1diag6up[1] = - 1.0 / ( 18.0 * self.d )
      
          B1diagonals  = np.array([Bdiag2dw, B1diag0,
                                   B1diag2up, B1diag4up,
                                   B1diag6up], dtype=object)
          Bdiag_pos    = np.array([-2, 0, 2, 4, 6])
      
          B1           = sparse.diags(B1diagonals, Bdiag_pos,
                                          shape  = (2*self.N, 2*self.N),
                                          format = 'csc',
                                          dtype  = 'float64')
          
          B2diag0      = np.zeros(2 * self.N)
          B2diag0[0]   = -17.0 / ( 6.0 * self.d )
          B2diag0[1]   = -17.0 / ( 6.0 * self.d )
      
          B2diag2up    = np.ones(2 * self.N - 2) * ( 3.0 / self.d)
          B2diag2up[0] =   3.0 / ( 2.0 * self.d )
          B2diag2up[1] =   3.0 / ( 2.0 * self.d )
      
          B2diag4up    = np.zeros(2 * self.N - 4)
          B2diag4up[0] =   3.0 / ( 2.0 * self.d )
          B2diag4up[1] =   3.0 / ( 2.0 * self.d )
      
          B2diag6up    = np.zeros(2 * self.N - 6)
          B2diag6up[0] = - 1.0 / ( 6.0 * self.d )
          B2diag6up[1] = - 1.0 / ( 6.0 * self.d )
      
          B2diagonals  = np.array([Bdiag2dw, B2diag0,
                                   B2diag2up, B2diag4up,
                                   B2diag6up], dtype=object)
      
          B2           = sparse.diags(B2diagonals, Bdiag_pos,
                                          shape  = (2*self.N, 2*self.N),
                                          format = 'csc',
                                          dtype  = 'float64')
          
          # Calculate P matrix
          P            = np.zeros((2 * self.N, 2 * self.N))
          P[0,0]       = 1.0
          P[1,1]       = 1.0
      
          P            = sparse.coo_matrix(P)
          
          ############
          # OLD CODE #
          ############
          '''
          # Calculate LHS matrix, i.e. M + (10c/(3gamma)) * B2 @ Minv @ P
          LHSmat       = M2 - ( 2.0 * self.c / (3.0 * self.p.gamma) ) * B2 @ invM1 @ P
          
          # Calculate RHS matrix, i.e. B2 @ Minv @ B1
          RHSmat       = B2 @ invM1 @ B1
          
          # Calculate A matrix, i.e. LHSmat_inv @ RHS, including spatial equation
          # x_t(t) = q(0,t) + u(y(t),t)
          invLHSmat    = sparse.linalg.inv(LHSmat)
          A            = invLHSmat @ RHSmat
          A.data[abs(A.data) < 1e-25] = 0.0
          
          rightsubA    = np.zeros((2 * self.N, 2))
          lowsubA      = np.zeros((2, 2 * self.N_inf))
          lowsubA[0,0] = 1.0
          lowsubA[1,1] = 1.0
          
          A            = sparse.hstack((A, rightsubA))
          A            = sparse.vstack((A, lowsubA))
          self.A       = sparse.coo_matrix(A)
          
          aux_mat      = invLHSmat @ B2 @ invM1
          aux_mat.data[abs(aux_mat.data)<1e-25] = 0.0
          aux_mat.eliminate_zeros()
          
          # Calculate matrix of coefficients of f
          f_coeff  = (-2.0 * self.c / (3.0 * self.p.gamma)) * aux_mat
          f_coeff.data[abs(f_coeff.data) < 1e-25] = 0.0
          f_coeff.eliminate_zeros()
          self.f_coeff  = f_coeff[:,0].toarray().T[0]
          self.g_coeff  = f_coeff[:,1].toarray().T[0]
          '''
          
          Psi          = M2 - ( 2.0 * self.c / (3.0 * self.p.gamma) ) * B2 @ invM1 @ P
          self.Psi     = sparse.block_diag((Psi, np.eye(2))).tocsr()
          
          # Calculate RHS matrix, i.e. B2 @ Minv @ B1
          A            = B2 @ invM1 @ B1
          A.data[abs(A.data) < 1e-25] = 0.0
          A.eliminate_zeros()
          
          rightsubA    = np.zeros((2 * self.N, 2))
          lowsubA      = np.zeros((2, 2 * self.N_inf))
          lowsubA[0,0] = 1.0
          lowsubA[1,1] = 1.0
          
          A            = A.toarray()
          A            = np.hstack((A, rightsubA))
          A            = np.vstack((A, lowsubA))
          self.A       = sparse.coo_matrix(A)
          
          aux_mat      = B2 @ invM1
          aux_mat.data[abs(aux_mat.data)<1e-25] = 0.0
          aux_mat.eliminate_zeros()
          
          # Calculate matrix of coefficients of f
          f_coeff  = (-2.0 * self.c / (3.0 * self.p.gamma)) * aux_mat
          f_coeff.data[abs(f_coeff.data) < 1e-25] = 0.0
          f_coeff.eliminate_zeros()
          self.f_coeff  = f_coeff[:,0].toarray().T[0]
          self.g_coeff  = f_coeff[:,1].toarray().T[0]
          

  def calculate_f(self, x, t, qv):
      '''
      Calculate flow function f(q(0,t),X(t),t) in the boundary condition
      given the velocity field as a class.
      
      Parameters
      ----------
      x : sequence, ndarray
          Position of the particle
      t : float, int
          Time t at which we calculate f.
      qv : sequence, ndarray
          Relative velocity vector of the particle wrt the field. Needed
          since f(q(0,t),X(t),t) depends on q(0,t).

      Returns
      -------
      f : float
          Horizontal component of f(q(0,t),X(t),t)
      g : float
          Vertical component of f(q(0,t),X(t),t)
      '''
      
      coeff          = (1.0 / self.p.R) - 1.0
      
      u, v           = self.vel.get_velocity(x[0], x[1], t)
      ux, uy, vx, vy = self.vel.get_gradient(x[0], x[1], t)
      
      ut, vt         = self.vel.get_dudt(x[0], x[1], t)
      
      f              = coeff * ut + (coeff * u - qv[0]) * ux + \
                                    (coeff * v - qv[1]) * uy
      g              = coeff * vt + (coeff * u - qv[0]) * vx + \
                                    (coeff * v - qv[1]) * vy
      return f, g, u, v


  def calculate_v(self, q0, t):
      '''
      Function that calculates the vector that includes the nonlinear part.
      
      More info in Notes.

      Parameters
      ----------
      q0 : sequence, ndarray
          q(x_i, t) vector for which the first two entries are the
                    relative velocity of the particle.

      Returns
      -------
      vec : ndarray
          vector v.
      '''
      if self.FD_Tag == 2:
          # Second order implementation
          coeff        = 2.0 / (2.0 + self.p.gamma * self.psi0)
      
          f_n, g_n, u_n, v_n     = self.calculate_f(q0[-2:], t, q0[:2])
      
          vec          = np.zeros([1, 2*self.N+2])[0]
          vec[0]       = coeff * f_n
          vec[1]       = coeff * g_n
          vec[-2]      = u_n
          vec[-1]      = v_n
      
      elif self.FD_Tag == 4:
      
          # Fourth order implementation     
          f_n, g_n, u_n, v_n     = self.calculate_f(q0[-2:], t, q0[:2])
          
          vec          = np.zeros([1, 2*self.N+2])[0]
          vec[:2*self.N] = f_n * self.f_coeff + g_n * self.g_coeff
          vec[-2:]     = np.array([u_n, v_n]) #np.append(vec, np.array([u_n, v_n]))
            
      return vec
  
  
  def Rfun(self, t, q):
    '''
    Right hand side of the function
    '''
    return self.A @ q + self.calculate_v(q, t)

    
  def solve_k0(self, t, q):
    '''
    System to obtain k0
    '''
    if self.parallel_flag == False:
        solution = self.LU_psi.solve(self.Rfun(t, q))
    else:
        solution = sparse.linalg.solve(self.Psi, self.Rfun(t, q))
    
    return solution


  def solve_ki(self, t, qk):
    '''
    Function that needs to be solved in the nonlinear solver to obtain k_i's
    '''
    return lambda k: self.Psi @ k - self.Rfun(t, qk + (self.dt/4.) * k)

  def update(self):
    
    tol =1e-10
    
    # ESDIRK4(3)6L[2]SA
    k0 = self.solve_k0(self.time, self.q_n)
    k1 = newton_krylov(self.solve_ki(self.time + self.dt * self.c_vec[1],
                                     self.q_n  + k0 * self.dt * self.A_mat[1,0]),
                       k0, f_tol=tol)
    k2 = newton_krylov(self.solve_ki(self.time + self.dt * self.c_vec[2],
                                     self.q_n  + k0 * self.dt * self.A_mat[2,0]+\
                                                 k1 * self.dt * self.A_mat[2,1]),
                       k1, f_tol=tol)
    k3 = newton_krylov(self.solve_ki(self.time + self.dt * self.c_vec[3],
                                     self.q_n  + k0 * self.dt * self.A_mat[3,0]+\
                                                 k1 * self.dt * self.A_mat[3,1]+\
                                                 k2 * self.dt * self.A_mat[3,2]),
                       k2, f_tol=tol)
    k4 = newton_krylov(self.solve_ki(self.time + self.dt * self.c_vec[4],
                                     self.q_n  + k0 * self.dt * self.A_mat[4,0]+\
                                                 k1 * self.dt * self.A_mat[4,1]+\
                                                 k2 * self.dt * self.A_mat[4,2]+\
                                                 k3 * self.dt * self.A_mat[4,3]),
                       k3, f_tol=tol)
    k5 = newton_krylov(self.solve_ki(self.time + self.dt,
                                     self.q_n  + k0 * self.dt * self.A_mat[5,0]+\
                                                 k1 * self.dt * self.A_mat[5,1]+\
                                                 k2 * self.dt * self.A_mat[5,2]+\
                                                 k3 * self.dt * self.A_mat[5,3]+\
                                                 k4 * self.dt * self.A_mat[5,4]),
                       k4, f_tol=tol)

    self.q_n = self.q_n + k0 * self.dt * self.b_vec[0] +\
                          k1 * self.dt * self.b_vec[1] +\
                          k2 * self.dt * self.b_vec[2] +\
                          k3 * self.dt * self.b_vec[3] +\
                          k4 * self.dt * self.b_vec[4] +\
                          k5 * self.dt/4.
    
    self.pos_vec = np.vstack([self.pos_vec, self.q_n[-2:]])
    self.q0_vec  = np.vstack([self.q0_vec, self.q_n[:2]])
    self.q1_vec  = np.vstack([self.q1_vec, self.q_n[2:4]])

    self.time += self.dt

    return self.q_n 