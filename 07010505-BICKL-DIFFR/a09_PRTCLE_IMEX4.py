import numpy as np
import scipy as scp

from progressbar import progressbar
from scipy.sparse.linalg import spsolve
from scipy import sparse as sp

from a00_PMTERS_CONST import mr_parameter

'''
The class "maxey-riley_imex3" given below calculates the trajectory and
velocity of particles whose dynamics is governed by the FULL MAXEY-RILEY
by using an approach by a 2nd order FD + 3nd order ImEx methods.

The class "update_particle" is provided for the parallelization of the
calculations (since the ammount of particles to be calculated could
be huge for some cases ~10^4).
'''

'''
###############################################################################
##########################FULL MAXEY RILEY (IMEX4)#############################
###############################################################################
'''

def update_particle(particle):
  results    = particle.update()
  return results



class maxey_riley_imex(object):

  def __init__(self, tag, x, v, velocity_field, z_v, control, dt, tini,
               particle_density=1, fluid_density=1, particle_radius=1,
               kinematic_viscosity=1, time_scale=1, IMEXOrder = 4,
               FDOrder = 4, parallel_flag = False):
      '''
      Calculates particles trajectory and relative velocity by using a 3nd
      order ImEx method (*SPEDIFY METHODS*) for the time integration and
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
           
      IMEXOrder: int
           Convergence order of the Time integrator, which is an IMEX method
           
           Values to choose: 1, 2, 3 or 4
           
      FDOrder: int
           Convergence order of the Space integrator, which is a finite
           difference method.
           
           Values to choose: 2 or 4
           
           For the value 2, we use the FD approximation created by M. Koleva
           (2006), based on Fazio's method (see R. Fazio et al (2014)),
           which is used for unbounded differential problems.
           
           For the value 4, we use the method Julio developed, based on M.
           Koleva and R. Fazio's schemes but extended as a 4th order method
           for differential problems in unbounded domains. This method
           consists on a CFD scheme for the unbounded problem.
           Further explanation in notes.
      
      parallel_flag: Bool
           Flag for parallel coding.
           
           If True, it means that we are calculating many particle trajectories
           in parallel.
           
           If False, particle trajectories are calculated in serial.
           
           This is important because the sparse.linalg.splu built-in
           function works well in serial but not in parallel, since it is
           written in C and cannot be pickled. Therefore, a different approach
           had to be used: solving the system directly, without the LU
           decomposition.
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
      
      self.coeff1  = (1.0 / self.p.R) - 1.0
      
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
      
      # Calculate matrix A of the Semidiscrete system (2nd order Centered Diff)
      self.calculate_A()
      
      # Decide which butcher tableau/IMEX scheme will be used
      self.IMEX_Tag = IMEXOrder
      
      # If order is 4, then create matrix of coeff for 4th order method
      if IMEXOrder == 1:          
          self.ExpMat  = np.array([[0., 0.],
                                   [1., 0.]])
          
          self.ImpMat  = np.array([[0., 0.],
                                   [0., 1.]])
          
          self.b_vec   = np.array([0., 1.])
          self.c_vec   = np.array([0., 1.])
      
      elif IMEXOrder == 2:
          self.ExpMat  = np.array([[0.,    0.],
                                   [1./2., 0.]])
          
          self.ImpMat  = np.array([[0., 0.],
                                   [0., 1./2.]])
          
          self.b_vec   = np.array([0., 1.])
          self.c_vec   = np.array([0., 1./2.])
      
      elif IMEXOrder == 3:
          gamma_coeff  = (3. + np.sqrt(3.)) / 6.
          self.ExpMat  = np.array([[0.,               0.,                    0.],
                                   [gamma_coeff,      0.,                    0.],
                                   [gamma_coeff - 1., 2.*(1. - gamma_coeff), 0.]])
          
          self.ImpMat  = np.array([[0., 0.,                   0.],
                                   [0., gamma_coeff,          0.],
                                   [0., 1. - 2.* gamma_coeff, gamma_coeff]])
          
          self.b_vec   = np.array([0., 1./2., 1./2.])
          self.c_vec   = np.array([0., gamma_coeff, 1. - gamma_coeff])
          
      elif IMEXOrder == 4:
          self.ExpMat  = np.array([[0.,0.,0.,0.,0.,0.],
                              [1./2.,0.,0.,0.,0.,0.],
                              [13861./62500.,6889./62500.,0.,0.,0.,0.],
                              [-116923316275./2393684061468.,-2731218467317./15368042101831.,9408046702089./11113171139209.,0.,0.,0.],
                              [-451086348788./2902428689909.,-2682348792572./7519795681897.,12662868775082./11960479115383.,3355817975965./11060851509271.,0.,0.],
                              [ 647845179188./3216320057751.,73281519250./8382639484533.,552539513391./3454668386233.,3354512671639./8306763924573.,4040./17871.,0.]])
          
          self.ImpMat  = np.array([[0.,0.,0.,0.,0.,0.],
                              [1./4.,1./4.,0.,0.,0.,0.],
                              [8611./62500.,-1743./31250.,1./4.,0.,0.,0.],
                              [5012029./34652500.,-654441./2922500.,174375./388108.,1./4.,0.,0.],
                              [15267082809./155376265600.,-71443401./120774400.,730878875./902184768.,2285395./8070912.,1./4.,0.],
                              [82889./524892.,0.,15625./83664.,69875./102672.,-2260./8211.,1./4.]])
          
          self.b_vec   = np.array([82889./524892.,0,15625./83664.,69875./102672.,-2260./8211,1./4.])
          self.c_vec   = np.array([0., 1./2., 83./250., 31./50., 17./20., 1.])
      
      # Implicit Linear System that needs to be solved in each stage of runge kutta
      self.calculate_LA()
      
      # splu function works well in serial but not in parallel, since it is
      # written in C and cannot be pickled. Therefore, a different approach
      # has to be undertaken.
      if parallel_flag == False:
         self.LU = sp.linalg.splu(self.LA) # LU factorisation of the linear system 
         if FDOrder == 4:
             self.LU_psi = sp.linalg.splu(self.Psi)
      
  
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
          
          self.A       = sp.diags(diagonals, diag_pos,
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
      
          M1           = sp.diags(M1diagonals, Mdiag_pos,
                                         shape  = (2*self.N, 2*self.N),
                                         format = 'csc',
                                         dtype  = 'float64')
          M2           = sp.diags(M2diagonals, Mdiag_pos,
                                         shape  = (2*self.N, 2*self.N),
                                         format = 'csc',
                                         dtype  = 'float64')
          
          invM1        = sp.linalg.inv(M1)
          
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
      
          B1           = sp.diags(B1diagonals, Bdiag_pos,
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
      
          B2           = sp.diags(B2diagonals, Bdiag_pos,
                                          shape  = (2*self.N, 2*self.N),
                                          format = 'csc',
                                          dtype  = 'float64')
          
          # Calculate P matrix
          P            = np.zeros((2 * self.N, 2 * self.N))
          P[0,0]       = 1.0
          P[1,1]       = 1.0
      
          P            = sp.csr_matrix(P)
          
          Psi          = M2 - ( 2.0 * self.c / (3.0 * self.p.gamma) ) * B2 @ invM1 @ P
          self.Psi     = sp.block_diag((Psi, np.eye(2))).tocsr()
          
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
          self.A       = sp.coo_matrix(A)
          
          aux_mat      = B2 @ invM1
          aux_mat.data[abs(aux_mat.data)<1e-25] = 0.0
          aux_mat.eliminate_zeros()
          
          # Calculate matrix of coefficients of f
          f_coeff  = (-2.0 * self.c / (3.0 * self.p.gamma)) * aux_mat
          f_coeff.data[abs(f_coeff.data) < 1e-25] = 0.0
          f_coeff.eliminate_zeros()
          self.f_coeff  = f_coeff[:,0].toarray().T[0]
          self.g_coeff  = f_coeff[:,1].toarray().T[0]
      
  def calculate_LA(self):

      Id = np.eye(2*self.N + 2)

      if self.IMEX_Tag == 1:
          
          if self.FD_Tag == 2:
              self.LA = Id - self.dt*self.A
          elif self.FD_Tag == 4:
              self.LA = self.Psi - self.dt*self.A
      
      elif self.IMEX_Tag == 2:
          
          if self.FD_Tag == 2:
              self.LA = Id - self.dt/2.*self.A
          elif self.FD_Tag == 4:
              self.LA = self.Psi - self.dt/2.*self.A
      
      elif self.IMEX_Tag == 3:
          gamma = (np.sqrt(3)+3)/6
          
          if self.FD_Tag == 2:
              self.LA = Id - self.dt*gamma*self.A
          elif self.FD_Tag == 4:
              self.LA = self.Psi - self.dt*gamma*self.A
      
      elif self.IMEX_Tag == 4:
          
          if self.FD_Tag == 2:
              self.LA = Id - self.dt/4.*self.A
          elif self.FD_Tag == 4:
              self.LA = self.Psi - (self.dt/4.)*self.A        
      
      self.LA = sp.csc_matrix(self.LA)

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
      
      u, v           = self.vel.get_velocity(x[0], x[1], t)
      ux, uy, vx, vy = self.vel.get_gradient(x[0], x[1], t)
      
      ut, vt         = self.vel.get_dudt(x[0], x[1], t)
      
      f              = self.coeff1 * ut + (self.coeff1 * u - qv[0]) * ux +\
                                (self.coeff1 * v - qv[1]) * uy
      g              = self.coeff1 * vt + (self.coeff1 * u - qv[0]) * vx +\
                                (self.coeff1 * v - qv[1]) * vy
                                
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
      
          x_n          = q0[-2:]
      
          f_n, g_n, u_n, v_n     = self.calculate_f(x_n, t, q0[:2])
      
          vec          = np.zeros([1, 2*self.N+2])[0]
          vec[0]       = coeff * f_n
          vec[1]       = coeff * g_n
          vec[-2]      = u_n
          vec[-1]      = v_n
      
      elif self.FD_Tag == 4:
          # Fourth order implementation      
          x_n          = q0[-2:]
      
          f_n, g_n, u_n, v_n     = self.calculate_f(x_n, t, q0[:2])
          
          vec          = np.zeros([1, 2*self.N+2])[0]
          vec[:2*self.N] = f_n * self.f_coeff + g_n * self.g_coeff
          vec[-2:]     = np.array([u_n, v_n])
                  
      return vec

      
  # Leon's update method
  def update(self):

      if self.IMEX_Tag == 1:
          
          if self.FD_Tag == 2:
              
              if self.parallel_flag == False:
                  self.q_n = self.LU.solve(self.q_n + self.dt * self.b_vec[1] * self.calculate_v(self.q_n, self.time + self.dt * self.c_vec[0]))
              else:
                  self.q_n = sp.linalg.spsolve(self.LA,
                                           self.q_n + self.dt * self.b_vec[1] * self.calculate_v(self.q_n, self.time + self.dt * self.c_vec[0]))
          
          elif self.FD_Tag == 4:
              
              nstages = 2
              ndof    = self.A.shape[0]
              stages  = np.zeros((nstages, ndof))
              A_vec   = np.zeros((nstages, ndof))
              v_vec   = np.zeros((nstages, ndof))
              
              PsiQ    = self.Psi @ self.q_n
              
              # Solve for stages
              for i in range(0, nstages):
                  
                  if i == 0:
                      stages[i,:] = self.q_n.copy()
                  else:
                      # Construct RHS
                      rhs   = np.copy(PsiQ)
                      
                      for j in range(0,i):
                          rhs += self.dt * (self.ImpMat[i,j] * A_vec[j] +\
                                            self.ExpMat[i,j] * v_vec[j])
                              
                      if self.parallel_flag == False:
                           stages[i,:] = self.LU.solve(rhs)
                      else:
                           stages[i,:] = scp.sparse.linalg.spsolve(self.LA, rhs)
                      
                  A_vec[i,:]  = self.A @ stages[i,:]
                  A_vec[i,:][abs(A_vec[i,:]) < 1e-50] = 0.0
                  v_vec[i,:]  = self.calculate_v(stages[i,:], self.time + self.c_vec[i]*self.dt)
                  v_vec[i,:][abs(v_vec[i,:]) < 1e-50] = 0.0
                  
              # Update
              rhs   = PsiQ
              for i in range(0, nstages):
                  rhs += self.dt * self.b_vec[i] * (A_vec[i] + v_vec[i])
              
              if self.parallel_flag == False:
                  self.q_n = self.LU_psi.solve(rhs)
              else:
                  self.q_n = scp.sparse.linalg.spsolve(self.LA_psi,  rhs)
              
              
      elif self.IMEX_Tag == 2:
          
          if self.FD_Tag == 2:
              H1 = self.calculate_v(self.q_n, self.time + self.dt * self.c_vec[0])
            
              if self.parallel_flag == False:
                  qtemp = self.LU.solve(self.q_n + self.dt * self.ExpMat[1,0] * H1)
              else:
                  qtemp = sp.linalg.spsolve(self.LA,
                                        self.q_n + self.dt * self.ExpMat[1,0] * H1)
              
              K1 = self.A @ qtemp
              K1[abs(K1)<1e-20] = 0.0
              
              H2 = self.calculate_v(qtemp, self.time + self.dt * self.c_vec[1])
              
              self.q_n = self.q_n + self.dt * self.b_vec[1] * (K1+H2)
              
          elif self.FD_Tag == 4:
              
              nstages = 2
              ndof    = self.A.shape[0]
              stages  = np.zeros((nstages, ndof))
              A_vec   = np.zeros((nstages, ndof))
              v_vec   = np.zeros((nstages, ndof))
              
              PsiQ    = self.Psi @ self.q_n
              
              # Solve for stages
              for i in range(0, nstages):
                  
                  if i == 0:
                      stages[i,:] = self.q_n.copy()
                  else:
                      # Construct RHS
                      rhs   = np.copy(PsiQ)
                      
                      for j in range(0,i):
                          rhs += self.dt * (self.ImpMat[i,j] * A_vec[j] +\
                                            self.ExpMat[i,j] * v_vec[j])
                              
                      if self.parallel_flag == False:
                           stages[i,:] = self.LU.solve(rhs)
                      else:
                           stages[i,:] = scp.sparse.linalg.spsolve(self.LA, rhs)
                      
                  A_vec[i,:]  = self.A @ stages[i,:]
                  A_vec[i,:][abs(A_vec[i,:]) < 1e-50] = 0.0
                  v_vec[i,:]  = self.calculate_v(stages[i,:], self.time + self.c_vec[i]*self.dt)
                  v_vec[i,:][abs(v_vec[i,:]) < 1e-50] = 0.0
                  
              # Update
              rhs   = PsiQ
              for i in range(0, nstages):
                  rhs += self.dt * self.b_vec[i] * (A_vec[i] + v_vec[i])
              
              if self.parallel_flag == False:
                  self.q_n = self.LU_psi.solve(rhs)
              else:
                  self.q_n = scp.sparse.linalg.spsolve(self.LA_psi,  rhs)
        
      elif self.IMEX_Tag == 3:
          
          if self.FD_Tag == 2:
              H1 = self.calculate_v(self.q_n, self.time + self.dt * self.c_vec[0])
              
              if self.parallel_flag == False:
                  q1 = self.LU.solve(self.q_n + self.dt * self.ExpMat[1,0] * H1)
              else:
                  q1 = sp.linalg.spsolve(self.LA,
                                     self.q_n + self.dt * self.ExpMat[1,0] * H1)
                
              K1 = self.A @ q1
              K1[abs(K1)<1e-20] = 0.0
              H2 = self.calculate_v(q1, self.time + self.dt * self.c_vec[1])
              
              if self.parallel_flag == False:
                  q2 = self.LU.solve(self.q_n + self.dt * self.ImpMat[2,1] * K1 +\
                                                self.dt * self.ExpMat[2,0] * H1 +\
                                                self.dt * self.ExpMat[2,1] * H2)
              else:
                  q2 = sp.linalg.spsolve(self.LA,
                                     self.q_n + self.dt * self.ImpMat[2,1] * K1 +\
                                                self.dt * self.ExpMat[2,0] * H1 +\
                                                self.dt * self.ExpMat[2,1] * H2)
            
              K2 = self.A @ q2
              K2[abs(K2)<1e-20] = 0.0
              H3 = self.calculate_v(q2, self.time + self.dt * self.c_vec[2])
              
              self.q_n += self.dt * (self.b_vec[1] * (H2+K1) + self.b_vec[2] * (H3+K2))
              
          elif self.FD_Tag == 4:
             
             nstages = 3
             ndof    = self.A.shape[0]
             stages  = np.zeros((nstages, ndof))
             A_vec   = np.zeros((nstages, ndof))
             v_vec   = np.zeros((nstages, ndof))
             
             PsiQ    = self.Psi @ self.q_n
             
             # Solve for stages
             for i in range(0, nstages):
                 
                 if i == 0:
                     stages[i,:] = self.q_n.copy()
                 else:
                     # Construct RHS
                     rhs   = np.copy(PsiQ)
                     
                     for j in range(0,i):
                         rhs += self.dt * (self.ImpMat[i,j] * A_vec[j] +\
                                           self.ExpMat[i,j] * v_vec[j])
                             
                     if self.parallel_flag == False:
                          stages[i,:] = self.LU.solve(rhs)
                     else:
                          stages[i,:] = scp.sparse.linalg.spsolve(self.LA, rhs)
                     
                 A_vec[i,:]  = self.A @ stages[i,:]
                 A_vec[i,:][abs(A_vec[i,:]) < 1e-50] = 0.0
                 v_vec[i,:]  = self.calculate_v(stages[i,:], self.time + self.c_vec[i]*self.dt)
                 v_vec[i,:][abs(v_vec[i,:]) < 1e-50] = 0.0
                 
             # Update
             rhs   = PsiQ
             for i in range(0, nstages):
                 rhs += self.dt * self.b_vec[i] * (A_vec[i] + v_vec[i])
             
             if self.parallel_flag == False:
                 self.q_n = self.LU_psi.solve(rhs)
             else:
                 self.q_n = scp.sparse.linalg.spsolve(self.LA_psi,  rhs)
          
      elif self.IMEX_Tag == 4:
          
          if self.FD_Tag == 2:
        
              K0 = self.A @ self.q_n
              K0[abs(K0)<1e-50] = 0.0
              H1 = self.calculate_v(self.q_n, self.time + self.c_vec[0])
                
              if self.parallel_flag == False:
                  q1 = self.LU.solve(self.q_n + self.dt * self.ImpMat[1,0] * K0 +\
                                                self.dt * self.ExpMat[1,0] * H1)
              else:
                  q1 = sp.linalg.spsolve(self.LA,
                                     self.q_n + self.dt * self.ImpMat[1,0] * K0 +\
                                                self.dt * self.ExpMat[1,0] * H1)
                
              K1 = self.A @ q1
              K1[abs(K1)<1e-50] = 0.0
              H2 = self.calculate_v(q1, self.time + self.dt * self.c_vec[1])
                
              if self.parallel_flag == False:
                  q2 = self.LU.solve(self.q_n + self.dt * self.ImpMat[2,0] * K0 +\
                                                self.dt * self.ImpMat[2,1] * K1 +\
                                                self.dt * self.ExpMat[2,0] * H1 +\
                                                self.dt * self.ExpMat[2,1] * H2)
              else:
                  q2 = sp.linalg.spsolve(self.LA,
                                     self.q_n + self.dt * self.ImpMat[2,0] * K0 +\
                                                self.dt * self.ImpMat[2,1] * K1 +\
                                                self.dt * self.ExpMat[2,0] * H1 +\
                                                self.dt * self.ExpMat[2,1] * H2)
                    
              K2 = self.A @ q2
              K2[abs(K2)<1e-50] = 0.0
              H3 = self.calculate_v(q2, self.time + self.dt * self.c_vec[2])
                
              if self.parallel_flag == False:
                  q3 = self.LU.solve(self.q_n + self.dt * self.ImpMat[3,0] * K0 +\
                                                self.dt * self.ImpMat[3,1] * K1 +\
                                                self.dt * self.ImpMat[3,2] * K2 +\
                                                self.dt * self.ExpMat[3,0] * H1 +\
                                                self.dt * self.ExpMat[3,1] * H2 +\
                                                self.dt * self.ExpMat[3,2] * H3)
              else:
                  q3 = sp.linalg.spsolve(self.LA,
                                     self.q_n + self.dt * self.ImpMat[3,0] * K0 +\
                                                self.dt * self.ImpMat[3,1] * K1 +\
                                                self.dt * self.ImpMat[3,2] * K2 +\
                                                self.dt * self.ExpMat[3,0] * H1 +\
                                                self.dt * self.ExpMat[3,1] * H2 +\
                                                self.dt * self.ExpMat[3,2] * H3)
                    
              K3 = self.A @ q3
              K3[abs(K3)<1e-50] = 0.0
              H4 = self.calculate_v(q3, self.time + self.dt * self.c_vec[3])
                
              if self.parallel_flag == False:
                  q4 = self.LU.solve(self.q_n + self.dt * self.ImpMat[4,0] * K0 +\
                                                self.dt * self.ImpMat[4,1] * K1 +\
                                                self.dt * self.ImpMat[4,2] * K2 +\
                                                self.dt * self.ImpMat[4,3] * K3 +\
                                                self.dt * self.ExpMat[4,0] * H1 +\
                                                self.dt * self.ExpMat[4,1] * H2 +\
                                                self.dt * self.ExpMat[4,2] * H3 +\
                                                self.dt * self.ExpMat[4,3] * H4)
              else:
                  q4 = sp.linalg.spsolve(self.LA,
                                     self.q_n + self.dt * self.ImpMat[4,0] * K0 +\
                                                self.dt * self.ImpMat[4,1] * K1 +\
                                                self.dt * self.ImpMat[4,2] * K2 +\
                                                self.dt * self.ImpMat[4,3] * K3 +\
                                                self.dt * self.ExpMat[4,0] * H1 +\
                                                self.dt * self.ExpMat[4,1] * H2 +\
                                                self.dt * self.ExpMat[4,2] * H3 +\
                                                self.dt * self.ExpMat[4,3] * H4)
                    
              K4 = self.A @ q4
              K4[abs(K4)<1e-50] = 0.0
              H5 = self.calculate_v(q4, self.time + self.dt * self.c_vec[4])
                
              if self.parallel_flag == False:
                  q5 = self.LU.solve(self.q_n + self.dt * self.ImpMat[5,0] * K0 +\
                                                self.dt * self.ImpMat[5,2] * K2 +\
                                                self.dt * self.ImpMat[5,3] * K3 +\
                                                self.dt * self.ImpMat[5,4] * K4 +\
                                                self.dt * self.ExpMat[5,0] * H1 +\
                                                self.dt * self.ExpMat[5,1] * H2 +\
                                                self.dt * self.ExpMat[5,2] * H3 +\
                                                self.dt * self.ExpMat[5,3] * H4 +\
                                                self.dt * self.ExpMat[5,4] * H5)
              else:
                  q5 = sp.linalg.spsolve(self.LA,
                                     self.q_n + self.dt * self.ImpMat[5,0] * K0 +\
                                                self.dt * self.ImpMat[5,2] * K2 +\
                                                self.dt * self.ImpMat[5,3] * K3 +\
                                                self.dt * self.ImpMat[5,4] * K4 +\
                                                self.dt * self.ExpMat[5,0] * H1 +\
                                                self.dt * self.ExpMat[5,1] * H2 +\
                                                self.dt * self.ExpMat[5,2] * H3 +\
                                                self.dt * self.ExpMat[5,3] * H4 +\
                                                self.dt * self.ExpMat[5,4] * H5)
                    
              K5 = self.A @ q5
              K5[abs(K5)<1e-50] = 0.0
              H6 = self.calculate_v(q5, self.time + self.dt * self.c_vec[5])
                
              self.q_n += self.dt * (self.b_vec[0] * (K0+H1) +\
                                     self.b_vec[2] * (K2+H3) +\
                                     self.b_vec[3] * (K3+H4) +\
                                     self.b_vec[4] * (K4+H5) +\
                                     self.b_vec[5] * (K5+H6))
       
          elif self.FD_Tag == 4:
              nstages = 6
              ndof    = self.A.shape[0]
              stages  = np.zeros((nstages, ndof))
              A_vec   = np.zeros((nstages, ndof))
              v_vec   = np.zeros((nstages, ndof))
              
              PsiQ    = self.Psi @ self.q_n
              
              # Solve for stages
              for i in range(0, nstages):
                  
                  if i == 0:
                      stages[i,:] = self.q_n.copy()
                  else:
                      # Construct RHS
                      rhs   = np.copy(PsiQ)
                      
                      for j in range(0,i):
                          rhs += self.dt * (self.ImpMat[i,j] * A_vec[j] +\
                                            self.ExpMat[i,j] * v_vec[j])
                              
                      if self.parallel_flag == False:
                           stages[i,:] = self.LU.solve(rhs)
                      else:
                           stages[i,:] = scp.sparse.linalg.spsolve(self.LA, rhs)
                      
                  A_vec[i,:]  = self.A @ stages[i,:]
                  A_vec[i,:][abs(A_vec[i,:]) < 1e-50] = 0.0
                  v_vec[i,:]  = self.calculate_v(stages[i,:], self.time + self.c_vec[i]*self.dt)
                  v_vec[i,:][abs(v_vec[i,:]) < 1e-50] = 0.0
                  
              # Update
              rhs   = PsiQ
              for i in range(0, nstages):
                  rhs += self.dt * self.b_vec[i] * (A_vec[i] + v_vec[i])
              
              if self.parallel_flag == False:
                  self.q_n = self.LU_psi.solve(rhs)
              else:
                  self.q_n = scp.sparse.linalg.spsolve(self.LA_psi,  rhs)
          
      self.pos_vec = np.vstack([self.pos_vec, self.q_n[-2:]])
      self.q0_vec  = np.vstack([self.q0_vec,  self.q_n[:2]])
      self.q1_vec  = np.vstack([self.q1_vec,  self.q_n[2:4]])

      self.time += self.dt



  def forces_fd(self, time_vec):
      '''
      Method that calculates Forces acting on the particle.

      Parameters
      ----------
      time_vec : sequence, ndarray
          Time array

      Returns
      -------
      F_PT : ndarray
          Force term obtained by the material derivative term.
      F_St : ndarray
          Force term obtained by Stokes drag term.
      F_HT : ndarray
          Force term obtained by the Basset History term.

      '''
      coeff          = self.p.gamma / (2.0 * self.omega0)
      
      F_PT           = np.array([])
      F_St           = np.array([])
      F_HT           = np.array([])
      for tt in progressbar(range(0, len(time_vec))):
          
          u, v           = self.vel.get_velocity(self.pos_vec[tt][0],
                                             self.pos_vec[tt][1],
                                             time_vec)
          ut, vt         = self.vel.get_dudt(self.pos_vec[tt][0],
                                             self.pos_vec[tt][1],
                                             time_vec)
          ux, uy, vx, vy = self.vel.get_gradient(self.pos_vec[tt][0],
                                             self.pos_vec[tt][1],
                                             time_vec)
      
          du_dt          = np.array([ut, vt])
          u_gradu        = np.array([ u*ux + v*uy, u*vx + v*vy ])
          
          F_PT           = np.append(F_PT, ( 1.0/self.p.R ) * \
                            np.linalg.norm( du_dt + u_gradu ))
          
          F_St           = np.append(F_St, self.p.alpha * \
                                 np.linalg.norm( self.q0_vec[tt] ))
          
          if tt != len(time_vec) - 1:
              dq0_dt         = (self.q0_vec[tt+1] - self.q0_vec[tt])/self.dt
          else:
              dq0_dt         = (self.q0_vec[tt] - self.q0_vec[tt-1])/self.dt
      
          bracket        = self.q1_vec[tt] - self.q0_vec[tt] - \
                              self.psi0 * self.omega0 * dq0_dt
      
          F_HT           = np.append(F_HT, coeff * np.linalg.norm(bracket))
      
      return F_PT, F_St, F_HT