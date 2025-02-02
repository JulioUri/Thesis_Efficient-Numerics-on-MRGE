import numpy as np
import scipy as scp
import scipy.sparse.linalg as spla
from scipy.optimize import newton, root, newton_krylov, broyden1, fsolve, broyden1

from progressbar import progressbar
from scipy.sparse.linalg import spsolve
from scipy import sparse
from a00_PMTERS_CONST import mr_parameter

'''
The class "maxey-riley_trapezoidal" given below calculates the trajectory and
velocity of particles whose dynamics is governed by the FULL MAXEY-RILEY
by using an approach by a 2nd order FD + 2nd order Trapezoidal rule methods.

The 2nd order FD used is given in M. Koleva (2006) and it is used for
differential problems in unbounded domains.

The class "update_particle" is provided for the parallelization of the
calculations (since the ammount of particles to be calculated could
be huge for some cases ~10^4).
'''

'''
###############################################################################
########################FULL MAXEY RILEY (Trapezoidal)#########################
###############################################################################
'''

class maxey_riley_trapezoidal(object):

  def __init__(self, tag, x, v, velocity_field, z_v, control, dt, tini,
               particle_density=1, fluid_density=1, particle_radius=1,
               kinematic_viscosity=1, time_scale=1):
      '''
      Calculates particles trajectory and relative velocity by using a 2nd
      order method (Trapezoidal rule) for the time integration and
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
          
      Parameters: tag: int
                      Natural number used for the identification of the
                      particle.
                  
                  x: float, sequence, or ndarray
                      Initial position of the particle
                  
                  v: float, sequence, or ndarray
                      Initial absolute velocity of the particle
                
                  velocity_field: class
                      Class which includes methods that define the
                      velocity field, i.e. get_velocity(), get_gradient(),
                      get_du_dt
                      
                      This is used for the definition of the vector 
                      U(X(t)) and the flow function f(q(0,t), X(t), t).
                      
                      Check notes for further information.
                
                  z_v: sequence, or ndarray
                      Discretized pseudospace.
                
                  control: float, int
                      Parameter "c" in the logarithmic map that produces
                      the Quasi-Uniform Grid from the Uniform Grid.
                      
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
      
      self.vel     = velocity_field
      u0, v0       = velocity_field.get_velocity(x[0], x[1], tini)
                                     # flow's velocity at initial position.
      
      # q_n = [q(x_i, tini), p(x_i, tini)] vector
      # (semidiscretized vector or rel. velocities on space grid)
      self.q_n     = np.zeros([1, 2*self.N])[0]
      self.q_n[0]  = v[0] - u0
      self.q_n[1]  = v[1] - v0
      
      self.q_n     = np.append(self.q_n, x)
      
      # "mr_parameter" is a class that has all the methods that calculate
      # all the parameters needed for the MRE.
      # For example: alpha will be called as self.p.alpha
      self.p       = mr_parameter(particle_density, fluid_density,
                                  particle_radius, kinematic_viscosity,
                                  time_scale)
      
      self.pos_vec = np.copy(x)
      
      # Relative velocity vectors for first and second pseudospatial nodes.
      # This is needed for the calculation of the Forces acting on the particle.
      self.q0_vec  = np.array([self.q_n[0], self.q_n[1]])
      self.q1_vec  = np.array([self.q_n[2], self.q_n[3]])
      
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
      self.dt      = dt
      
      # Calculate matrix A of the Semidiscrete system (2nd order Centered Diff)
      self.calculate_A()
      
      # Calculate matrices M of the fully discrete system (Trap. rule)
      self.calculate_M_matrices()
      
      M_x          = lambda x: self.M_left @ x
      self.Precond = spla.LinearOperator((2*self.N + 2, 2*self.N + 2), matvec=M_x)

  
  def calculate_A(self):
      '''
      This method obtains the matrix A that is obtained from
          - either the 2nd order semidiscretized system with the
            Quasi-Unifrom Grid from Koleva or,
          - the 4th semidiscretized system developed by us with
            the Quasi-Uniform Grid from Koleva (and Fazio).
      
      All entries are defined in the notes.
      
      The matrix has entries in certain diagonals (mainly main diagonal and
      the ones above and below it), therefore the matrix A is built as
      a sparce matrix with from the vector of its diagonals, i.e.
      we will use "scp.sparse.diags" to build A.
      
      Returns
      -------
      None, all data saved in global variables
      '''
      
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
                  np.log((2.0*self.N_inf-2.0*elem+1.0)/(2.0*self.N_inf-2.0*elem-1.0))
            
          omegai          = self.c * \
                  np.log((4.0*self.N_inf-4.0*elem-1.0)/(4.0*self.N_inf-4.0*elem-3.0))
                
          mui             = self.c * \
                  np.log((4.0*self.N_inf-4.0*elem+3.0)/(4.0*self.N_inf-4.0*elem+1.0))
        
        
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
      diagonals    = np.array([diag0, diag2up, diag2dw, diag_ones],dtype=object)
      diag_pos     = np.array([0, 2, -2, -2*self.N])
      
      self.A       = scp.sparse.diags(diagonals, diag_pos,
                                          shape=(2*self.N + 2, 2*self.N + 2),
                                          dtype='float64')
      


  def calculate_M_matrices(self):
      '''
      Create matrices used in the full discretization.
      
      Returns
      -------
      None, all data saved in global variables
      '''
      
      Id           = np.eye(2*self.N + 2)
      
      self.M_left  = sparse.csr_matrix(Id - self.A * self.dt / 2.0)
      self.M_right = sparse.csr_matrix(Id + self.A * self.dt / 2.0)



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



  def calculate_v(self, t, q0):
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
      
      # Second order implementation
      coeff        = 2.0 / (2.0 + self.p.gamma * self.psi0)
  
      f_n, g_n, u_n, v_n     = self.calculate_f(q0[-2:], t, q0[:2])
  
      vec          = np.zeros([1, 2*self.N+2])[0]
      vec[0]       = coeff * f_n
      vec[1]       = coeff * g_n
      vec[-2]      = u_n
      vec[-1]      = v_n
      
      return vec



  def func(self, q_guess):
      '''
      Function on which we muss use a root finding algorithm.
      This is the function that represents the whole discretized system.
      Since we are using the Trapezoidal rule, which is an algorithm
      with an implicit part, we require this nonlinear solver for it, since
      there is some nonlinearity in the system (vector v)

      Parameters
      ----------
      q_guess : sequence, ndarray
          Root guess

      Returns
      -------
      zero : ndarray
          Solution of what we obtain from using q_guess as a guess of the
          root. In case q_guess is the right solution, the return is a vector
          of zeros.
      '''
      
      vec_np1      = self.calculate_v(self.t_np1, q_guess)
      
      LHS          = self.M_left @ q_guess #self.M_left.dot(q_guess)
      RHS          = self.lin_RHS + (self.dt / 2.0) * (self.vec_n + vec_np1)
      
      return RHS - LHS



  def update(self):
      '''
      Method that calculates the solution at the next time step
      by using a root finding algorithm.
      
      We use here the root finding algorithm given in one of these
      built-in functions:
          - root
          - newton
          - newton_krylov
          
      # DANIEL's advice:
      # In case of failure of all three previous methods, one could
      # use an approximation of the Jacobian, either obtained by hand or
      # obtained by Broyden's method.
      
      As a guess of the root we substitute the implicit part of v{n+1},
      by the explicit part, i.e. v^{n} and calculate the root.
      
      More info in the notes.
      
      Returns
      -------
      solution[0]: ndarray
          Solution to the nonlinear system.

      '''
      
      scp.special.seterr(underflow='ignore')
      np.seterr(under='ignore')
      
      # Calculate guess
      self.vec_n   = self.calculate_v(self.time, self.q_n)
      self.lin_RHS = self.M_right @ self.q_n #self.M_right.dot(self.q_n)
      
      RHS          = self.lin_RHS + self.dt * self.vec_n
      q_n_guess    = spsolve(self.M_left, RHS)
      
      self.t_np1   = self.time + self.dt
      
      # Run Root-finding algorithm
      tolerance    = 1e-10
      iter_limit   = 5000
      
      solution     = newton_krylov(self.func,
                                   q_n_guess,
                                   maxiter=iter_limit,
                                   inner_M=self.Precond,
                                   f_tol=tolerance)
      
      self.q0_vec  = np.vstack([self.q0_vec, solution[:2]])
      self.q_n     = solution
      self.q1_vec  = np.vstack([self.q1_vec, solution[2:4]])

      output       = solution

      self.pos_vec = np.vstack([self.pos_vec, solution[-2:]])
      self.time   += self.dt

      return output


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