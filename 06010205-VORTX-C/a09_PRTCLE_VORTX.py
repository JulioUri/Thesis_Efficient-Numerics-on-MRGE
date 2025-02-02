from a00_PMTERS_CONST import mr_parameter
import numpy as np
from scipy.special import erfc
from scipy.integrate import quad
import h5py

'''
###############################################################################
##############################CONVERGENCE TEST#################################
###############################################################################
'''
class maxey_riley_analytic(object):

  def __init__(self, tag, x, v, tini, velocity_field,
               particle_density, fluid_density, particle_radius,
               kinematic_viscosity, time_scale):
      self.tag       = tag               # particle name/tag
      self.x         = np.copy(x)        # particle position
      self.v         = np.copy(v)        # particle velocity
      self.vel       = velocity_field
      self.p         = mr_parameter(particle_density, fluid_density,
                                    particle_radius, kinematic_viscosity,
                                    time_scale)
      #print(time_scale)
      self.pseud_St  = (particle_radius**2.0) / (9.0 * kinematic_viscosity * time_scale)
      self.time      = tini
      self.pos_vec   = np.copy(x)
      self.vel_vec   = np.copy(v)
      
      A_cnst         = 1.0 / ((2.0 * self.p.beta + 1.0) * self.pseud_St)
      B_cnst         = (3.0 - (1j / self.pseud_St)) / (2.0 * self.p.beta + 1.0)
      C_cnst         = -3.0 / ((2.0 * self.p.beta + 1.0) * \
                                np.sqrt(np.pi * self.pseud_St))
      
      Y1             = self.solve_poly(A_cnst, B_cnst, C_cnst)
      
      P1             = 0.5 * (-C_cnst * np.sqrt(np.pi) - \
                        np.sqrt(C_cnst**2.0 * np.pi - 4.0 * A_cnst + 4.0 * Y1))
      P2             = 0.5 * (-C_cnst * np.sqrt(np.pi) + \
                        np.sqrt(C_cnst**2.0 * np.pi - 4.0 * A_cnst + 4.0 * Y1))
      Q1             = 0.5 * ( Y1 - np.sqrt( Y1**2.0 - 4.0 * B_cnst ))
      Q2             = 0.5 * ( Y1 + np.sqrt( Y1**2.0 - 4.0 * B_cnst ))
      
      X1             = 0.5 * (-P1 + np.sqrt(P1**2.0 - 4.0 * Q1))
      X2             = 0.5 * (-P1 - np.sqrt(P1**2.0 - 4.0 * Q1))
      X3             = 0.5 * (-P2 + np.sqrt(P2**2.0 - 4.0 * Q2))
      X4             = 0.5 * (-P2 - np.sqrt(P2**2.0 - 4.0 * Q2))
      
      self.X         = np.array([X1, X2, X3, X4])
      
      Z0             = x[0] + x[1] * 1j
      U0             = v[0] + v[1] * 1j
      
      den            = np.array([])
      for ii in range(0, 4):
          product      = 1.0
          for jj in range(0, 4):
              if ii != jj:
                  product    *= self.X[ii] - self.X[jj]
          assert product != 0.0, "Product of elements in denominator equal to zero."
          den          = np.append(den, product)
          
      self.A         = np.array([])
      sumAi          = 0.0
      sumAi_Xi       = 0.0
      sumAiXi        = 0.0
      for ii in range(0, 4):
          Ai         = (U0 * (self.X[ii]**2.0 - C_cnst * np.sqrt(np.pi) * self.X[ii]) - \
                        B_cnst * Z0) / den[ii]
          self.A     = np.append(self.A, Ai)
          sumAi     += Ai
          sumAi_Xi  += Ai / self.X[ii]
          sumAiXi   += Ai * self.X[ii]
      
      if abs(sumAi.real) < 1e-14 and abs(sumAi.imag) < 1e-14:
           sumAi      = 0.0
      
      assert abs(sumAi.real) < 1e-13 or \
              abs(sumAi.imag) < 1e-13,"Sum of A_i/X_i 's must be equal to initial position"
      
      assert abs(sumAi_Xi.real - self.x[0]) < 1e-13 or \
              abs(sumAi_Xi.imag - self.x[1]) < 1e-13,"Sum of A_i/X_i 's must be equal to initial position"
      
      assert abs(sumAiXi.real - self.v[0]) < 1e-13 or \
              abs(sumAiXi.imag - self.v[1]) < 1e-13,"Sum of A_i*X_i 's must be equal to initial velocity"
      
      if self.vel.limits == True:
          if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
              raise Exception("Particle's initial position is outside the spatial domain")
  
  def solve_poly(self, A, B, C):
      p        = np.array([1.0, -A, -(4.0 * B + np.pi * C**2.0 * 1j),
                           4.0 * A * B + C**2.0 * np.pi * (1.0 - B)])
      
      root_vec = np.roots(p)
      
      real_max = -np.inf
      for ii in range(0, len(root_vec)):
          if real_max.real < root_vec[ii].real:
              real_max = root_vec[ii]
    
      return real_max
  
  def exp_x_erfc(self, z):
      np.seterr(all="raise")
      # Avoid Overflow using log
      result  = z**2.0 + np.log(erfc(z))
      result  = np.exp(result)
      return result
  
  def solve(self, t):
      
      pos_result = 0.0
      vel_result = 0.0
      for ii in range(4):
          coeff   = self.A[ii] / self.X[ii]
          #print("X_i: " + str(self.X[ii]))
          pos_result += coeff * self.exp_x_erfc(-self.X[ii] * np.sqrt(t))
          vel_result += self.A[ii] * self.X[ii] * np.exp(t * self.X[ii]**2.0) *\
                          erfc(-self.X[ii] * np.sqrt(t))
      
      x            = np.array([pos_result.real, pos_result.imag])
      v            = np.array([vel_result.real, vel_result.imag])
      
      self.pos_vec = np.vstack((self.pos_vec, x))
      self.vel_vec = np.vstack((self.vel_vec, v))
      