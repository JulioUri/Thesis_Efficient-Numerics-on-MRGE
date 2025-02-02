from a00_PMTERS_CONST import mr_parameter
import numpy as np
import numpy.polynomial.chebyshev as cheb

'''
###############################################################################
##############################OSCILLATORY FLOW#################################
###############################################################################
'''
class maxey_riley_relaxing(object):

  def __init__(self, tag, x, v, tini,
               particle_density, fluid_density, particle_radius,
               kinematic_viscosity, time_scale):
      self.tag       = tag               # particle name/tag
      self.x         = np.copy(x)        # particle position
      self.v         = np.copy(v)        # particle velocity
      self.p         = mr_parameter(particle_density, fluid_density,
                                    particle_radius, kinematic_viscosity,
                                    time_scale)
      self.tini      = tini
      self.pos_vec   = np.copy(x)
      self.vel_vec   = np.copy(v)
      
      Nz             = 100
      index_v        = np.arange(0, Nz)
      
      self.k_hat_v   = (1.0 - np.cos(index_v * np.pi / Nz)) - 1.0
      self.k_v       = (1.0 + self.k_hat_v) / (1.0 - self.k_hat_v)
      
  
  def solve(self, t):
      
      # Calculate velocity
      # Add initial position
      vxresult   = np.copy(self.v[0])
      vyresult   = np.copy(self.v[1])
      
      vintfun    = lambda k: self.p.gamma * k**2. * np.exp(-k**2.0 * t) / \
                  ((k**2.0 - self.p.alpha)**2.0 + (k * self.p.gamma)**2.0)
      
      v_vec      = (2.0 / np.pi) * vintfun(self.k_v) * (2. / (1. - self.k_hat_v)**2.0)
      
      vcoeff     = cheb.chebfit(self.k_hat_v, v_vec, len(self.k_hat_v) - 1)
      vcoeff_int = cheb.chebint(vcoeff)
      
      vintgrl    = cheb.chebval(1.0, vcoeff_int) - cheb.chebval(-1.0, vcoeff_int)
      
      # Add influence of the velocity field (considered cte in horizontal component)
      vxresult   = self.v[0] * vintgrl
      vyresult   = self.v[1] * vintgrl
      
      V_vec      = np.array([vxresult, vyresult])
      self.vel_vec = np.vstack((self.vel_vec, V_vec))
      
      
      # Calculate trajectory
      # Add initial position
      xresult    = np.copy(self.x[0])
      yresult    = np.copy(self.x[1])
      
      yintfun    = lambda k: self.p.gamma * (np.exp(-k**2.0 * self.tini) - np.exp(-k**2.0 * t)) / \
                  ((k**2.0 - self.p.alpha)**2.0 + (k * self.p.gamma)**2.0)
      
      y_vec      = (2.0 / np.pi) * yintfun(self.k_v) * (2. / (1. - self.k_hat_v)**2.0)
      
      ycoeff     = cheb.chebfit(self.k_hat_v, y_vec, len(self.k_hat_v) - 1)
      ycoeff_int = cheb.chebint(ycoeff)
      
      yintgrl     = cheb.chebval(1.0, ycoeff_int) - cheb.chebval(-1.0, ycoeff_int)
      
      # Add influence of the velocity field (considered cte in horizontal component)
      xresult   += self.v[0] * yintgrl
      yresult   += self.v[1] * yintgrl
      
      Y_vec      = np.array([xresult, yresult])
      self.pos_vec = np.vstack((self.pos_vec, Y_vec))
      