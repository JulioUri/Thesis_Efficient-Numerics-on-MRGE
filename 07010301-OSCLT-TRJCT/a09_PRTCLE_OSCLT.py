from a00_PMTERS_CONST import mr_parameter
import numpy as np
from scipy.integrate import quad

'''
###############################################################################
##############################OSCILLATORY FLOW#################################
###############################################################################
'''
class maxey_riley_oscillatory(object):

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
      self.time      = tini
      self.pos_vec   = np.copy(x)
      self.vel_vec   = np.copy(v)
      
      
      if self.vel.limits == True:
          if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
              raise Exception("Particle's initial position is outside the spatial domain")
  
  
  def solve(self, t):
      
      ## Calculate horizontal coordinate
      # Add initial position
      xresult  = np.copy(self.x[0])
      
      # Add influence of the velocity field (considered cte in horizontal component)
      u01, u02 = self.vel.get_velocity(self.x[0], self.x[1], 0.0)
      xresult += u01 * t
      
      # Add the influence of the initial velocity (Relaxing particle behaviour)
      vx0      = self.vel_vec[0]
      fx       = lambda k: self.p.gamma * (1.0 - np.exp(-k**2.0 * t)) / \
          ((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)
      
      xresult += (vx0 - u01) * (2.0 / np.pi) * \
          (quad(fx,    0.0,    5.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx,    5.0,   10.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx,   10.0,   50.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx,   50.0,  100.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx,  100.0,  200.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx,  200.0,  400.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx,  400.0,  600.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx,  600.0, 1000.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
           quad(fx, 1000.0, 4000.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0])
      
      ## Calculate vertical coordinate
      # Add initial position
      yresult  = np.copy(self.x[1])
      
      # Add the influence of the background flow, through the relative velocity
      yresult += (1.0 - np.cos(t)) #(1.0 - np.cos(self.vel.Lambda * t)) / self.vel.Lambda
      
      # Add the influence of the initial velocity      
      vy0      = self.vel_vec[1]
      fy1      = lambda k: self.p.gamma * (1.0 - np.exp(-k**2.0 * t)) / \
                  ((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)
      
      yresult += (vy0 - u02) * (2.0 / np.pi) * \
         (quad(fy1,   0.0,    5.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fy1,   5.0,   10.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fy1,  10.0,   30.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fy1,  30.0,   50.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fy1,  50.0,  100.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] +\
          quad(fy1, 100.0,  500.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] +\
          quad(fy1, 500.0, 1000.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0])
      
      # Add influence of the background flow, through the flow function "f".
      coeff    = (1.0 - self.p.R) * 2.0 / ( np.pi * self.p.R ) #(1.0 - self.p.R) * self.vel.Lambda * 2.0 / ( np.pi * self.p.R )
      
      fy2      = lambda k: k**2.0 * self.p.gamma * np.exp(-k**2.0 * t) / \
          (((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)*(k**4.0 + 1.))
      fy3      = lambda k: k**4.0 * self.p.gamma * np.sin(t) / \
          (((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)*(k**4.0 + 1.))
      fy4      = lambda k: k**2.0 * self.p.gamma * np.cos(t) / \
          (((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)*(k**4.0 + 1.))
      
      
      # fy2      = lambda k: k**2.0 * self.p.gamma * np.exp(-k**2.0 * t) / \
      #     (((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)*(k**4.0 + self.vel.Lambda**2.0))
      # fy3      = lambda k: k**4.0 * self.p.gamma * np.sin(t * self.vel.Lambda) / \
      #     (((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)*(k**4.0 + self.vel.Lambda**2.0)*self.vel.Lambda)
      # fy4      = lambda k: k**2.0 * self.p.gamma * np.cos(t * self.vel.Lambda) / \
      #     (((k*self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0)*(k**4.0 + self.vel.Lambda**2.0))
      
      fun      = lambda k: fy2(k) + fy3(k) - fy4(k)
      
      yresult += coeff * \
         (quad(fun,   0.0,    5.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fun,   5.0,   10.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fun,  10.0,   30.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fun,  30.0,   50.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] + \
          quad(fun,  50.0,  100.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] +\
          quad(fun, 100.0,  500.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0] +\
          quad(fun, 500.0, 1000.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0])
      
      x        = np.array([xresult, yresult])
      
      #Check we are still within the bounds for which we have information
      if self.vel.limits == True:
          if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
              raise Exception("Particle's position exits the spatial domain")
      
      self.pos_vec = np.vstack((self.pos_vec, x))
      