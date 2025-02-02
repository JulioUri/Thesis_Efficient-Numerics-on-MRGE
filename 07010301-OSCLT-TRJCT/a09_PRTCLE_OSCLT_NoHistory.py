from a00_PMTERS_CONST import mr_parameter
import numpy as np
from scipy.integrate import quad

'''
###############################################################################
##############################OSCILLATORY FLOW#################################
###############################################################################
'''
class maxey_riley_oscillatory_noHistory(object):

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
      self.pos_vec_nohistory   = np.copy(x)
      
      
      if self.vel.limits == True:
          if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
              raise Exception("Particle's initial position is outside the spatial domain")
  
  
  def solve_noHistory(self, t):
      
      ## Calculate horizontal coordinate
      # Add initial velocity
      expfun     = 1. - np.exp(-(t - self.time) / (self.p.R * self.p.S))
      
      u01, u02   = self.vel.get_velocity(self.x[0], self.x[1], self.time)
      
      # Calculate trajectory
      # Add initial position
      xresult    = np.copy(self.x[0])
      xresult   += self.p.R * self.p.S * (self.v[0] - u01) * expfun +\
                   u01 * (t - self.time)
      
      ## Calculate vertical coordinate      
      c1         = (((1. - self.p.R) * self.vel.Lambda**2. * self.p.S *\
                   np.cos(self.vel.Lambda * self.time) +\
                   (1. + self.vel.Lambda**2. * self.p.R * self.p.S**2.) *\
                   self.vel.Lambda * np.sin(self.vel.Lambda * self.time)) /\
                   (self.vel.Lambda + self.vel.Lambda**3. * self.p.R**2. * self.p.S**2.) -\
                   self.v[1]) * np.exp(self.time / (self.p.R * self.p.S))
      
      c2         = self.x[1] - c1 * self.p.R * self.p.S *\
                   np.exp(-self.time / (self.p.R * self.p.S)) -\
                   (np.sin(self.vel.Lambda * self.time) *\
                   self.vel.Lambda * self.p.S * (1. - self.p.R) -\
                   np.cos(self.vel.Lambda * self.time) *\
                   (1. + self.vel.Lambda**2. * self.p.R * self.p.S**2.)) /\
                   (self.vel.Lambda + self.vel.Lambda**3. * self.p.R**2.*self.p.S**2.)
      
      yresult    = c1 * self.p.R * self.p.S *\
                   np.exp(-t/(self.p.R * self.p.S)) + c2 +\
                   ((np.sin(self.vel.Lambda * t) * self.vel.Lambda *\
                   self.p.S * (1. - self.p.R) - np.cos(self.vel.Lambda * t) *\
                   (1. + self.vel.Lambda**2. * self.p.R * self.p.S**2.)) /\
                   (self.vel.Lambda + self.vel.Lambda**3. * self.p.R**2. * self.p.S**2.))
      
      #Check we are still within the bounds for which we have information
      if self.vel.limits == True:
          if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
              raise Exception("Particle's position exits the spatial domain")
      
      Y_vec      = np.array([xresult, yresult])
      self.pos_vec_nohistory = np.vstack((self.pos_vec_nohistory, Y_vec))  