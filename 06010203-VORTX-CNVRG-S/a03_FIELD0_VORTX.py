#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:12:39 2020

@author: cfg4065
"""

# import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Analytical(velocity_field):

  def __init__(self, omega):
    self.omega    = omega
    self.limits   = False
    self.T        = 1. / self.omega
    self.L        = 1.
    self.U        = self.L / self.T
    
  def get_velocity(self, x, y, t):
    '''
    Parameters
    ----------
      x, y : floats
          Nondimensional position at which we want to obtain the velocity.
      t : float
          Nondimensional time at which we want to obtain the velocity.

    Returns
    -------
      u, v : floats
          Nondimensional velocity.
          
          [u; v] / U = T * omega * [-y / L; x / L]
    '''
    
    u = -y * self.omega * self.T
    v =  x * self.omega * self.T
    
    return u, v
  
  def get_gradient(self, x, y, t):
    
    '''
    Parameters
    ----------
      x, y : floats
          Nondimensional position at which we want to obtain the gradient.
      t : float
          Nondimensional time at which we want to be obtain the gradient.

    Returns
    -------
      ux, uy, vx, vy : floats
          Nondimensional velocity gradients
          (derivatives of the velocities w.r.t. space).
          
          [[ux, uy], [vx, vy]] = omega * T * [[0, 1], [1, 0]]
    '''
    
    ux =  0.
    uy = -1. * self.omega * self.T
    vx =  1. * self.omega * self.T
    vy =  0.
    
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
    '''
    Parameters
    ----------
      x, y : floats
          Nondimensional position at which we want to be obtain the time derivative.
      t : float
          Nondimensional time at which we want to be obtain the time derivative.

    Returns
    -------
      ut, vt : floats
          Nondimensional time derivatives of the velocity.
          
          [ut, vt] = [0, 0]
    '''
    
    ut = 0.
    vt = 0.
    
    return ut, vt