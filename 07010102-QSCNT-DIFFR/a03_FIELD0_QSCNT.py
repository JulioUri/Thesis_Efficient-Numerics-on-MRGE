#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:12:39 2020

@author: cfg4065
"""

from a03_FIELD0_00000 import velocity_field

class velocity_field_Quiescent(velocity_field):

  def __init__(self):
    self.limits  = False
    self.T       = 1.
    self.L       = 1.
    self.U       = self.L / self.T
    self.x_left  = None
    self.x_right = None
    self.y_down  = None
    self.y_up    = None
  
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
    '''
    u = 0.0
    v = 0.0
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
    '''
    ux =  0.0
    uy =  0.0
    vx =  0.0
    vy =  0.0
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
    '''
    ut = 0.0
    vt = 0.0
    return ut, vt