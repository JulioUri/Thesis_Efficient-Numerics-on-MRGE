import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Oscillatory(velocity_field):
    
    def __init__(self):
        self.limits  = False
        self.Lambda  = 5 # / (2.0 * np.pi)
        self.T       = 1. / self.Lambda
        self.U       = 0.5 # horizontal component of the velocity
        self.L       = self.U * self.T
    
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
              
              [u, v] / U = [1, sin(lambda * t)]
        '''
        
        u = 1.
        v = np.sin(t)
        
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
              
              [[ux, uy], [vx, vy]] = [[0, 0], [0, 0]]
        '''
        
        ux, uy = 0., 0.
        vx, vy = 0., 0.
        
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
              
              [ut, vt] = [0, T * lambda * cos(lambda * t)]
        '''
        
        ut = 0.0
        vt = np.cos(t)
        
        return ut, vt