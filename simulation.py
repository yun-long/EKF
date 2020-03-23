"""
A Simple Pendulum

# ----------------------
# p = pivot point
# ----------------------
#           p           
#           | \
#           |  \
#           |   \
#           |    \
#           |     \
#           |      \
#           |       o
#           o 
"""

import numpy as np

#
kTheta = 0
kDotTheta = 1

class Pendulum_v0(object):
    #
    def __init__(self, pivot_point, dt):
        self.s_dim = 2
        self.a_dim = 0
        
        #
        self._damping = 0.1
        self._mass = 2.0
        self._gz = 9.81
        self._dt = dt
        self.pivot_point = pivot_point # e.g., np.array([2.0, 0.0, 2.0])
        
        self._state = np.zeros(shape=self.s_dim)
        self._state[0] = np.pi/4
        
        # initial state
        self._theta_box = np.array([-0.8, 0.8]) * np.pi
        self._dot_theta_box = np.array([-0.1, 0.1]) * np.pi

        self.length = 2.0  # distance between pivot point to the gate center
        self.width = 1.0   # gate width (for visualization only)
        self.height = 0.5  # gate heiht (for visualization only)
            
        #
        self._t = 0.0
    
    def reset(self, init_theta):
        self._state[kTheta] = init_theta[0]
        self._state[kDotTheta] = init_theta[1]
        #
        self._t = 0.0
        return self._state

    def run(self,):
        self._t = self._t + self._dt
        
        # rk4 int
        M = 4
        DT = self._dt/M
        
        X = self._state
        for _ in range(M):
            k1 = DT * self._f(X)
            k2 = DT * self._f(X + 0.5 * k1)
            k3 = DT * self._f(X + 0.5 * k2)
            k4 = DT * self._f(X + k3)
            #
            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0
        #
        self._state = X
        return self._state[0] + np.random.normal(loc=0, scale=0.1)

    def _f(self, state):
        #
        theta = state[0]
        dot_theta = state[1]
        # return np.array([dot_theta, \
        #     -((self._gz/self.length)*np.sin(theta)+(self._damping/self._mass)*dot_theta)])
        return np.array([dot_theta, -(self._gz/self.length)*np.sin(theta)])

    def get_state(self,):
        return self._state
        
  