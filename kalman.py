"""
A Kalman Filter to estimate the state of a pendulum 
(near linear when initial angule is small)
"""
import numpy as np
#
from simulation import Pendulum_v0

class Kalman(object):

    def __init__(self, dt):
        #
        self._mass = 2.0
        self._gz = 9.81
        self._length = 2.0
        self._damping = 0.1
        
        # System Matrix (state)
        self.A = np.array([[1, dt], 
            [-dt*self._gz/self._length, 1]])
        # self.A = np.array([[0, 1], 
        #     [-self._gz/self._length, self._damping/self._mass]])
        # System Matrix (input)
        self.B = np.array([ [0], [0] ])
        # Observation Matrix (state)
        self.C = np.array([ [1, 0] ])
        # Observation Matrix (input)
        self.D = np.array([ [0] ])
        
        # State error covariance matrix
        self.P = np.array([ [1.0, 0.0], [0.0, 1.0] ])
        # Process noise covariance matrix
        self.Q = np.diag([0, 1e-4])
        # Measurement noise covariance
        self.R = np.array([ [1.0] ]) 
        # Indentity matrix
        self.I = np.diag([1.0, 1.0])
        
        #
        self.x = np.array([[0.0], [0.0]])
        self.u = np.array([[0]] )

    def reset(self, x):
        self.x = x

    def predition(self,):
        """
        Predicte prior state and error covariance matrix
        """
        # update the state (prior)
        x_prior = self.A@self.x + self.B@self.u
        self.x = x_prior

        # update the error covariance matrix (prior)
        P_prior = self.A@self.P@self.A.T + self.Q
        self.P = P_prior

    def update(self, y):
        """
        Update state and error covariance matrix
        """
        # compute the Kalman gain
        K = (self.P@self.C.T)/(self.C@self.P@self.C.T + self.R)

        # Update the state (posteriror estimate)
        x_post = self.x + K@(y - self.C@self.x)
        self.x = x_post
        
        # Update the error covariance matrix
        p_post = (self.I - K*self.C)*self.P
        self.P = p_post

    def get_state(self):
        return self.x[:, 0]
    
def main():
    #
    dt = 0.05
    
    #
    kalman = Kalman(dt)
    pend_sim = Pendulum_v0(pivot_point=[1,1,1], dt=dt)
    
    #
    init_theta = np.pi/2
    pend_sim.reset(init_theta=[init_theta, 0])
    kalman.reset(x=[init_theta, 0])
    #
    true_state = []
    pred_state = []
    meas_y = []
    #
    T = np.arange(0.0, 20.0, dt)
    for _ in T:        
        # observe
        x = pend_sim.get_state()
        y = pend_sim.run() 
        
        #
        kalman.predition()
        
        #
        kalman.update(np.array([[y]]))
        hat_x = kalman.get_state()
        
        #
        meas_y.append(y)
        true_state.append(x)
        pred_state.append(hat_x)
    #
    true_state = np.array(true_state)
    pred_state = np.array(pred_state)
    
    #
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(6, 4))
    #
    axes[0].plot(T, true_state[:, 0], label=r"true $\theta$")
    axes[0].plot(T, pred_state[:, 0], label=r"pred $\theta$")
    axes[0].plot(T, meas_y, "r--", label=r"measured $\theta$")
    axes[0].legend()
    axes[0].grid(True)
    #
    axes[1].plot(T, true_state[:, 1], label=r"true $\dot{\theta}$")
    axes[1].plot(T, pred_state[:, 1], label=r"pred $\dot{\theta}$")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

