import numpy as np
from .data import OtterState
from .config import *

M_U, M_V, I_Z = 60.0, 110.0, 18.0
D_U, D_V, D_R = 50.0, 180.0, 40.0
B = 0.85

class Simulator:
    
    def __init__(self, config: SystemConfig):
        self.config = config.simulation
        self.time = 0.0
        self.state = self.config.start_state
        
    def _current(self):
        Vcx = self.config.disturbance_amplitude_x * (1.0 + 1.2 * np.sin(1.5 * self.time) + 0.4 * np.sin(6.0 * self.time))
        Vcy = self.config.disturbance_amplitude_y * (1.0 + 1.0 * np.cos(1.3 * self.time) + 0.5 * np.sin(5.5 * self.time))
        return Vcx, Vcy
    
    def _get_derivatives(self, u): # Note: Could pull the observer into another class that does state estimation
        tr = np.clip(u[0], -100, 100) 
        tl = np.clip(u[1], -100, 100)

        Vcx, Vcy = self._current()
        
        du = (1/M_U) * ((tl + tr) - D_U*self.state.surge + M_V*self.state.sway*self.state.yaw_rate)
        dv = (1/M_V) * (-D_V*self.state.sway - M_U*self.state.surge*self.state.yaw_rate)
        dr = (1/I_Z) * (0.5*B*(tr - tl) - D_R*self.state.yaw_rate)

        dx = self.state.surge*np.cos(self.state.psi) - self.state.sway*np.sin(self.state.psi) + Vcx
        dy = self.state.surge*np.sin(self.state.psi) + self.state.sway*np.cos(self.state.psi) + Vcy
        dpsi = self.state.yaw_rate

        return OtterState(dx, dy, dpsi, du, dv, dr)
    
    def step(self, u):
        self.time += self.config.integration_step
        # Use 2nd-order integrator (RK2) for better accuracy
        dt = self.config.integration_step
        k1 = self._get_derivatives(u)
        temp_state = self.state + k1 * 0.5 * dt 
        old_state = self.state
        self.state = temp_state
        k2 = self._get_derivatives(u)
        self.state = old_state + k2 * dt