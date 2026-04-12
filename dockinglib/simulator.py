"""
    simulator.py
    ------------
    Holds the system state and performs RK4 integration.
    Owns the disturbance generator and internal time tracking.
"""

import numpy as np

from .model import OtterModel
from .config import SystemConfig
from .dgen import DisturbanceGenerator

class Simulator:

    def __init__(self, global_config: SystemConfig, model: OtterModel):
        
        self.model = model
        self.dt = global_config.simulation.dt

        self.eta = np.array(global_config.simulation.initial_position)
        self.v = np.array([0.0, 0.0, 0.0], dtype=float)

        self.disturbance = DisturbanceGenerator(global_config.disturbances)

        self.time = 0.0

    # ------------------------------------------------------------------
    # Public disturbance accessor
    # ------------------------------------------------------------------
    def get_current_disturbance(self) -> np.ndarray:
        """Returns the body-frame disturbance acceleration at the current sim time."""
        return self.disturbance.get_disturbance(self.time, self.eta[2])

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def _rk4(self, v, u, dist_accel):
        """RK4 with a fixed disturbance snapshot (computed once per step)."""
        def dynamics_func(_v, _u):
            return self.model.dynamics(_v, _u) + dist_accel

        k1 = dynamics_func(v, u)
        k2 = dynamics_func(v + 0.5 * self.dt * k1, u)
        k3 = dynamics_func(v + 0.5 * self.dt * k2, u)
        k4 = dynamics_func(v + self.dt * k3, u)
        return v + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, u):
        """Advance the simulator by one dt."""
        dist_accel = self.get_current_disturbance()   
        self.v = self._rk4(self.v, u, dist_accel)
        eta_dot = self.model.kinematics(self.eta[2], self.v)
        self.eta = self.eta + eta_dot * self.dt
        self.eta[2] = (self.eta[2] + np.pi) % (2 * np.pi) - np.pi

        self.time += self.dt 