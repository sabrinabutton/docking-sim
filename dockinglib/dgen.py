"""
    dgen.py
    --------------
    Holds the DisturbanceGenerator class and several presets for wind simulation.
"""

import numpy as np
from dockinglib.config import DisturbanceConfig

heavy_storm = {
    "active": True,
    "magnitude": 0.4,       
    "base_dir": np.pi/2,    
    "sweep_angle": np.pi/6, 
    "yaw_mag": 0.1         
}

semilinear_winds = {
    "active": True,
    "magnitude": 0.4,
    "base_dir": np.pi/2,
    "sweep_angle": np.pi/8,
    "yaw_mag": 0.0
}

calm_weather = {"active": False}

presets = {
    "heavy_storm": heavy_storm,
    "semilinear_winds": semilinear_winds,
    "calm_weather": calm_weather,
    "none": None
}

class DisturbanceGenerator:
    """
    Simulates localized, sweeping wind disturbances and rotational chop.
    """
    def __init__(self, config:DisturbanceConfig):
        
        self.config = config
        self.freq = 1.0 
        
        if(presets[self.config.preset] is not None):
            self.active = presets[self.config.preset]["active"]
            self.magnitude = presets[self.config.preset]["magnitude"]
            self.base_dir = presets[self.config.preset]["base_dir"]
            self.sweep_angle = presets[self.config.preset]["sweep_angle"]
            self.yaw_mag = presets[self.config.preset]["yaw_mag"]
        else:
            self.active = self.config.active
            self.magnitude = self.config.magnitude  
            self.base_dir = self.config.base_dir    
            self.sweep_angle = self.config.sweep_angle 
            self.yaw_mag = self.config.yaw_mag

    def get_disturbance(self, t, psi):
        """Calculates the body-frame acceleration caused by the wind at time t."""
        if not self.active:
            return np.zeros(3)

        # Sweeping direction
        current_dir = self.base_dir + (self.sweep_angle * np.sin(self.freq * t))
        
        # Global Wind Vector
        current_mag = self.magnitude * (1.0 + 0.1 * np.random.randn())
        global_acc_x = current_mag * np.cos(current_dir)
        global_acc_y = current_mag * np.sin(current_dir)
        
        # Rotate Global Vector to Body Frame [u, v]
        acc_u =  global_acc_x * np.cos(psi) + global_acc_y * np.sin(psi)
        acc_v = -global_acc_x * np.sin(psi) + global_acc_y * np.cos(psi)
        
        # Rotational disturbance (yaw)
        acc_r = self.yaw_mag * np.sin(self.freq * t * 1.3)
        
        return np.array([acc_u, acc_v, acc_r])