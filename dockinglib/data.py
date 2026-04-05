from dataclasses import dataclass
from enum import Enum, IntEnum
import numpy as np
    
@dataclass
class Pose:
    x: float
    y: float
    psi: float
    
    def hypot(self, point: Pose):
        return np.hypot(self.x - point.x, self.y - point.y) 
    
@dataclass
class OtterState:
    x: float
    y: float
    psi: float
    surge: float
    sway: float
    yaw_rate: float
    point_idx_achieved: int
    
    def copy(self):
        return OtterState(self.x, self.y, self.psi,
                          self.surge, self.sway, self.yaw_rate,
                          self.point_idx_achieved)
    
    def __add__(self, other):
        return OtterState(self.x + other.x, self.y + other.y, self.psi + other.psi,
                          self.surge + other.surge, self.sway + other.sway, self.yaw_rate + other.yaw_rate,
                          self.point_idx_achieved)
        
    def __mul__(self, scalar):
        return OtterState(self.x * scalar, self.y * scalar, self.psi * scalar,
                          self.surge * scalar, self.sway * scalar, self.yaw_rate * scalar,
                          self.point_idx_achieved)