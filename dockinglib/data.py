from dataclasses import dataclass
from enum import Enum, IntEnum

    
@dataclass
class Pose:
    x: float
    y: float
    psi: float
    
@dataclass
class OtterState:
    x: float
    y: float
    psi: float
    surge: float
    sway: float
    yaw_rate: float
    point_idx_achieved: int = -1
    
    def copy(self):
        return OtterState(self.x, self.y, self.psi,
                          self.surge, self.sway, self.yaw_rate)
    
    def __add__(self, other):
        return OtterState(self.x + other.x, self.y + other.y, self.psi + other.psi,
                          self.surge + other.surge, self.sway + other.sway, self.yaw_rate + other.yaw_rate)
        
    def __mul__(self, scalar):
        return OtterState(self.x * scalar, self.y * scalar, self.psi * scalar,
                          self.surge * scalar, self.sway * scalar, self.yaw_rate * scalar)