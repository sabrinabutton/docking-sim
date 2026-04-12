"""
    data.py
    -------
    Holds Position, Velocity, and TrajectoryPoint clases.
    Position and Velocity are defined simply for type safety; they are just 1x3 numpy vectors.

"""
import numpy as np

class Position(np.ndarray):
    """A 3-element numpy array representing [x, y, psi] with property accessors."""
    
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype=float).view(cls)
        if obj.shape != (3,):
            raise ValueError("Position must be a 3-element array.")
        return obj

class Velocity(np.ndarray):
    """A 3-element numpy array representing [u, v, r] with property accessors."""
    
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype=float).view(cls)
        if obj.shape != (3,):
            raise ValueError("Velocity must be a 3-element array.")
        return obj
        
class TrajectoryPoint:
    """A point on a docking trajectory containing a position and velocity."""
    def __init__(self, eta: Position, v: Velocity):
        self.eta = eta
        self.v = v
    