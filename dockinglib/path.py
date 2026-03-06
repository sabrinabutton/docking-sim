import numpy as np
from .data import Pose, PointType, Status

class ManeuverGenerator:
    
    def __init__(self, config):
        self.config = config.maneuver_generator
        
    def _get_arc(self, l, c, start, reverse=False):
        r = (c / 2) + (l**2 / (8 * c))
        direction = -1 if reverse else 1
        
        center_x = start.x - (direction * r * np.sin(start.psi))
        center_y = start.y + (direction * r * np.cos(start.psi))

        anchor_angle = np.arctan2(start.y - center_y, start.x - center_x)
        delta_angle = 2 * np.arcsin(l / (2 * r))
        
        angles = np.linspace(
            anchor_angle - (direction * delta_angle) if reverse else anchor_angle,
            anchor_angle if reverse else anchor_angle + delta_angle,
            self.config.interpolation_count
        )
        
        return [
            Pose(
                x = center_x + r * np.cos(a),
                y = center_y + r * np.sin(a),
                psi = a + (direction * np.pi / 2),
                p_type = PointType.APPROACH if reverse else PointType.BERTH
            )
            for a in angles
        ]
        
    def _get_start_to_approach(self, start_point, approach_entry):
        xs = np.linspace(start_point.x, approach_entry.x, self.config.interpolation_count)
        ys = np.linspace(start_point.y, approach_entry.y, self.config.interpolation_count)
        psis = np.linspace(start_point.psi, approach_entry.psi, self.config.interpolation_count)
        
        start_to_approach = [
            Pose(x=x, y=y, psi=p, p_type=PointType.SETUP) 
            for x, y, p in zip(xs, ys, psis)
        ]
        
        return start_to_approach
        
        
    def _get_docking_maneuver(self, start_point, dock_point, l_a, l_b, c_a, c_b, verbose=True, status:Status=Status.START):    
        berth_arc = self._get_arc(l_b, c_b, dock_point)
        berth_point = berth_arc[-1]
        approach_arc = self._get_arc(l_a, c_a, berth_point, reverse=True)
        approach_point = approach_arc[0]
        start_to_approach = self._get_start_to_approach(start_point, approach_point)
        
        path = []
        
        if status <= Status.START: path += start_to_approach
        if status <= Status.SETUP_ACHIEVED: path += approach_arc
        if status <= Status.APPROACH_ACHIEVED: path += berth_arc[::-1]
        
        return path
        
    def get_docking_maneuver(self, start_point, dock_point, params, status:Status=Status.START):
        return self._get_docking_maneuver(start_point, dock_point, params[0], params[1], params[2], params[3])
    
    
        
    