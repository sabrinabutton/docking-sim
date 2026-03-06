import numpy as np
from .data import Pose

APPROACH_LENGTH = 5
BERTH_LENGTH = 5
TOTAL_LENGTH = APPROACH_LENGTH + BERTH_LENGTH

# Param order
R_APPROACH_IDX = 0
THETA_APPROACH_IDX = 1
R_BERTH_IDX = 2
THETA_BERTH_IDX = 3

class ManeuverGenerator:
    
    def __init__(self, config):
        self.config = config.maneuver_generator
    
    def _get_arc(end, r, theta, pts):
        if pts <= 0:
            return []

        # Determine the ICC based on the ENDPOINT pose
        center_x = end.x - (r * np.sin(end.psi))
        center_y = end.y + (r * np.cos(end.psi))

        # Determine the angular coordinate of the ENDPOINT relative to the ICC
        end_angle = np.arctan2(end.y - center_y, end.x - center_x)
        angles = np.linspace(end_angle - theta, end_angle, int(pts))
        
        r_mag = np.abs(r)
        direction = np.copysign(1.0, r)
        
        return np.array([
            Pose(
                x = center_x + r_mag * np.cos(a),
                y = center_y + r_mag * np.sin(a),
                psi = a + (direction * np.pi / 2)
            )
            for a in angles
        ])
    
    def generate_maneuver(self, dock_point, point_idx_achieved, r_approach, theta_approach, r_berth, theta_berth):
        pts_approach = max(0, (APPROACH_LENGTH - 1) - point_idx_achieved)
        pts_berth = max(0, min(BERTH_LENGTH, TOTAL_LENGTH - 1 - point_idx_achieved))
        
        berth = self._get_arc(dock_point, r_berth, theta_berth, pts_berth)
        
        if(berth.size == 0):
            print("Failure. Berth couldn't generate a path.")
        
        approach = self._get_arc(berth[0], r_approach, theta_approach, pts_approach)
        
        return approach + berth
    
    def generate_maneuver(self, dock_point, point_idx_achieved, params):
        # Assume params are in order
        return self.generate_maneuver(dock_point, point_idx_achieved, 
                                      params[R_APPROACH_IDX], params[THETA_APPROACH_IDX], params[R_BERTH_IDX], params[THETA_BERTH_IDX])
    
    
        
    