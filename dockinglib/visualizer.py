"""
    vizualizer.py
    --------------
    Creates an animation of an ASV following the planned trajectory while tracing its actual trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt

from .data import Position
from .model import R
from .config import ModelConfig

class Visualizer:
    
    def __init__(self, modelconf:ModelConfig, methods=['vanilla']):
        self.modelconf = modelconf
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-10, 40)
        self.ax.set_ylim(-10, 40)
        self.ax.set_aspect('equal')
        
        self.methods = methods
        
        # Color Palettes for Comparison
        self.colors = {
            'vanilla': {'boat': 'r', 'actual': 'darkred', 'planned': 'lightcoral'},
            'multi-rate': {'boat': 'b', 'actual': 'darkblue', 'planned': 'cornflowerblue'},
            'single-shoot': {'boat': 'g', 'actual': 'darkgreen', 'planned': 'lightgreen'}
        }
        
        self.plots = {}
        self.trajectory_history = {m: [] for m in methods}
        
        legend_handles = []
        legend_labels = []
        
        # Initialize plot elements for each method
        for m in methods:
            c = self.colors.get(m, {'boat': 'k', 'actual': 'k', 'planned': 'gray'})
            
            boat_poly, = self.ax.plot([], [], color=c['boat'], linestyle='-', linewidth=2)
            actual_traj, = self.ax.plot([], [], color=c['actual'], linestyle='-', linewidth=1.5, alpha=0.8)
            planned_traj, = self.ax.plot([], [], color=c['planned'], linestyle='--', linewidth=1.5, alpha=0.6)
            
            self.plots[m] = {
                'boat': boat_poly,
                'actual': actual_traj,
                'planned': planned_traj
            }
            
            legend_handles.extend([boat_poly, planned_traj, actual_traj])
            legend_labels.extend([f'{m} (Boat)', f'{m} (Planned)', f'{m} (Actual)'])

        self.target_marker, = self.ax.plot([], [], 'go', markersize=8)
        legend_handles.append(self.target_marker)
        legend_labels.append('Target Dock')
        
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("SPaC Architecture Comparison")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(legend_handles, legend_labels, loc='upper right', fontsize='small')
        
    def _get_boat_shape(self, eta: Position):
        l = self.modelconf.length / 2
        w = self.modelconf.width / 2
        
        pts = np.array([
            [l, 0], # Nose
            [-l, w], # Back Left
            [-l, -w], # Back Right
            [l, 0] # Nose
        ])
        
        pts_global = (R(eta[2])[0:2, 0:2] @ pts.T).T + np.array([eta[0], eta[1]])
        return pts_global[:, 0], pts_global[:, 1]
        
    def update(self, state_dict):
        """
        Expects state_dict format: {'vanilla': (eta, v, maneuver), 'multi-rate': (eta, v, maneuver)}
        """
        for m, state in state_dict.items():
            if state is None:
                continue
                
            eta, v, maneuver = state
            
            self.trajectory_history[m].append((eta[0], eta[1]))
            
            if len(self.trajectory_history[m]) > 1:
                traj_x = [p[0] for p in self.trajectory_history[m]]
                traj_y = [p[1] for p in self.trajectory_history[m]]
                self.plots[m]['actual'].set_data(traj_x, traj_y)
            
            path_x = [p.eta[0] for p in maneuver]
            path_y = [p.eta[1] for p in maneuver]
            self.plots[m]['planned'].set_data(path_x, path_y)

            boat_shape = self._get_boat_shape(eta)
            self.plots[m]['boat'].set_data(boat_shape[0], boat_shape[1])
            
            if len(maneuver) > 0:
                self.target_marker.set_data([maneuver[-1].eta[0]], [maneuver[-1].eta[1]])
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()