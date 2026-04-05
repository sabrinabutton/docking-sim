import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    
    def __init__(self, config):
        self.config = config.viz
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        
        self.boat_polygon, = self.ax.plot([], [], 'b-', linewidth=2) # The boat hull
        self.path_line, = self.ax.plot([], [], 'r--', alpha=0.5)   # The planned path
        self.trajectory_line, = self.ax.plot([], [], 'grey', linewidth=1)  # Actual trajectory
        self.target_marker, = self.ax.plot([], [], 'go', markersize=8) # The dock
        self.waypoint_marker, = self.ax.plot([], [], 'ko', markersize=4)  # Current target waypoint
        
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Autonomous Docking Simulation")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend([self.boat_polygon, self.path_line, self.trajectory_line, self.target_marker, self.waypoint_marker],
                       ['Boat', 'Planned Path', 'Actual Trajectory', 'Dock', 'Target Waypoint'], loc='upper right')
        
        self.frame_count = 0
        self.trajectory_history = []  
        
        # --- NEW: Initialize text containers ---
        self.waypoint_texts = []  # Holds the text objects for waypoint numbers
        # Create a text box pinned to the top-left of the axes
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                        verticalalignment='top', 
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.ion()
        self.fig.canvas.draw()
        self.fig.show()
        
    def _get_boat_shape(self, state):
        pts = np.array([
            [self.config.boat_length/2, 0], # Nose
            [-self.config.boat_length/2, self.config.boat_width/2], # Back Left
            [-self.config.boat_length/2, -self.config.boat_width/2], # Back Right
            [self.config.boat_length/2, 0] # Nose
        ])
        
        c, s = np.cos(state.psi), np.sin(state.psi)
        R = np.array([[c, -s], [s, c]])
        
        pts_global = (R @ pts.T).T + np.array([state.x, state.y])
        
        return pts_global[:, 0], pts_global[:, 1]
        
    def update(self, state, maneuver, dock_point):
        # Record actual trajectory
        self.trajectory_history.append((state.x, state.y))
        
        if len(self.trajectory_history) > 1:
            traj_x = [p[0] for p in self.trajectory_history]
            traj_y = [p[1] for p in self.trajectory_history]
            self.trajectory_line.set_data(traj_x, traj_y)
        
        path_x = [p.x for p in maneuver]
        path_y = [p.y for p in maneuver]
        self.path_line.set_data(path_x, path_y)

        # Update boat position and orientation
        boat_shape = self._get_boat_shape(state)
        self.boat_polygon.set_data(boat_shape[0], boat_shape[1])
        
        self.target_marker.set_data([dock_point.x], [dock_point.y])
        
        # --- NEW: Update the text box ---
        self.status_text.set_text(f"point_idx_achieved: {state.point_idx_achieved}")
        
        # --- NEW: Update waypoint index numbers ---
        # 1. Clear old waypoint text objects from the plot
        for txt in self.waypoint_texts:
            txt.remove()
        self.waypoint_texts.clear()
        
        if len(maneuver) > 0:
            # 2. Draw new text objects for each waypoint
            for i, p in enumerate(maneuver):
                # Adding a small offset (+0.3) so the number doesn't sit exactly on the line
                txt = self.ax.text(p.x + 0.3, p.y + 0.3, str(i), fontsize=9, color='darkred')
                self.waypoint_texts.append(txt)
            
            # Safe-guard target waypoint logic
            next_wp_idx = min(state.point_idx_achieved + 1, len(maneuver) - 1)
            self.waypoint_marker.set_data([maneuver[next_wp_idx].x], [maneuver[next_wp_idx].y])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()