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
        
        if len(maneuver) > 0:
            self.waypoint_marker.set_data([maneuver[0].x], [maneuver[0].y])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()