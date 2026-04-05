import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. IMPORTS FROM YOUR LIBRARY
# ==========================================
from dockinglib.path import (
    Pose,
    ManeuverGenerator
)

# A simple mock for the config object your generator expects
class MockConfig:
    def __init__(self):
        self.maneuver_generator = None

# ==========================================
# 2. DEBUG PLOTTING LOGIC
# ==========================================
def plot_maneuver(generator, dock_point, params, label, color):
    """Generates a maneuver and plots the path with heading arrows."""
    # Generate the path starting from point index 0
    path = generator.generate_maneuver(dock_point, 0, params)
    
    # Check if path generation failed (handling both lists and numpy arrays)
    if path is None or len(path) == 0:
        print(f"[{label}] Path generation returned an empty array.")
        return
    
    # Extract coordinates and headings
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    psis = [p.psi for p in path]
    
    # Plot the path line and the points
    plt.plot(xs, ys, label=f"{label} (Path)", color=color, marker='o', markersize=4, linestyle='--')
    
    # Calculate vector components for the heading arrows (u = x-direction, v = y-direction)
    us = np.cos(psis)
    vs = np.sin(psis)
    
    # Plot the headings using quiver
    plt.quiver(xs, ys, us, vs, color=color, alpha=0.6, scale=15, width=0.005, 
               headwidth=3, headlength=4)

if __name__ == "__main__":
    # Initialize your generator
    config = MockConfig()
    generator = ManeuverGenerator(config)
    
    # The endpoint we are aiming for (e.g., origin, pointing exactly to the right -> psi=0)
    dock_point = Pose(x=0.0, y=0.0, psi=0.0)
    
    # Define a few sets of parameters to debug
    # Format: ([r_approach, theta_approach, r_berth, theta_berth], Label, Color)
    test_cases = [
        ([10.0, np.pi/4, 5.0, np.pi/4], "Positive Curves", "blue"),
        ([-8.0, np.pi/3, -4.0, np.pi/4], "Negative Curves", "red"),
        ([5.0, np.pi/2, -5.0, np.pi/2], "S-Curve", "green")
    ]
    
    # Setup the plot
    plt.figure(figsize=(10, 8))
    
    # Plot each test case
    for params, label, color in test_cases:
        plot_maneuver(generator, dock_point, params, label, color)
        
    # Mark the docking point prominently
    plt.plot(dock_point.x, dock_point.y, 'k*', markersize=15, label="Dock Point")
    plt.quiver(dock_point.x, dock_point.y, np.cos(dock_point.psi), np.sin(dock_point.psi), 
               color='k', scale=10, width=0.008, label="Dock Heading")
    
    # Formatting the plot
    plt.title("Maneuver Generator Path Debugger")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # VERY IMPORTANT: Ensures angles and arcs look geometrically correct!
    
    plt.show()