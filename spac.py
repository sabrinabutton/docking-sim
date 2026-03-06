from dockinglib.simulator import Simulator
from dockinglib.mpc import MPCController
from dockinglib.visualizer import Visualizer
from dockinglib.config import SystemConfig
from dockinglib.errtrack import ErrorTracker  # Import ErrorTracker
from dockinglib.path import ManeuverGenerator
from dockinglib.data import Pose
import numpy as np

def spac():
    
    global_config = SystemConfig.from_yaml("config.yaml")
    sim = Simulator(global_config)
    mpc = MPCController(global_config)
    viz = Visualizer(global_config)
    error_tracker = ErrorTracker() 
    
    start_point = Pose(sim.state.x, sim.state.y, sim.state.psi)
    dock_point = global_config.simulation.dock_pose
    
    # Only used if we do vanilla MPC
    mgen = ManeuverGenerator(global_config)
    maneuver = mgen.get_docking_maneuver(start_point, dock_point, [10, 10, 1, 1])
    
    print("Configs loaded and dock point saved. Starting simulation.")
    
    for t in range(global_config.simulation.total_timesteps):
        # Compute control for current maneuver
        #u_control = mpc.solve_with_fixed_maneuver(sim.state, maneuver)
        u_control, maneuver = mpc.solve(sim.state, dock_point, maneuver)
    
        error_tracker.record_error(sim.state, maneuver[0], t)  # Record the error
        sim.step(u_control)

        if t % global_config.viz.timestep_skip == 0:
            viz.update(sim.state, maneuver, dock_point)
            
    error_tracker.plot_errors()  # Plot errors after simulation
    print("Exiting without error.")
    
if __name__ == "__main__":
    spac()