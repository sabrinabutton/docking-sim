from dockinglib.simulator import Simulator
from dockinglib.mpc import MPCController
from dockinglib.visualizer import Visualizer
from dockinglib.config import SystemConfig
from dockinglib.errtrack import ErrorTracker
from dockinglib.path import ManeuverGenerator
from dockinglib.data import Pose
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
from tqdm import tqdm
import keyboard  # Provides non-blocking keypress detection

def spac():
    global_config = SystemConfig.from_yaml("config.yaml")
    sim = Simulator(global_config)
    mpc = MPCController(global_config)
    viz = Visualizer(global_config)
    error_tracker = ErrorTracker() 
    
    start_point = Pose(sim.state.x, sim.state.y, sim.state.psi)
    dock_point = global_config.simulation.dock_pose
    
    mgen = ManeuverGenerator(global_config)
    maneuver = mgen.generate_maneuver(dock_point, -1, [7, np.pi/2, 5, np.pi/2])
    
    print("Configs loaded and dock point saved. Starting simulation.")
    
    state_history = []
    maneuver_history = []
    
    # 1. SIMULATION PHASE
    total_steps = global_config.simulation.total_timesteps
    for t in tqdm(range(total_steps), desc="Simulating Time Steps"):
        
        u_control = mpc.solve_with_fixed_maneuver(sim.state, maneuver)
        # u_control, maneuver = mpc.solve(sim.state, dock_point, maneuver)
        error_tracker.record_error(sim.state, maneuver[0], t)
        
        state_history.append(copy.deepcopy(sim.state))
        maneuver_history.append(copy.deepcopy(maneuver))
        
        if sim.state.point_idx_achieved == (maneuver.size - 1):
            # Clip t here 
            total_steps = t + 1
            break
        
        sim.step(u_control, maneuver)
        
        
        
    # 2. VISUALIZATION PHASE
    print("\nSimulation complete. Starting visualization playback.")
    print(">>> Press and hold 'q' to stop playback and view the error plot. <<<")
    
    viz_delay = 0.05 
    keep_playing = True
    
    plt.ion()
    
    # Loop continuously until keep_playing becomes False
    while keep_playing:
        for t in range(total_steps):
            
            # Check if the user is pressing 'q'
            if keyboard.is_pressed('q'):
                keep_playing = False
                break  # Break out of the for-loop immediately
                
            if state_history[t].point_idx_achieved + 1 >= (maneuver_history[t].size - 1):
                # Clip t here 
                # total_steps = t + 1
                break    
            
            if t % global_config.viz.timestep_skip == 0:
                viz.update(state_history[t], maneuver_history[t], dock_point)
                plt.pause(viz_delay)
                
        plt.ioff() # Turn interactive mode back off for the final error plot
                
    # 3. ERROR PLOTTING
    print("\nPlayback stopped. Generating error plot...")
    error_tracker.plot_errors() 
    print("Exiting without error.")
    
if __name__ == "__main__":
    spac()