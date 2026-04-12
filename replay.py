import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from dockinglib.config import SystemConfig
from dockinglib.visualizer import Visualizer
from dockinglib.data import Position, Velocity, TrajectoryPoint

def load_run_data(folder_path, method_name):
    """Loads and parses the CSV data back into NumPy arrays and TrajectoryPoints."""
    eta_file = os.path.join(folder_path, f"{method_name}_eta.csv")
    v_file = os.path.join(folder_path, f"{method_name}_v.csv")
    m_file = os.path.join(folder_path, f"{method_name}_maneuver.csv")

    if not os.path.exists(eta_file) or not os.path.exists(m_file):
        raise FileNotFoundError(f"Data for '{method_name}' not found.")

    eta_data, v_data, maneuver_data = [], [], []
    
    # Load ETA
    with open(eta_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            eta_data.append(Position(np.array([float(row[0]), float(row[1]), float(row[2])])))

    # Load Velocity
    with open(v_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            v_data.append(Velocity(np.array([float(row[0]), float(row[1]), float(row[2])])))

    # Load Maneuver 
    with open(m_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            raw_list = json.loads(row[0])
            step_maneuver = [TrajectoryPoint(Position(np.array(pt[0])), Velocity(np.array(pt[1]))) for pt in raw_list]
            maneuver_data.append(step_maneuver)

    return eta_data, v_data, maneuver_data

def play_comparison(folder_path, methods, playback_speed=1.0):
    """Plays back multiple datasets synchronously."""
    print(f"Loading data from {folder_path}...")
    
    all_data = {}
    max_frames = 0
    
    # Load all requested methods
    for m in methods:
        try:
            eta, v, man = load_run_data(folder_path, m)
            all_data[m] = {'eta': eta, 'v': v, 'maneuver': man}
            max_frames = max(max_frames, len(eta))
            print(f"  Loaded {m}: {len(eta)} frames")
        except FileNotFoundError:
            print(f"  Skipping {m}: Files not found.")

    valid_methods = list(all_data.keys())
    if not valid_methods:
        print("Error: No valid data found to play.")
        return

    # Setup Visualizer with only the loaded methods
    config = SystemConfig.from_yaml("config.yaml")
    viz = Visualizer(config.model, methods=valid_methods)
    
    plt.ion()
    plt.show()

    sim_dt = 0.1 
    frame_delay = sim_dt / playback_speed

    print("\nStarting comparison playback...")
    
    for i in tqdm(range(max_frames), desc="Playing Timeline"):
        if not plt.fignum_exists(viz.fig.number):
            print("\nPlayback interrupted.")
            break
            
        state_dict = {}
        
        for m in valid_methods:
            # If this specific method converged early, hold it at its final frame
            frame_idx = min(i, len(all_data[m]['eta']) - 1)
            
            eta = all_data[m]['eta'][frame_idx]
            v = all_data[m]['v'][frame_idx]
            maneuver = all_data[m]['maneuver'][frame_idx]

            if len(maneuver) == 0:
                maneuver = [TrajectoryPoint(eta, v)]
                
            state_dict[m] = (eta, v, maneuver)

        viz.update(state_dict)
        plt.pause(frame_delay)

    plt.ioff()
    print("Playback finished.")
    
    if plt.fignum_exists(viz.fig.number):
        plt.show(block=True)

if __name__ == "__main__":
    
    TARGET_FOLDER = "data/run_20260412_190813" 
    
    # Pass a list of the algorithms you want to compare on the same plot!
    METHODS_TO_COMPARE = ["vanilla"] 
    
    SPEED = 2.0 
    
    play_comparison(TARGET_FOLDER, METHODS_TO_COMPARE, playback_speed=SPEED)