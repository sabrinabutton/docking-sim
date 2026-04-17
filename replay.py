import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from dockinglib.config import SystemConfig
from dockinglib.visualizer import Visualizer
from dockinglib.data import Position, Velocity, TrajectoryPoint

def load_run_data(folder_path, method_name):
    """Loads and parses the CSV data back into NumPy arrays, TrajectoryPoints, and Disturbance vectors."""
    eta_file = os.path.join(folder_path, f"{method_name}_eta.csv")
    v_file = os.path.join(folder_path, f"{method_name}_v.csv")
    m_file = os.path.join(folder_path, f"{method_name}_maneuver.csv")
    d_file = os.path.join(folder_path, f"{method_name}_disturbances.csv") # New disturbance file

    if not os.path.exists(eta_file) or not os.path.exists(m_file):
        raise FileNotFoundError(f"Data for '{method_name}' not found.")

    eta_data, v_data, maneuver_data, dist_data = [], [], [], []
    
    # 1. Load Position Data
    with open(eta_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            eta_data.append(Position(np.array([float(row[0]), float(row[1]), float(row[2])])))

    # 2. Load Velocity Data
    with open(v_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            v_data.append(Velocity(np.array([float(row[0]), float(row[1]), float(row[2])])))

    # 3. Load Maneuver/Planned Path Data
    with open(m_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            raw_list = json.loads(row[0])
            step_maneuver = [TrajectoryPoint(Position(np.array(pt[0])), Velocity(np.array(pt[1]))) for pt in raw_list]
            maneuver_data.append(step_maneuver)

    # 4. Load Disturbance Data (if it exists)
    if os.path.exists(d_file):
        with open(d_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                # Assuming saved as dist_u, dist_v, dist_r
                dist_data.append(np.array([float(row[0]), float(row[1]), float(row[2])]))
    else:
        # Fallback to zeros if file is missing
        dist_data = [np.zeros(3) for _ in range(len(eta_data))]

    return eta_data, v_data, maneuver_data, dist_data

def play_comparison(folder_path, methods, playback_speed=1.0, paper_mode=False):
    """Plays back multiple datasets synchronously or renders a static paper plot."""
    print(f"Loading data from {folder_path}...")
    
    all_data = {}
    max_frames = 0
    
    for m in methods:
        try:
            # Now unpacking 4 items including 'dist'
            eta, v, man, dist = load_run_data(folder_path, m)
            all_data[m] = {'eta': eta, 'v': v, 'maneuver': man, 'disturbances': dist}
            max_frames = max(max_frames, len(eta))
            print(f"  Loaded {m}: {len(eta)} frames")
        except FileNotFoundError:
            print(f"  Skipping {m}: Files not found.")

    valid_methods = list(all_data.keys())
    if not valid_methods:
        print("Error: No valid data found to play.")
        return

    config = SystemConfig.from_yaml("config.yaml")
    viz = Visualizer(config.model, methods=valid_methods, folder_path=folder_path, paper_mode=paper_mode)
    
    sim_dt = 0.1 
    frame_delay = sim_dt / playback_speed

    if not paper_mode:
        plt.ion()
        plt.show()
    
    desc_text = "Processing Frames" if paper_mode else "Playing Timeline"

    for i in tqdm(range(max_frames), desc=desc_text):
        if not paper_mode and not plt.fignum_exists(viz.fig.number):
            break
            
        state_dict = {}
        for m in valid_methods:
            frame_idx = min(i, len(all_data[m]['eta']) - 1)
            
            eta = all_data[m]['eta'][frame_idx]
            v = all_data[m]['v'][frame_idx]
            maneuver = all_data[m]['maneuver'][frame_idx]
            dist = all_data[m]['disturbances'][frame_idx] # Grab disturbance

            if len(maneuver) == 0:
                maneuver = [TrajectoryPoint(eta, v)]
            
            # Update state_dict to include disturbance for the visualizer
            # Ensure Visualizer.update() is ready to receive (eta, v, maneuver, dist)
            state_dict[m] = (eta, v, maneuver, dist)

        viz.update(state_dict)
        
        if not paper_mode:
            plt.pause(frame_delay)

    if paper_mode:
        save_file = os.path.join(folder_path, "trajectory_plots.pdf")
        viz.render_paper_plot(save_path=save_file)
    else:
        plt.ioff()
        
    if plt.fignum_exists(viz.fig.number):
        plt.show(block=True)

# ... (main block remains the same)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Playback and compare docking algorithm runs.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Method flags
    parser.add_argument("-v", "--vanilla",    action="append_const", dest="methods", const="vanilla",    help="Include vanilla method")
    parser.add_argument("-n", "--nmpc",       action="append_const", dest="methods", const="nmpc",       help="Include tracking NMPC baseline method")
    parser.add_argument("-b", "--bilevel",    action="append_const", dest="methods", const="bilevel",    help="Include bilevel method")
    parser.add_argument("-m", "--monolevel", action="append_const", dest="methods", const="monolevel", help="Include monolevel method")

    # Configs
    parser.add_argument("--target_folder", metavar="GSD", required=True, help="Path to the run data folder (e.g. data/run_20260412_190813)")
    parser.add_argument("--speed", type=float, default=2.0, metavar="SPEED", help="Playback speed multiplier (default: 2.0)")
    parser.add_argument("--paper", action="store_true", dest="paper_mode", default=False)

    args = parser.parse_args()

    methods = args.methods or []
    if not methods:
        parser.error("At least one method flag is required: -v, -b, -m or -n")

    play_comparison(args.target_folder, methods, playback_speed=args.speed, paper_mode=args.paper_mode)