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
from dockinglib.errtrack import ErrorTracker

def load_run_data(folder_path, method_name):
    """Loads and parses the CSV data back into NumPy arrays and TrajectoryPoints."""
    eta_file = os.path.join(folder_path, f"{method_name}_eta.csv")
    v_file = os.path.join(folder_path, f"{method_name}_v.csv")
    m_file = os.path.join(folder_path, f"{method_name}_maneuver.csv")
    u_file = os.path.join(folder_path, f"{method_name}_u.csv")

    if not os.path.exists(eta_file) or not os.path.exists(m_file):
        raise FileNotFoundError(f"Data for '{method_name}' not found.")

    eta_data, v_data, maneuver_data, u_data = [], [], [], []

    with open(eta_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            eta_data.append(Position(np.array([float(row[0]), float(row[1]), float(row[2])])))

    with open(v_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            v_data.append(Velocity(np.array([float(row[0]), float(row[1]), float(row[2])])))

    with open(m_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            raw_list = json.loads(row[0])
            step_maneuver = [TrajectoryPoint(Position(np.array(pt[0])), Velocity(np.array(pt[1]))) for pt in raw_list]
            maneuver_data.append(step_maneuver)

    if os.path.exists(u_file):
        with open(u_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                u_data.append([float(x) for x in row])

    return eta_data, v_data, maneuver_data, u_data


def play_comparison(folder_path, methods, playback_speed=1.0, plot_errors=False):
    """Plays back multiple datasets synchronously, optionally plotting errors afterward."""
    print(f"Loading data from {folder_path}...")

    all_data = {}
    max_frames = 0

    for m in methods:
        try:
            eta, v, man, u = load_run_data(folder_path, m)
            all_data[m] = {'eta': eta, 'v': v, 'maneuver': man, 'u': u}
            max_frames = max(max_frames, len(eta))
            print(f"  Loaded {m}: {len(eta)} frames")
        except FileNotFoundError:
            print(f"  Skipping {m}: Files not found.")

    valid_methods = list(all_data.keys())
    if not valid_methods:
        print("Error: No valid data found to play.")
        return

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
        plt.show(block=not plot_errors) 
        
    if plot_errors:
        missing_u = [m for m in valid_methods if not all_data[m]['u']]
        if missing_u:
            print(f"\n  Warning: No control input (u) data found for: {', '.join(missing_u)}")
            print("  These methods will be excluded from error plots.")

        runs_dict = {
            m: {
                "eta_hist":      [p.array for p in all_data[m]['eta']],
                "v_hist":        [p.array for p in all_data[m]['v']],
                "u_hist":        all_data[m]['u'],
                "maneuver_hist": all_data[m]['maneuver'],
            }
            for m in valid_methods
            if all_data[m]['u']  
        }

        if runs_dict:
            print("\nPlotting error analysis...")
            tracker = ErrorTracker(dt=sim_dt)
            tracker.plot_comparison(runs_dict)
        else:
            print("\nNo methods with control input data available for error plotting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Playback and compare docking algorithm runs.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Method flags
    parser.add_argument("-v", "--vanilla",      action="append_const", dest="methods", const="vanilla",      help="Include vanilla method")
    parser.add_argument("-m", "--multirate",    action="append_const", dest="methods", const="multirate",    help="Include multirate method")
    parser.add_argument("-s", "--single-shoot", action="append_const", dest="methods", const="single_shoot", help="Include single shoot method")

    # Configs
    parser.add_argument("--target_folder", metavar="GSD", required=True, help="Path to the run data folder (e.g. data/run_20260412_190813)")
    parser.add_argument("--speed",         type=float, default=2.0,    metavar="SPEED", help="Playback speed multiplier (default: 2.0)")
    parser.add_argument("-p", "--plot",    action="store_true",         help="Plot pose, velocity, and control error after playback")

    args = parser.parse_args()

    methods = args.methods or []
    if not methods:
        parser.error("At least one method flag is required: -v, -m, or -s")

    play_comparison(args.target_folder, methods, playback_speed=args.speed, plot_errors=args.plot)