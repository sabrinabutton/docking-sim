import os
import copy
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

from dockinglib.simulator import Simulator
from dockinglib.trajopt import TrajectoryOptimizer
from dockinglib.config import SystemConfig
from dockinglib.model import OtterModel
from dockinglib.errtrack import ErrorTracker 
from dockinglib.dgen import DisturbanceGenerator
from dockinglib.logging import TxtLogger, CSVLogger

def run_simulation(method, sim: Simulator, mpc, global_config):
    """Core simulation loop. sim owns dt, time, and wind internally."""

    u = np.array([0.0, 0.0], dtype=float)
    target_dock = np.array(global_config.simulation.target_dock)
    current_p = np.array(global_config.simulation.initial_maneuver_params)
    
    # Added d_hist to store disturbances
    eta_hist, v_hist, u_hist, maneuver_hist, d_hist = [], [], [], [], []
    
    replan_interval = global_config.simulation.bilevel_replanning_interval
    converged = False
    converged_step = global_config.simulation.max_steps

    for step in tqdm(range(global_config.simulation.max_steps), desc=f"Running {method}"):
        distance = np.linalg.norm(sim.eta[0:2] - target_dock[0:2])
        if distance < 0.5:
            print(f"\n[{method}] Converged at dock in {step} steps!")
            converged = True
            converged_step = step
            break

        # Disturbance estimate (This is what we want to save)
        d_current = sim.get_current_disturbance()

        # Solve logic
        if method == "vanilla":
            maneuver = mpc.mgen.generate_maneuver(sim.eta, target_dock, params=current_p)
            u = mpc.vanilla_solve(sim.eta, sim.v, u, maneuver, d_hat=d_current)
            
        elif method == "nmpc":
            # NMPC tracking a static path (baseline)
            maneuver = mpc.mgen.generate_maneuver(sim.eta, target_dock, params=current_p)
            u = mpc.nmpc_solve(sim.eta, sim.v, u, maneuver, d_hat=d_current)

        elif method == "bilevel":
            if step % replan_interval == 0:
                u, current_p = mpc.bilevel_optimization(sim.eta, sim.v, u, target_dock, current_p)
            maneuver = mpc.mgen.generate_maneuver(sim.eta, target_dock, params=current_p)
            if step % replan_interval != 0:
                u = mpc.vanilla_solve(sim.eta, sim.v, u, maneuver, d_hat=d_current)

        elif method == "monolevel":
            if step % replan_interval == 0:
                u, current_p = mpc.monolevel_optimization(sim.eta, sim.v, u, target_dock, current_p)
            maneuver = mpc.mgen.generate_maneuver(sim.eta, target_dock, params=current_p)
            if step % replan_interval != 0:
                u = mpc.vanilla_solve(sim.eta, sim.v, u, maneuver, d_hat=d_current)

        sim.step(u)

        # Record state and the ACTUAL disturbance encountered
        eta_hist.append(copy.deepcopy(sim.eta))
        v_hist.append(copy.deepcopy(sim.v))
        u_hist.append(copy.deepcopy(u))
        maneuver_hist.append(copy.deepcopy(maneuver))
        d_hist.append(copy.deepcopy(d_current)) 

    return eta_hist, v_hist, u_hist, maneuver_hist, d_hist, converged, converged_step


def run_experiments(methods_to_run=["vanilla", "nmpc", "bilevel", "monolevel"], show_plots=True):
    """Main execution function to run batch comparisons."""
    
    global_config = SystemConfig.from_yaml("config.yaml")
    model = OtterModel(global_config.model)
    mpc = TrajectoryOptimizer(global_config, model)
    
    initial_eta = global_config.simulation.initial_position
    target_dock = global_config.simulation.target_dock
    
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[START] Batch Run: {output_dir}")
    print(f"[METHODS] {methods_to_run}")

    txt_log = TxtLogger(output_dir)
    csv_log = CSVLogger(output_dir)

    txt_log.write_header(timestamp, initial_eta, target_dock, mpc, wind_config=None)

    all_results = {}

    for method in methods_to_run:
        # Added "nmpc" to the valid methods list
        if method not in ["vanilla", "nmpc", "bilevel", "monolevel"]:
            print(f"[WARNING] Unknown method '{method}'. Skipping.")
            continue

        start_time = time.time()
        sim = Simulator(global_config, model)
        
        eta_h, v_h, u_h, m_h, d_h, converged, steps = run_simulation(
            method, sim, mpc, global_config
        )
        
        exec_time = time.time() - start_time

        # Added the frequency string for the new NMPC baseline
        freq_str = {
            "vanilla":      "10Hz (Static Linear)",
            "nmpc":         "10Hz (Static NMPC)",
            "bilevel":      "1Hz Plan / 10Hz Track",
            "monolevel":   "1Hz Single-Shoot",
        }[method]

        txt_log.log_result(method, converged, steps, exec_time, freq_str)
        csv_log.save(method, eta_h, v_h, u_h, m_h, d_h)        

        all_results[method] = {"eta_hist": eta_h, "v_hist": v_h,
                               "u_hist": u_h,   "maneuver_hist": m_h,
                               "d_hist": d_h}
        print(f"[LOGS] Completed {method} (Time: {exec_time:.2f}s)\n")

    txt_log.save() 
    print(f"[DONE] Logged to output directory: {output_dir}/")

    if show_plots and all_results:
        tracker = ErrorTracker(dt=0.1)
        tracker.plot_comparison(all_results)
        import matplotlib.pyplot as plt
        plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate an optimization for one or more algorithms.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Method flags
    parser.add_argument("-v", "--vanilla",    action="append_const", dest="methods", const="vanilla",    help="Include vanilla method")
    parser.add_argument("-n", "--nmpc",       action="append_const", dest="methods", const="nmpc",       help="Include tracking NMPC baseline method")
    parser.add_argument("-b", "--bilevel",    action="append_const", dest="methods", const="bilevel",    help="Include bilevel method")
    parser.add_argument("-m", "--monolevel", action="append_const", dest="methods", const="monolevel", help="Include monolevel method")

    # Configs
    parser.add_argument("--plot", action="store_true", dest="plot", default=False, help="Toggle showing error plots on completion.")

    args = parser.parse_args()

    methods = args.methods or []
    if not methods:
        # Updated error message to include -n
        parser.error("At least one method flag is required: -v, -n, -b, or -m")
    
    run_experiments(methods, show_plots=args.plot)