import os
import copy
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm

from dockinglib.simulator import Simulator
from dockinglib.mpc import SPaCSolver
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
    eta_hist, v_hist, u_hist, maneuver_hist = [], [], [], []
    
    replan_interval = global_config.simulation.multirate_replanning_interval
    
    converged = False
    converged_step = global_config.simulation.max_steps

    for step in tqdm(range(global_config.simulation.max_steps), desc=f"Running {method}"):
        # Convergence check
        distance = np.linalg.norm(sim.eta[0:2] - target_dock[0:2])
        if distance < 2.0 and np.linalg.norm(sim.v) < 0.2:
            print(f"\n[{method}] Converged at dock in {step} steps!")
            converged = True
            converged_step = step
            break

        # Disturbance estimate
        d_current = sim.get_current_disturbance()

        # Solve
        if method == "vanilla":
            maneuver = mpc.mgen.generate_maneuver(sim.eta, target_dock, params=current_p)
            u = mpc.vanilla_solve(sim.eta, sim.v, u, maneuver, d_hat=d_current)

        elif method == "multi-rate":
            if step % replan_interval == 0:
                u, current_p = mpc.spac_solve(sim.eta, sim.v, u, target_dock, current_p)
            maneuver = mpc.mgen.generate_maneuver(sim.eta, target_dock, params=current_p)
            if step % replan_interval != 0:
                u = mpc.vanilla_solve(sim.eta, sim.v, u, maneuver, d_hat=d_current)

        elif method == "single-shoot":
            if step % replan_interval == 0:
                u, current_p = mpc.nonlinear_spac_solver(sim.eta, sim.v, u, target_dock, current_p)
            maneuver = mpc.mgen.generate_maneuver(sim.eta, target_dock, params=current_p)
            if step % replan_interval != 0:
                u = mpc.vanilla_solve(sim.eta, sim.v, u, maneuver, d_hat=d_current)

        sim.step(u)

        eta_hist.append(copy.deepcopy(sim.eta))
        v_hist.append(copy.deepcopy(sim.v))
        u_hist.append(copy.deepcopy(u))
        maneuver_hist.append(copy.deepcopy(maneuver))

    return eta_hist, v_hist, u_hist, maneuver_hist, converged, converged_step

def run_experiments(methods_to_run=["vanilla", "multi-rate", "single-shoot"], show_plots=True):
    """Main execution function to run batch comparisons."""
    
    global_config = SystemConfig.from_yaml("config.yaml")
    model = OtterModel(global_config.model)
    mpc = SPaCSolver(global_config, model)
    
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
        if method not in ["vanilla", "multi-rate", "single-shoot"]:
            print(f"[WARNING] Unknown method '{method}'. Skipping.")
            continue

        start_time = time.time()
        sim = Simulator(global_config, model)
        
        eta_h, v_h, u_h, m_h, converged, steps = run_simulation(
            method, sim, mpc, global_config
        )
        
        exec_time = time.time() - start_time

        freq_str = {
            "vanilla":      "10Hz (Static)",
            "multi-rate":   "1Hz Plan / 10Hz Track",
            "single-shoot": "1Hz Single-Shoot",
        }[method]

        txt_log.log_result(method, converged, steps, exec_time, freq_str)
        csv_log.save(method, eta_h, v_h, u_h, m_h)        

        all_results[method] = {"eta_hist": eta_h, "v_hist": v_h,
                               "u_hist": u_h,   "maneuver_hist": m_h}
        print(f"[LOGS] Completed {method} (Time: {exec_time:.2f}s)\n")

    txt_log.save() 
    print(f"[DONE] Logged to output directory: {output_dir}/")

    if show_plots and all_results:
        tracker = ErrorTracker(dt=0.1)
        tracker.plot_comparison(all_results)
        import matplotlib.pyplot as plt
        plt.show(block=True)

if __name__ == "__main__":
    run_experiments(["vanilla"], show_plots=True)