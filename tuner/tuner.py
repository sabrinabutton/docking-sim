"""
    Optimizes Q (control) and W (maneuver) weights using differential evolution
    on the bilevel framework. 
    
    Outputs optimal weights to 'weights.yaml' 
"""

import numpy as np
from scipy.optimize import differential_evolution
import yaml

from dockinglib.trajopt import TrajectoryOptimizer
from dockinglib.config import SystemConfig
from dockinglib.model import OtterModel

def objective_function(params, global_config, model):
    """
    params[0:6] = [q_pos, q_psi, q_surge, q_sway, q_yaw, q_u]
    params[6:10] = [W_length, W_diff_R, W_diff_Theta, W_collapse]
    """
   
    Q_eta = np.diag([params[0], params[0], params[1]])
    Q_v = np.diag([params[2], params[3], params[4]])
    Q_u = np.diag([params[5], params[5]])

    mpc = TrajectoryOptimizer(global_config, model, Q_eta=Q_eta, Q_v=Q_v, Q_u=Q_u)
    mpc.W_length = params[6]
    mpc.W_p_diff = np.array([params[7], params[8], params[7], params[8]])
    mpc.W_collapse = params[9]

    # Initial conditions
    dt = 0.1
    eta = np.array([30.0, 20.0, np.pi], dtype=float)
    v = np.array([0.0, 0.0, 0.0], dtype=float)
    u = np.array([0.0, 0.0], dtype=float)
    target_dock = np.array([10.0, 20.0, 0.0])
    
    # Initial S-Curve guess
    current_p = np.array([-10.0, np.pi/2, 10.0, np.pi/2])
    replan_interval = 10
    
    total_error = 0.0
    converged = False
    
    for step in range(500):
        if step % replan_interval == 0:
            u, current_p = mpc.bilevel_optimization(
                eta, v, u, target_dock, current_p, 
                dt=dt, cruise_speed=1.0, reverse_speed=0.5
            )
        else:
            maneuver = mpc.mgen.generate_maneuver(
                current_eta=eta, dock_point=target_dock, 
                params=current_p, dt=dt, cruise_speed=1.0, reverse_speed=0.5
            )
            u = mpc.vanilla_solve(eta, v, u, maneuver)
        
        # Model step
        k1 = model.dynamics(v, u)
        k2 = model.dynamics(v + 0.5 * dt * k1, u)
        k3 = model.dynamics(v + 0.5 * dt * k2, u)
        k4 = model.dynamics(v + dt * k3, u)
        v = v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        eta = eta + (model.kinematics(eta[2], v) * dt)
        eta[2] = (eta[2] + np.pi) % (2 * np.pi) - np.pi
        
        # Cost calculation
        dist = np.linalg.norm(eta[0:2] - target_dock[0:2])
        psi_err = np.abs((target_dock[2] - eta[2] + np.pi) % (2 * np.pi) - np.pi)
        
        total_error += (dist**2 + psi_err**2 + np.sum(u**2) * 0.1)

        if dist < 0.5 and np.linalg.norm(v) < 0.1:
            converged = True
            total_error -= (500 - step) * 2.0 
            break
            
    if not converged:
        total_error += 5000.0 + (dist * 100.0)

    return total_error


def params_to_dict(params: np.ndarray) -> dict:
    return {
        "weights": {
            "Q_eta_pos":       float(params[0]),
            "Q_eta_psi":       float(params[1]),
            "Q_v_surge":       float(params[2]),
            "Q_v_sway":        float(params[3]),
            "Q_v_yaw":         float(params[4]),
            "Q_u":             float(params[5]),
            "W_length":        float(params[6]),
            "W_p_diff_R":      float(params[7]),
            "W_p_diff_Theta":  float(params[8]),
            "W_collapse":      float(params[9]),
        }
    }


def save_weights(params: np.ndarray, path: str = "weights.yaml") -> None:
    with open(path, "w") as f:
        yaml.dump(params_to_dict(params), f, default_flow_style=False, sort_keys=False)


class ObjectiveWrapper:
    def __init__(self, global_config, model):
        self.best_cost = float('inf')
        self.eval_count = 0
        self.global_config = global_config
        self.model = model

    def __call__(self, params):
        self.eval_count += 1
        cost = objective_function(params, self.global_config, self.model)

        if cost < self.best_cost:
            self.best_cost = cost
            print(f"\n[Eval {self.eval_count}] >>> New Best: {self.best_cost:.2f}")
            save_weights(params)
            
        elif self.eval_count % 20 == 0:
            print(f"[Eval {self.eval_count}] Searching... (Current Best: {self.best_cost:.2f})")
            
        return cost


def run_overnight_tuner():
    global_config = SystemConfig.from_yaml("config.yaml")
    model = OtterModel(global_config.model)

    bounds = [
        # --- MPC Weights ---
        (1.0, 500.0),   # Q_pos
        (10.0, 1000.0), # Q_psi
        (1.0, 200.0),   # Q_surge
        (50.0, 500.0),  # Q_sway
        (1.0, 100.0),   # Q_yaw
        (0.1, 50.0),    # Q_u
        # --- Planner Weights ---
        (1.0, 100.0),   # W_length
        (1.0, 100.0),   # W_diff_R
        (10.0, 500.0),  # W_diff_Theta
        (100.0, 2000.0) # W_collapse 
    ]
    
    wrapper = ObjectiveWrapper(global_config, model)
    
    result = differential_evolution(
        wrapper, 
        bounds, 
        strategy='best1bin',
        popsize=8,
        mutation=(0.5, 1),
        recombination=0.7,
        tol=0.01,
        disp=False 
    )

if __name__ == "__main__":
    run_overnight_tuner()