"""
    errortrack.py
    --------------
    Plots pose, velocity, and control error from a set of SPaC runs; can compare multiple methods.
"""

import matplotlib.pyplot as plt
import numpy as np

class ErrorTracker:
    def __init__(self, dt=0.1):
        self.dt = dt

    def plot_comparison(self, runs_dict):
        """
        runs_dict structure:
        {
            "Method Name": {
                "eta_hist": [...],
                "v_hist": [...],
                "u_hist": [...],
                "maneuver_hist": [[TrajectoryPoint, ...], ...]
            }
        }
        """
        fig, (ax_eta, ax_v, ax_u) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        plt.subplots_adjust(hspace=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, len(runs_dict)))

        for (name, data), color in zip(runs_dict.items(), colors):
            eta = np.array(data["eta_hist"])
            v = np.array(data["v_hist"])
            u = np.array(data["u_hist"])
            
            target_eta = np.array([m[0].eta for m in data["maneuver_hist"]])
            target_v = np.array([m[0].v for m in data["maneuver_hist"]])

            time = np.arange(len(eta)) * self.dt

            # Position Error
            pos_error = np.linalg.norm(eta[:, 0:2] - target_eta[:, 0:2], axis=1)
            psi_error = (target_eta[:, 2] - eta[:, 2] + np.pi) % (2 * np.pi) - np.pi
            
            ax_eta.plot(time, pos_error, label=f"{name} Pos Error (m)", color=color, linewidth=2)
            ax_eta.plot(time, np.abs(psi_error), '--', color=color, alpha=0.6, label=f"{name} Psi Error (rad)")

            # Velocity Error
            v_error = np.abs(target_v[:, 0] - v[:, 0])
            r_error = np.abs(target_v[:, 2] - v[:, 2])
            ax_v.plot(time, v_error, label=f"{name} Surge Error (m/s)", color=color, linewidth=2)
            ax_v.plot(time, r_error, '--', color=color, alpha=0.6, label=f"{name} Yaw Error (rad/s)")

            # Control Inputs (Thrust for Port/Starboard)
            ax_u.plot(time, u[:, 0], label=f"{name} Port Thrust", color=color, linewidth=1.5)
            ax_u.plot(time, u[:, 1], ':', label=f"{name} Stbd Thrust", color=color, linewidth=1.5)

        # Pose Plot
        ax_eta.set_title("Pose Error (Tracking Accuracy)")
        ax_eta.set_ylabel("Error Magnitude")
        ax_eta.legend(loc='upper right', fontsize='small', ncol=2)
        ax_eta.grid(True, alpha=0.3)

        # Velocity Plot
        ax_v.set_title("Velocity Error (Surge Tracking)")
        ax_v.set_ylabel("Error (m/s)")
        ax_v.legend(loc='upper right', fontsize='small')
        ax_v.grid(True, alpha=0.3)

        # Control Plot
        ax_u.set_title("Control Inputs (Thrust Effort)")
        ax_u.set_ylabel("Normalized Thrust [-1, 1]")
        ax_u.set_xlabel("Time (s)")
        ax_u.legend(loc='upper right', fontsize='small', ncol=2)
        ax_u.grid(True, alpha=0.3)

        plt.show(block=True)