"""
    logging.py
    ----------
    Holds TxtLogger and CSVLogger.
"""

import numpy as np
import os
import csv
import json

class TxtLogger:
    """Builds and writes the plain-text run summary report."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._lines: list[str] = []

    def write_header(
        self,
        timestamp: str,
        initial_eta,
        target_dock,
        mpc,
        wind_config: dict | None,
    ) -> None:
        """Emit the static preamble (config, weights, environment)."""
        if wind_config and wind_config.get("active", False):
            w_status  = "ACTIVE"
            w_details = (
                f"Mag: {wind_config.get('magnitude', 0)} m/s^2 | "
                f"Dir: {wind_config.get('base_dir', 0):.2f} rad | "
                f"Sweep: {wind_config.get('sweep_angle', 0):.2f} | "
                f"Yaw: {wind_config.get('yaw_mag', 0)}"
            )
        else:
            w_status, w_details = "INACTIVE", "N/A"

        self._lines += [
            "==================================================",
            "              RUN SUMMARY REPORT             ",
            "==================================================",
            f"Timestamp:    {timestamp}",
            f"Initial Pose: {initial_eta}",
            f"Target Dock:  {target_dock}",
            "--------------------------------------------------",
            "                CONTROLLER WEIGHTS                ",
            "--------------------------------------------------",
            f"Q_eta (Pose):     {np.diag(mpc.Q_eta)}",
            f"Q_v   (Velocity): {np.diag(mpc.Q_v)}",
            f"Q_u   (Thrust):   {np.diag(mpc.Q_u)}",
            f"W_length:         {mpc.W_length}",
            f"W_p_diff:         {mpc.W_p_diff}",
            f"W_collapse:       {mpc.W_collapse}",
            "--------------------------------------------------",
            "                ENVIRONMENT (WIND)                ",
            "--------------------------------------------------",
            f"Status:           {w_status}",
            f"Details:          {w_details}",
            "==================================================",
            f"{'Method':<15} | {'Converged':<9} | {'Steps':<6} | {'Time (s)':<8} | {'Frequency'}",
            "--------------------------------------------------",
        ]

    def log_result(
        self,
        method: str,
        converged: bool,
        steps: int,
        exec_time: float,
        freq_str: str,
    ) -> None:
        """Append one result row to the table."""
        self._lines.append(
            f"{method:<15} | {str(converged):<9} | {steps:<6} | {exec_time:<8.2f} | {freq_str}"
        )

    def save(self) -> str:
        """Flush all buffered lines to disk; returns the file path."""
        path = os.path.join(self.output_dir, "run_summary.txt")
        with open(path, "w") as f:
            f.write("\n".join(self._lines))
        return path
    
class CSVLogger:
    """Serialises per-method simulation histories to CSV files."""

    # Column headers for each file type
    _HEADERS = {
        "eta":      ["x", "y", "psi"],
        "v":        ["u_surge", "v_sway", "r_yaw"],
        "u":        ["thrust_port", "thrust_stbd"],
        "maneuver": ["raw_trajectory_array"],
        "disturbances": ["acc_u", "acc_r", "acc_v"]
    }

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    # ------------------------------------------------------------------ 
    #  Private helpers                                                     
    # ------------------------------------------------------------------ 

    def _path(self, method_name: str, suffix: str) -> str:
        return os.path.join(self.output_dir, f"{method_name}_{suffix}.csv")

    def _write_array_csv(self, method_name: str, kind: str, rows: list) -> None:
        """Generic writer for flat numeric histories (eta / v / u)."""
        with open(self._path(method_name, kind), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._HEADERS[kind])
            writer.writerows(rows)

    def _write_maneuver_csv(self, method_name: str, maneuver_hist: list) -> None:
        """Serialise the projected trajectory path at each timestep."""
        with open(self._path(method_name, "maneuver"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._HEADERS["maneuver"])
            for maneuver in maneuver_hist:
                if maneuver:
                    raw_path = [
                        [list(pt.eta), list(pt.v)]
                        for pt in maneuver
                    ]
                    writer.writerow([json.dumps(raw_path)])
                else:
                    writer.writerow(["[]"])
                    
                    
    # ------------------------------------------------------------------ 
    #  Public API                                                          
    # ------------------------------------------------------------------ 

    def save(
        self,
        method_name: str,
        eta_hist:     list,
        v_hist:       list,
        u_hist:       list,
        maneuver_hist: list,
        d_hist: list
    ) -> None:
        """Write all four CSV files for a single method in one call."""
        self._write_array_csv(method_name, "eta",      eta_hist)
        self._write_array_csv(method_name, "v",        v_hist)
        self._write_array_csv(method_name, "u",        u_hist)
        self._write_maneuver_csv(method_name, maneuver_hist)
        self._write_array_csv(method_name, "disturbances", d_hist)