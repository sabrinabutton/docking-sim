"""
    vizualizer.py
    --------------
    Creates an animation of an ASV following the planned trajectory while tracing its actual trajectory.
    Includes a wind conditions inset parsed from run_summary.txt in the data folder.
    
    Now supports `paper_mode` for generating clean, static multi-path plots.
    Automatically formats into a 2x2 grid if exactly 4 methods are provided.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from .data import Position
from .model import R
from .config import ModelConfig

plt.rcParams.update({
    "text.usetex": False,         
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def parse_wind_from_summary(folder_path):
    """
    Reads run_summary.txt and extracts wind parameters.
    Returns a dict with keys: active, mag, dir, sweep, yaw
    or None if the file/section is missing.
    """
    summary_path = os.path.join(folder_path, "run_summary.txt")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, 'r') as f:
        text = f.read()

    wind_match = re.search(
        r'ENVIRONMENT \(WIND\).*?Status:\s*(\w+).*?Details:\s*Mag:\s*([\d.]+).*?Dir:\s*([\d.]+).*?Sweep:\s*([\d.]+).*?Yaw:\s*([\d.]+)',
        text, re.DOTALL | re.IGNORECASE
    )

    if not wind_match:
        return None

    return {
        'active': wind_match.group(1).upper() == 'ACTIVE',
        'mag':    float(wind_match.group(2)),
        'dir':    float(wind_match.group(3)),
        'sweep':  float(wind_match.group(4)),
        'yaw':    float(wind_match.group(5)),
    }


class Visualizer:

    def __init__(self, modelconf: ModelConfig, methods=['vanilla'], folder_path=None, paper_mode=False):
        self.modelconf = modelconf
        self.paper_mode = paper_mode
        self.methods = methods
        n_methods = len(self.methods)

        # ------------------------------------------------------------------
        # Dynamic Subplot Layout (2x2 for 4 methods, Stacked otherwise)
        # ------------------------------------------------------------------
        if n_methods == 4:
            self.fig, self.axes = plt.subplots(
                nrows=2, ncols=2, 
                figsize=(12, 10), 
                sharex=True, sharey=True
            )
            self.axes_flat = self.axes.flatten()
        else:
            self.fig, self.axes = plt.subplots(
                nrows=n_methods, ncols=1, 
                figsize=(13, 3.5 * n_methods), 
                sharex=True, sharey=True
            )
            self.axes_flat = np.atleast_1d(self.axes)

        self.ax_map = {m: ax for m, ax in zip(self.methods, self.axes_flat)}

        # Format grids and labels based on layout
        for i, ax in enumerate(self.axes_flat):
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # TODO: Rename vanilla Linearized MPC globally
            if self.methods[i] is "vanilla":
                ax.set_title("LINEARIZED MPC")
            else:
                ax.set_title(self.methods[i].upper()) # Add title to identify each subplot
            
            if n_methods == 4:
                # Bottom row gets X labels, Left column gets Y labels
                if i in [2, 3]: ax.set_xlabel('x (m)')
                if i in [0, 2]: ax.set_ylabel('y (m)')
            else:
                ax.set_ylabel('y (m)')
                if i == n_methods - 1:
                    ax.set_xlabel('x (m)')

        # Added 'nmpc' colors
        self.colors = {
            'vanilla':      {'boat': 'r',          'actual': 'gray', 'planned': 'lightcoral'},
            'nmpc':         {'boat': 'darkorange', 'actual': 'gray', 'planned': 'moccasin'},
            'bilevel':      {'boat': 'b',          'actual': 'gray', 'planned': 'cornflowerblue'},
            'monolevel':   {'boat': 'g',          'actual': 'gray', 'planned': 'lightgreen'}
        }

        self.plots = {}
        self.trajectory_history = {m: [] for m in methods}
        self.maneuver_history = {m: [] for m in methods}
        self.disturbance_plots = {m: [] for m in methods}
        self.target_markers = {}

        # Parse wind data once
        self._wind_dict = None
        if folder_path is not None:
            self._wind_dict = parse_wind_from_summary(folder_path)

        # ------------------------------------------------------------------
        # Standard Animation Setup (Only if NOT in paper mode)
        # ------------------------------------------------------------------
        if not self.paper_mode:
            for m in methods:
                ax = self.ax_map[m]
                c = self.colors.get(m, {'boat': 'k', 'actual': 'gray', 'planned': 'gray'})

                boat_poly,    = ax.plot([], [], color=c['boat'],    linestyle='-',  linewidth=2)
                actual_traj,  = ax.plot([], [], color='gray',       linestyle=':',  linewidth=1.2, alpha=0.8)
                planned_traj, = ax.plot([], [], color=c['planned'], linestyle='--', linewidth=1.5, alpha=0.6)
                target_marker, = ax.plot([], [], 'go', markersize=8)

                self.plots[m] = {
                    'boat':    boat_poly,
                    'actual':  actual_traj,
                    'planned': planned_traj,
                }
                self.target_markers[m] = target_marker

                legend_handles = [boat_poly, planned_traj, actual_traj, target_marker]
                legend_labels  = [f'{m} (Boat)', f'{m} (Planned)', f'{m} (Actual)', 'Target Dock']
                ax.legend(legend_handles, legend_labels, loc='upper right', fontsize='small')

            self._wind_inset = None
            if self._wind_dict is not None:
                self._draw_wind_inset(self._wind_dict)

    # ------------------------------------------------------------------
    # Wind Inset
    # ------------------------------------------------------------------

    def _draw_wind_inset(self, wind: dict):
        """Draws a minimal wind inset matching the main plot aesthetic."""
        # Attach inset to the very first subplot (Top Left in a 2x2)
        ax_w = self.axes_flat[0].inset_axes([0.01, 0.65, 0.17, 0.33])
        ax_w.set_xlim(-1.5, 1.5)
        ax_w.set_ylim(-1.5, 1.5)
        ax_w.set_aspect('equal')
        ax_w.axis('off')

        for spine_pos in ['top', 'bottom', 'left', 'right']:
            ax_w.spines[spine_pos].set_visible(False)

        rect = mpatches.FancyBboxPatch(
            (-1.45, -1.45), 2.9, 2.9,
            boxstyle="square,pad=0",
            facecolor='white', edgecolor='#cccccc', linewidth=0.8,
            zorder=0
        )
        ax_w.add_patch(rect)

        theta = np.linspace(0, 2 * np.pi, 120)
        ax_w.plot(np.cos(theta) * 0.9, np.sin(theta) * 0.9, color='#cccccc', lw=0.7, zorder=1)

        for label, angle in [('N', np.pi/2), ('E', 0), ('S', -np.pi/2), ('W', np.pi)]:
            ax_w.text(
                1.28 * np.cos(angle), 1.28 * np.sin(angle),
                label, ha='center', va='center', fontsize=5.5,
                color='#888888', zorder=2
            )

        if wind['active']:
            dx = np.cos(wind['dir'])
            dy = np.sin(wind['dir'])
            scale = np.clip(wind['mag'] / 2.0, 0.35, 0.88)

            if wind['sweep'] > 0.01:
                sweep_angles = np.linspace(
                    wind['dir'] - wind['sweep'],
                    wind['dir'] + wind['sweep'], 40
                )
                r_arc = scale * 0.72
                ax_w.plot(
                    r_arc * np.cos(sweep_angles),
                    r_arc * np.sin(sweep_angles),
                    color='#aaaaaa', lw=0.9, alpha=0.6, zorder=2
                )

            ax_w.annotate(
                '', xy=(dx * scale, dy * scale),
                xytext=(-dx * 0.12, -dy * 0.12),
                arrowprops=dict(
                    arrowstyle='->', color='#333333', lw=1.4,
                    mutation_scale=10
                ),
                zorder=3
            )

            info = f'|w| = {wind["mag"]:.2f} m/s²\n'
            ax_w.text(
                0, -1.42, info,
                ha='center', va='bottom', fontsize=6.5,
                color='#444444', linespacing=1.4, zorder=3
            )

        else:
            for x0, y0, x1, y1 in [(-0.3, -0.3, 0.3, 0.3), (-0.3, 0.3, 0.3, -0.3)]:
                ax_w.plot([x0, x1], [y0, y1], color='#bbbbbb', lw=1.2, zorder=2)

        self._wind_inset = ax_w

    # ------------------------------------------------------------------
    # Data Processing & Rendering
    # ------------------------------------------------------------------

    def _get_boat_shape(self, eta: Position):
        l = self.modelconf.length / 2
        w = self.modelconf.width / 2

        pts = np.array([
            [ l,  0],
            [-l,  w],
            [-l, -w],
            [ l,  0],
        ])

        pts_global = (R(eta[2])[0:2, 0:2] @ pts.T).T + np.array([eta[0], eta[1]])
        return pts_global[:, 0], pts_global[:, 1]

    def update(self, state_dict):
        """
        Expects state_dict: {'vanilla': (eta, v, maneuver), ...}
        """
        for m, state in state_dict.items():
            if state is None:
                continue

            eta, v, maneuver, dist = state

            # Accumulate data regardless of mode
            self.trajectory_history[m].append((eta[0], eta[1], eta[2]))
            if maneuver:
                self.maneuver_history[m].append(maneuver)
            if dist is not None:
                self.disturbance_plots[m].append(dist)


            # If in paper mode, do not animate. We will render at the very end.
            if self.paper_mode:
                continue

            # --- Animation Logic ---
            if len(self.trajectory_history[m]) > 1:
                traj_x = [p[0] for p in self.trajectory_history[m]]
                traj_y = [p[1] for p in self.trajectory_history[m]]
                self.plots[m]['actual'].set_data(traj_x, traj_y)

            path_x = [p.eta[0] for p in maneuver]
            path_y = [p.eta[1] for p in maneuver]
            self.plots[m]['planned'].set_data(path_x, path_y)

            boat_shape = self._get_boat_shape(eta)
            self.plots[m]['boat'].set_data(boat_shape[0], boat_shape[1])

            if len(maneuver) > 0:
                self.target_markers[m].set_data([maneuver[-1].eta[0]], [maneuver[-1].eta[1]])

            # Dynamically adjust bounds during animation using the flat array
            self.axes_flat[0].relim()
            self.axes_flat[0].autoscale_view()

        if not self.paper_mode:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


    def render_paper_plot(self, save_path=None):
        """
        To be called AFTER the simulation finishes when paper_mode=True.
        Draws the actual trajectory in gray and all planned maneuvers, fading from 
        pale (white-blended) to vivid (pure color base) over time.
        """
        if not self.paper_mode:
            print("Warning: render_paper_plot() is intended for paper_mode=True.")
            return

        all_x, all_y = [], []

        for m in self.methods:
            ax = self.ax_map[m]
            c = self.colors.get(m, {'actual': 'gray', 'planned': 'k'})
            base_color = mcolors.to_rgb(c.get('boat', 'k')) 
            
            maneuvers = self.maneuver_history[m]
            n_maneuvers = len(maneuvers)
            last_line = None

            legend_handles = []
            legend_labels = []
            
            disturbances = self.disturbance_plots[m]
            if len(disturbances) > 0:
                for i in range(0, len(disturbances), 50):
                    if i < len(maneuvers) and maneuvers[i]:
                        dist = disturbances[i]
                        eta = self.trajectory_history[m][i] # Position where it was felt
                        psi = eta[2]
                        
                        # A. Linear Surge/Sway Arrow (Gray)
                        # Transform body-frame dist[0, 1] to Global Frame
                        cos_p, sin_p = np.cos(psi), np.sin(psi)
                        gx = dist[0] * cos_p - dist[1] * sin_p
                        gy = dist[0] * sin_p + dist[1] * cos_p
                        
                        # Increased scale factor to 15.0 for longer, proportional arrows
                        s_linear = 5.0 
                        
                        lin_arrow = mpatches.FancyArrowPatch(
                            (eta[0], eta[1]), 
                            (eta[0] + gx * s_linear, eta[1] + gy * s_linear),
                            arrowstyle='->',
                            mutation_scale=15, # Consistent head size
                            color='gray',
                            linewidth=1.5, # Slightly heavier
                            alpha=0.4,
                            zorder=6
                        )
                        ax.add_patch(lin_arrow)

                        if abs(dist[2]) > 0.005:
                            direction = 1 if dist[2] > 0 else -1
                        
                            # 1. Unit vector along heading (Forward)
                            fwd_x, fwd_y = np.cos(psi), np.sin(psi)
                            # 2. Unit vector perpendicular to heading (Starboard/Right)
                            side_x, side_y = np.sin(psi), -np.cos(psi)
                            
                            # Define Start Point: Slightly behind and on the OPPOSITE side
                            # We flip the side based on direction to ensure it always crosses "through"
                            r_side = 1.2
                            r_fwd = 0.8
                            
                            x_start = eta[0] - (fwd_x * r_fwd) - (side_x * r_side * direction)
                            y_start = eta[1] - (fwd_y * r_fwd) - (side_y * r_side * direction)
                            
                            x_end = eta[0] + (fwd_x * r_fwd) + (side_x * r_side * direction)
                            y_end = eta[1] + (fwd_y * r_fwd) + (side_y * r_side * direction)

                            # Adjusting rad to 0.3 (smaller curve) and flipping sign if concavity was wrong
                            # Using negative rad usually forces the curve to "hug" the center point
                            yaw_arc = mpatches.FancyArrowPatch(
                                (x_start, y_start), (x_end, y_end),
                                connectionstyle=f"arc3,rad={-0.3 * direction}",
                                arrowstyle="->",
                                mutation_scale=15,
                                color="gray",
                                lw=1.5,
                                alpha=0.4,
                                zorder=9
                            )
                            ax.add_patch(yaw_arc)
                            
            # Plot all planned maneuvers over time
            for i, maneuver in enumerate(maneuvers):
                if not maneuver:
                    continue
                
                path_x = [p.eta[0] for p in maneuver]
                path_y = [p.eta[1] for p in maneuver]
                
                all_x.extend(path_x)
                all_y.extend(path_y)

                fade_factor = 0.85 * (1.0 - i / max(1, n_maneuvers - 1))
                color = tuple(bc + (1 - bc) * fade_factor for bc in base_color)

                is_last = (i == n_maneuvers - 1)
                lw = 1.5 if is_last else 0.8
                zorder = 3 if is_last else 2
                
                line, = ax.plot(path_x, path_y, color=color, linestyle='--', linewidth=lw, zorder=zorder)
                if is_last:
                    last_line = line

            if last_line:
                legend_handles.append(last_line)
                legend_labels.append(f'{m} (Planned)')

            # Plot the Actual Path cleanly (thin, dotted gray)
            actual_coords = self.trajectory_history[m]
            if len(actual_coords) > 1:
                traj_x = [p[0] for p in actual_coords]
                traj_y = [p[1] for p in actual_coords]
                
                all_x.extend(traj_x)
                all_y.extend(traj_y)
                
                actual_line, = ax.plot(traj_x, traj_y, color='black', linestyle=':', linewidth=3, zorder=4)
                
                legend_handles.append(actual_line)
                legend_labels.append(f'{m} (Actual)')

            # 3. Target Marker
            if n_maneuvers > 0 and maneuvers[-1]:
                target_x = maneuvers[-1][-1].eta[0]
                target_y = maneuvers[-1][-1].eta[1]
                target_marker, = ax.plot([target_x], [target_y], 'ko', markersize=8, zorder=5)
                
                legend_handles.append(target_marker)
                legend_labels.append('Target Dock')

            #ax.legend(legend_handles, legend_labels, loc='upper right', fontsize='small')

        # Fit bounds to data
        # Bounds fitting
        if all_x:
            mx, Mx, my, My = min(all_x), max(all_x), min(all_y), max(all_y)
            px, py = (Mx - mx) * 0.1 or 5.0, (My - my) * 0.1 or 5.0
            self.axes_flat[0].set_xlim(mx - px, Mx + px)
            self.axes_flat[0].set_ylim(my - py, My + py)

        # Draw Wind Inset
        if self._wind_dict is not None:
            self._draw_wind_inset(self._wind_dict)

        # Adjust layout and figure properties
        self.fig.tight_layout()
        # Add space at the bottom specifically for the legend
        self.fig.subplots_adjust(bottom=0.08)

        # Global legend at the bottom of the figure using proxy artists
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', lw=2, label='Planned Maneuver'),
            Line2D([0], [0], color='black', linestyle=':', lw=3, label='Actual Trajectory'),
            Line2D([0], [0], marker='o', color='w', label='Target Dock', markerfacecolor='black', markersize=8),
            mpatches.Patch(color='gray', alpha=0.5, label='Env. Disturbance (Linear/Yaw)')
        ]
        
        # Position at 'lower center' using normalized figure coordinates
        self.fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True, fontsize='medium')

        if save_path:
            self.fig.savefig(save_path, bbox_inches='tight', dpi=300)
            
        plt.show()