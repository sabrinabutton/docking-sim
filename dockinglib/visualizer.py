"""
    vizualizer.py
    --------------
    Creates an animation of an ASV following the planned trajectory while tracing its actual trajectory.
    Includes a wind conditions inset parsed from run_summary.txt in the data folder.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .data import Position
from .model import R
from .config import ModelConfig

# Use LaTeX rendering globally
plt.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman"],
    "axes.labelsize":     11,
    "axes.titlesize":     13,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
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

    def __init__(self, modelconf: ModelConfig, methods=['vanilla'], folder_path=None):
        self.modelconf = modelconf
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-10, 40)
        self.ax.set_ylim(-10, 40)
        self.ax.set_aspect('equal')

        self.methods = methods

        self.colors = {
            'vanilla':      {'boat': 'r',  'actual': 'darkred',   'planned': 'lightcoral'},
            'multi-rate':   {'boat': 'b',  'actual': 'darkblue',  'planned': 'cornflowerblue'},
            'single-shoot': {'boat': 'g',  'actual': 'darkgreen', 'planned': 'lightgreen'},
            'multirate':    {'boat': 'b',  'actual': 'darkblue',  'planned': 'cornflowerblue'},
            'single_shoot': {'boat': 'g',  'actual': 'darkgreen', 'planned': 'lightgreen'},
        }

        self.plots = {}
        self.trajectory_history = {m: [] for m in methods}

        legend_handles = []
        legend_labels  = []

        for m in methods:
            c = self.colors.get(m, {'boat': 'k', 'actual': 'k', 'planned': 'gray'})

            boat_poly,    = self.ax.plot([], [], color=c['boat'],    linestyle='-',  linewidth=2)
            actual_traj,  = self.ax.plot([], [], color=c['actual'],  linestyle='-',  linewidth=1.5, alpha=0.8)
            planned_traj, = self.ax.plot([], [], color=c['planned'], linestyle='--', linewidth=1.5, alpha=0.6)

            self.plots[m] = {
                'boat':    boat_poly,
                'actual':  actual_traj,
                'planned': planned_traj,
            }

            legend_handles.extend([boat_poly, planned_traj, actual_traj])
            legend_labels.extend([
                rf'\textit{{{m}}} (Boat)',
                rf'\textit{{{m}}} (Planned)',
                rf'\textit{{{m}}} (Actual)',
            ])

        self.target_marker, = self.ax.plot([], [], 'go', markersize=8)
        legend_handles.append(self.target_marker)
        legend_labels.append(r'Target Dock')

        self.ax.set_xlabel(r'$x$ (m)')
        self.ax.set_ylabel(r'$y$ (m)')
        self.ax.set_title(r'\textbf{SPaC Architecture Comparison}')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(legend_handles, legend_labels, loc='upper right', fontsize='small')

        # Wind inset
        self._wind_inset = None
        if folder_path is not None:
            wind = parse_wind_from_summary(folder_path)
            if wind is not None:
                self._draw_wind_inset(wind)

    # ------------------------------------------------------------------
    # Wind Inset
    # ------------------------------------------------------------------

    def _draw_wind_inset(self, wind: dict):
        """Draws a minimal wind inset matching the main plot aesthetic."""
        ax_w = self.ax.inset_axes([0.01, 0.74, 0.17, 0.24])
        ax_w.set_xlim(-1.5, 1.5)
        ax_w.set_ylim(-1.5, 1.5)
        ax_w.set_aspect('equal')
        ax_w.axis('off')

        # Thin bounding box matching the main axes style
        for spine_pos in ['top', 'bottom', 'left', 'right']:
            ax_w.spines[spine_pos].set_visible(False)

        rect = mpatches.FancyBboxPatch(
            (-1.45, -1.45), 2.9, 2.9,
            boxstyle="square,pad=0",
            facecolor='white', edgecolor='#cccccc', linewidth=0.8,
            zorder=0
        )
        ax_w.add_patch(rect)

        # Thin compass circle
        theta = np.linspace(0, 2 * np.pi, 120)
        ax_w.plot(np.cos(theta) * 0.9, np.sin(theta) * 0.9,
                  color='#cccccc', lw=0.7, zorder=1)

        # Cardinal labels in LaTeX serif
        for label, angle in [('N', np.pi/2), ('E', 0), ('S', -np.pi/2), ('W', np.pi)]:
            ax_w.text(
                1.28 * np.cos(angle), 1.28 * np.sin(angle),
                label,
                ha='center', va='center', fontsize=5.5,
                color='#888888', zorder=2
            )

        if wind['active']:
            dx = np.cos(wind['dir'])
            dy = np.sin(wind['dir'])
            scale = np.clip(wind['mag'] / 2.0, 0.35, 0.88)

            # Sweep arc
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

            # Main arrow — plain black, matching plot line style
            ax_w.annotate(
                '', xy=(dx * scale, dy * scale),
                xytext=(-dx * 0.12, -dy * 0.12),
                arrowprops=dict(
                    arrowstyle='->', color='#333333', lw=1.4,
                    mutation_scale=10
                ),
                zorder=3
            )

            # Stats text below
            info = (
                r'$|\mathbf{w}| = ' + f'{wind["mag"]:.2f}$' + r' m/s$^2$' + '\n'
                # r'$\psi_w = ' + f'{np.degrees(wind["dir"]):.1f}' + r'^\circ$' + '\n'
                # r'$\sigma = ' + f'{np.degrees(wind["sweep"]):.1f}' + r'^\circ$'
            )
            ax_w.text(
                0, -1.42, info,
                ha='center', va='bottom', fontsize=5.5,
                color='#444444', linespacing=1.4,
                zorder=3
            )

            title_color = '#333333'
            title_text  = r'\textit{Wind}'

        else:
            # Subtle cross for inactive
            for x0, y0, x1, y1 in [(-0.3, -0.3, 0.3, 0.3), (-0.3, 0.3, 0.3, -0.3)]:
                ax_w.plot([x0, x1], [y0, y1], color='#bbbbbb', lw=1.2, zorder=2)

            title_color = '#aaaaaa'
            title_text  = r'\textit{No Wind}'

        # ax_w.text(
        #     0, 1.42, title_text,
        #     ha='center', va='top', fontsize=6,
        #     color=title_color, zorder=3
        # )

        self._wind_inset = ax_w

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

            eta, v, maneuver = state

            self.trajectory_history[m].append((eta[0], eta[1]))

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
                self.target_marker.set_data([maneuver[-1].eta[0]], [maneuver[-1].eta[1]])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()