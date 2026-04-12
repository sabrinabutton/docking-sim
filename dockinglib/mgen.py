"""
    mgen.py
    -------
    Holds ManeuverGenerator class.
"""

import numpy as np

from .data import TrajectoryPoint, Position, Velocity
from .config import ManeuverConfig

# Parameter indices for the maneuver params vector
R_APPROACH_IDX = 0
THETA_APPROACH_IDX = 1
R_BERTH_IDX = 2
THETA_BERTH_IDX = 3

class ManeuverGenerator:
    """Generates docking maneuver trajectories composed of arc and straight-line segments."""

    def __init__(self, config:ManeuverConfig):
        self.capture_radius = config.capture_radius

    # ------------------------------------------------------------------
    # Spatial geometry helpers
    # ------------------------------------------------------------------

    def _get_arc(self, end_eta, r, theta, pts):
        """
        Build an arc of ``pts`` positions that ends at ``end_eta``.

        The arc lies on a circle of signed radius ``r`` (negative = port turn)
        and subtends angle ``theta`` (radians).  The vessel heading at each
        point is kept tangent to the circle.

        Args:
            end_eta: [x, y, psi] of the arc's final point.
            r:       Signed turning radius (m).  Sign encodes turn direction.
            theta:   Arc angle (rad).
            pts:     Number of discrete points to generate.

        Returns:
            List of Position objects, ordered start → end.
        """
        if pts <= 0:
            return []

        end_x, end_y, end_psi = end_eta[0], end_eta[1], end_eta[2]
        center_x = end_x - r * np.sin(end_psi)
        center_y = end_y + r * np.cos(end_psi)
        end_angle = np.arctan2(end_y - center_y, end_x - center_x)

        angles = np.linspace(end_angle - theta, end_angle, int(pts))
        r_mag = np.abs(r)
        direction = np.copysign(1.0, r)

        return [
            Position([
                center_x + r_mag * np.cos(a),
                center_y + r_mag * np.sin(a),
                a + direction * np.pi / 2,
            ])
            for a in angles
        ]

    def _get_straight_line(self, start_eta, end_eta, speed, dt):
        """
        Build a straight-line segment from ``start_eta`` to ``end_eta``.

        The number of points is derived from travel distance, speed, and
        timestep so that each step covers roughly ``speed * dt`` metres.
        The endpoint is excluded so segments can be concatenated cleanly.

        Args:
            start_eta: [x, y, psi] of the starting pose.
            end_eta:   [x, y, psi] of the ending pose.
            speed:     Desired travel speed (m/s).
            dt:        Timestep (s).

        Returns:
            List of Position objects, ordered start → (exclusive) end.
        """
        dist = np.linalg.norm(end_eta[:2] - start_eta[:2])
        pts = max(1, int(np.ceil(dist / (speed * dt)))) if speed > 0 else 1

        xs = np.linspace(start_eta[0], end_eta[0], pts, endpoint=False)
        ys = np.linspace(start_eta[1], end_eta[1], pts, endpoint=False)

        psi_start = start_eta[2]
        psi_end = psi_start + (end_eta[2] - psi_start + np.pi) % (2 * np.pi) - np.pi
        psis = np.linspace(psi_start, psi_end, pts, endpoint=False)

        return [Position([xs[i], ys[i], psis[i]]) for i in range(pts)]

    # ------------------------------------------------------------------
    # Speed-profile helpers
    # ------------------------------------------------------------------

    def _build_speed_profile(self, n_total, split_idx, approach_end_idx,
                              cruise_speed, reverse_speed, ramp_pts=8):
        """
        Return a float array of length ``n_total`` encoding the desired
        longitudinal speed (signed) at each trajectory point.

        The profile has three phases:

        * **Lead-in / approach** (indices 0 … approach_end_idx − 1):
          cruise at ``+cruise_speed``, ramping down near the transition.
        * **Berth / reverse** (indices approach_end_idx … n_total − 2):
          ramp up to ``−reverse_speed``, hold, then ramp back to zero.
        * **Final point** (index n_total − 1): always zero (vessel stopped).

        Args:
            n_total:         Total number of trajectory points.
            split_idx:       Index where arc points begin (after lead-in line).
            approach_end_idx: Global index of the approach-to-berth transition.
            cruise_speed:    Forward cruise speed (m/s, positive).
            reverse_speed:   Reverse berthing speed (m/s, positive magnitude).
            ramp_pts:        Number of points over which speed ramps occur.

        Returns:
            np.ndarray of shape (n_total,) with signed speeds.
        """
        speeds = np.full(n_total, cruise_speed, dtype=float)
        rev_kick = -0.7  # Initial reverse impulse to overcome low-speed dead-zone

        for i in range(n_total):
            if i < approach_end_idx:
                dist_to_transition = approach_end_idx - i
                if dist_to_transition < ramp_pts:
                    speeds[i] = max(0.1, cruise_speed * dist_to_transition / ramp_pts)
            else:
                dist_from_transition = i - approach_end_idx
                dist_to_end = n_total - i
                full_rev = -reverse_speed

                if dist_from_transition < ramp_pts:
                    t = dist_from_transition / ramp_pts
                    speeds[i] = rev_kick + (full_rev - rev_kick) * t
                elif dist_to_end < ramp_pts:
                    speeds[i] = full_rev * dist_to_end / ramp_pts
                else:
                    speeds[i] = full_rev

        return speeds

    # ------------------------------------------------------------------
    # Trajectory assembly
    # ------------------------------------------------------------------

    def _points_to_trajectory(self, points, speed_profile, dt):
        """
        Convert a list of Position points and a matching speed profile into
        a list of TrajectoryPoints (pose + velocity).

        Translational velocity ``u`` is taken directly from ``speed_profile``;
        rotational rate ``r`` is derived from the finite-difference of heading
        divided by ``dt``.  The final point is assigned zero velocity.

        Args:
            points:        List of Position objects.
            speed_profile: Array-like of floats, same length as ``points``.
            dt:            Timestep (s).

        Returns:
            List of TrajectoryPoint objects.
        """
        if not points:
            return []

        trajectory = []
        for i, eta in enumerate(points):
            if i < len(points) - 1:
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                dpsi = (points[i + 1][2] - points[i][2] + np.pi) % (2 * np.pi) - np.pi
                dist = np.hypot(dx, dy)
                u = speed_profile[i] if dist > 1e-6 else 0.0
                r = dpsi / dt if dist > 1e-6 else 0.0
                v = Velocity([u, 0.0, r])
            else:
                v = Velocity([0.0, 0.0, 0.0])

            trajectory.append(TrajectoryPoint(eta, v))

        return trajectory

    # ------------------------------------------------------------------
    # Core maneuver generation
    # ------------------------------------------------------------------

    def _arc_point_count(self, r, theta, speed, dt):
        """Return the minimum number of arc points for the given geometry and speed."""
        arc_len = abs(r * theta)
        return max(2, int(np.ceil(arc_len / (speed * dt))))

    def _generate_maneuver(self, current_eta, dock_point, params, dt,
                            cruise_speed, reverse_speed):
        """
        Build a complete docking trajectory from ``current_eta`` to ``dock_point``.

        The maneuver consists of:

        1. An **approach arc** that curves the vessel toward the berth entry
           at cruise speed.
        2. A **berth arc** that reverses the vessel into the dock.
        3. An optional **straight lead-in** prepended when the vessel is
           outside the capture radius of the approach arc.

        A continuous speed profile is applied across all phases with smooth
        ramps at phase transitions.

        Args:
            current_eta:   Current vessel pose [x, y, psi].
            dock_point:    Target docking pose [x, y, psi].
            params:        Four-element array [r_approach, θ_approach,
                           r_berth, θ_berth].
            dt:            Timestep (s).
            cruise_speed:  Forward speed during approach (m/s).
            reverse_speed: Speed magnitude during berthing (m/s).

        Returns:
            List of TrajectoryPoint objects.
        """
        pts_approach = self._arc_point_count(
            params[R_APPROACH_IDX], params[THETA_APPROACH_IDX], cruise_speed, dt
        )
        pts_berth = self._arc_point_count(
            params[R_BERTH_IDX], params[THETA_BERTH_IDX], reverse_speed, dt
        )

        berth = self._get_arc(
            dock_point,
            params[R_BERTH_IDX],
            -params[THETA_BERTH_IDX] * np.copysign(1.0, params[R_BERTH_IDX]),
            pts_berth,
        )
        berth_entry = berth[0] if berth else dock_point

        approach = self._get_arc(
            berth_entry,
            params[R_APPROACH_IDX],
            params[THETA_APPROACH_IDX] * np.copysign(1.0, params[R_APPROACH_IDX]),
            pts_approach,
        )

        all_arc_points = approach + berth
        arc_coords = np.array([[p[0], p[1]] for p in all_arc_points])
        distances = np.linalg.norm(arc_coords - current_eta[:2], axis=1)
        min_idx = int(np.argmin(distances))

        if distances[min_idx] > self.capture_radius:
            lead_in = self._get_straight_line(
                current_eta, all_arc_points[min_idx], cruise_speed, dt
            )
            total_points = lead_in + all_arc_points[min_idx:]
            split_idx = len(lead_in)
        else:
            total_points = all_arc_points[min_idx:]
            split_idx = 0

        approach_end_relative = max(0, pts_approach - min_idx)
        global_transition_idx = split_idx + approach_end_relative

        speeds = self._build_speed_profile(
            n_total=len(total_points),
            split_idx=split_idx,
            approach_end_idx=global_transition_idx,
            cruise_speed=cruise_speed,
            reverse_speed=reverse_speed,
        )

        return self._points_to_trajectory(total_points, speeds, dt)

    def generate_maneuver(
        self,
        current_eta,
        dock_point,
        params=None,
        dt=0.1,
        cruise_speed=1.2,
        reverse_speed=1.0,
    ):
        """
        Public entry point for maneuver generation.

        Args:
            current_eta:   Current vessel pose [x, y, psi].
            dock_point:    Target docking pose [x, y, psi].
            params:        Maneuver shape parameters [r_approach, θ_approach,
                           r_berth, θ_berth].  Defaults to
                           [10.0, π/4, 5.0, π/4].
            dt:            Timestep (s).  Default 0.1.
            cruise_speed:  Forward approach speed (m/s).  Default 1.2.
            reverse_speed: Reverse berthing speed (m/s).  Default 1.0.

        Returns:
            List of TrajectoryPoint objects describing the full maneuver.
        """
        if params is None:
            params = [10.0, np.pi / 4, 5.0, np.pi / 4]
        return self._generate_maneuver(
            current_eta, dock_point, params, dt, cruise_speed, reverse_speed
        )