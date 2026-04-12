"""
    mpc.py
    ------
    Holds MPC class
"""

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from .mgen import ManeuverGenerator
from .data import TrajectoryPoint, Position, Velocity
from .model import OtterModel, R
from .config import SystemConfig

class SPaCSolver:

    def __init__(self, config: SystemConfig, model: OtterModel):

        self.config = config.mpc
        self.model = model 

        weights = config.weights
        self.Q_eta = np.diag([weights.Q_eta_pos, weights.Q_eta_pos, weights.Q_eta_psi])
        self.Q_v = np.diag([weights.Q_v_surge, weights.Q_v_sway, weights.Q_v_yaw])
        self.Q_u = np.diag([weights.Q_u, weights.Q_u])
        
        self.W_length = weights.W_length
        self.W_p_diff = np.array([weights.W_p_diff_R, weights.W_p_diff_Theta, weights.W_p_diff_R, weights.W_p_diff_Theta])
        self.W_collapse = weights.W_collapse

        self.mgen = ManeuverGenerator(config.maneuver_generator)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _normalize_inputs(self, eta, v, u):
        """Cast and flatten eta, v, u to guaranteed 1D float arrays."""
        return (
            np.asarray(eta, dtype=float).flatten(),
            np.asarray(v, dtype=float).flatten(),
            np.asarray(u, dtype=float).flatten(),
        )

    def _pad_trajectory(self, trajectory, eta_val):
        """
        Ensure trajectory is exactly self.config.horizon steps long.
        If empty, fills with a zero-velocity hold at the current pose.
        If shorter than the horizon, pads by repeating the last item.
        """
        if len(trajectory) == 0:
            hold = TrajectoryPoint(Position(eta_val), Velocity([0.0, 0.0, 0.0]))
            return [hold] * self.config.horizon
        if len(trajectory) < self.config.horizon:
            return list(trajectory) + [trajectory[-1]] * (self.config.horizon - len(trajectory))
        return trajectory

    def _wrap_target_heading(self, t_eta, ref_heading):
        """
        Return a copy of t_eta with its heading component wrapped relative
        to ref_heading, so the angular error stays in (-pi, pi].
        """
        t = t_eta.copy()
        t[2] = ref_heading + ((t[2] - ref_heading + np.pi) % (2 * np.pi) - np.pi)
        return t

    def _build_linearized_system(self, eta_val, v_val, u_val):
        """
        Linearise the 6-DOF (eta, v) dynamics around the current operating
        point and discretise with a forward-Euler step of self.config.dt.

        Returns
        -------
        A_6d, B_6d : discrete-time state and input matrices (6x6, 6x2)
        x_dot_0    : continuous-time state derivative at the operating point
        """
        u_0, v_sw0, _ = v_val
        psi_0 = eta_val[2]

        v_dot_0 = self.model.dynamics(v_val, u_val)
        A_cont, B_cont = self.model.get_linearized_dynamics(v_val)

        # Jacobian of the kinematic map J(eta) w.r.t. eta, evaluated at psi_0
        J_K = np.zeros((3, 3))
        J_K[0, 2] = -u_0 * np.sin(psi_0) - v_sw0 * np.cos(psi_0)
        J_K[1, 2] =  u_0 * np.cos(psi_0) - v_sw0 * np.sin(psi_0)

        R_0 = R(psi_0)

        # Continuous 6x6 system
        A_6 = np.zeros((6, 6))
        A_6[0:3, 0:3] = J_K
        A_6[0:3, 3:6] = R_0
        A_6[3:6, 3:6] = A_cont

        B_6 = np.zeros((6, 2))
        B_6[3:6, :] = B_cont

        # Forward-Euler discretisation
        A_6d = np.eye(6) + A_6 * self.config.dt
        B_6d = B_6 * self.config.dt

        eta_dot_0 = R_0 @ v_val
        x_dot_0 = np.concatenate([eta_dot_0, v_dot_0])

        return A_6d, B_6d, x_dot_0

    def _disturbance_vector(self, d_hat):
        """Embed a 3-DOF body-frame disturbance into the 6-DOF state vector."""
        d_6 = np.zeros(6)
        d_6[3:6] = d_hat
        return d_6

    def _build_qp(self, eta_val, v_val, u_val, padded_traj,
                  A_6d, B_6d, x_dot_0, d_6):
        """
        Construct the CVXPY QP variables, cost, and constraints for one
        MPC horizon.

        Returns
        -------
        problem : cp.Problem  (not yet solved)
        du      : cp.Variable (2 x horizon) — incremental thrust decisions
        """
        dx = cp.Variable((6, self.config.horizon + 1))
        du = cp.Variable((2, self.config.horizon))

        cost = 0.0
        constraints = [dx[:, 0] == np.zeros(6)]

        for k in range(self.config.horizon):
            t_eta_k = self._wrap_target_heading(
                np.asarray(padded_traj[k].eta, dtype=float).flatten(),
                eta_val[2],
            )
            t_v_k = np.asarray(padded_traj[k].v, dtype=float).flatten()

            constraints += [
                dx[:, k + 1] == A_6d @ dx[:, k] + B_6d @ du[:, k] + (x_dot_0 + d_6) * self.config.dt
            ]

            eta_pred = eta_val + dx[0:3, k + 1]
            v_pred   = v_val   + dx[3:6, k + 1]

            eta_error = t_eta_k - eta_pred
            v_error   = t_v_k   - v_pred

            cost += cp.quad_form(eta_error, self.Q_eta)
            cost += cp.quad_form(v_error,   self.Q_v)

            # Halve the actuation penalty when the reference requests reverse
            Q_u_k = self.Q_u * (0.5 if t_v_k[0] < 0 else 1.0)
            cost += cp.quad_form(du[:, k], Q_u_k)

            # Hard state constraints
            constraints += [v_pred[0] >= -2.0, v_pred[0] <= 3.0]   # surge
            constraints += [v_pred[1] >= -1.0, v_pred[1] <= 1.0]   # sway
            constraints += [v_pred[2] >= -1.5, v_pred[2] <= 1.5]   # yaw rate

            abs_thrust = u_val + du[:, k]
            constraints += [abs_thrust >= -1.0, abs_thrust <= 1.0]

        return cp.Problem(cp.Minimize(cost), constraints), du

    def _spac_geometric_penalty(self, p_guess, p_prev):
        """
        Compute the three SPaC geometric penalties:
          1. Arc-length upper-bound  (prefer short paths)
          2. Continuity              (prefer small jumps from p_prev)
          3. Collapse lower-bound    (enforce minimum berth lengths)

        Returns a scalar penalty value.
        """
        R_a, Theta_a, R_b, Theta_b = p_guess

        MIN_BERTH_LEN = 15.0
        MIN_THETA     = 0.5
        MIN_APPROACH  = 10.0

        length_cost = self.W_length * (abs(R_a) * Theta_a + abs(R_b) * Theta_b)
        diff_cost   = np.sum(self.W_p_diff * (p_guess - p_prev) ** 2)

        collapse_cost = (
            self.W_collapse * max(0, MIN_BERTH_LEN - abs(R_b) * Theta_b) ** 2
            + self.W_collapse * max(0, MIN_THETA     - Theta_b) ** 2
            + self.W_collapse * max(0, MIN_APPROACH  - abs(R_a) * Theta_a) ** 2
        )

        return length_cost + diff_cost + collapse_cost

    def _generate_padded_trajectory(self, eta_val, dock_point, p_guess, dt,
                                    cruise_speed, reverse_speed):
        """Generate a maneuver and pad it to self.config.horizon steps."""
        trajectory = self.mgen.generate_maneuver(
            current_eta=eta_val,
            dock_point=dock_point,
            params=p_guess,
            dt=dt,
            cruise_speed=cruise_speed,
            reverse_speed=reverse_speed,
        )
        return self._pad_trajectory(trajectory, eta_val)

    # -------------------------------------------------------------------------
    # Solvers
    # -------------------------------------------------------------------------

    def vanilla_solve(self, eta, v, u, trajectory, d_hat=np.zeros(3)):
        eta_val, v_val, u_val = self._normalize_inputs(eta, v, u)

        padded_traj = self._pad_trajectory(trajectory, eta_val)
        A_6d, B_6d, x_dot_0 = self._build_linearized_system(eta_val, v_val, u_val)
        d_6 = self._disturbance_vector(d_hat)

        problem, du = self._build_qp(eta_val, v_val, u_val, padded_traj,
                                     A_6d, B_6d, x_dot_0, d_6)
        try:
            problem.solve(solver=cp.OSQP)
            if problem.status not in ("optimal", "optimal_inaccurate"):
                return np.array([0.0, 0.0])
            return u_val + du[:, 0].value
        except Exception:
            return np.array([0.0, 0.0])

    def spac_solve(self, eta, v, u_prev, dock_point, p_prev, dt=0.1,
                   cruise_speed=1.0, reverse_speed=0.5, d_hat=np.zeros(3)):

        eta_val, v_val, u_val = self._normalize_inputs(eta, v, u_prev)
        
        A_6d, B_6d, x_dot_0 = self._build_linearized_system(eta_val, v_val, u_val)
        d_6 = self._disturbance_vector(d_hat)

        def evaluate_p(p_guess):
            padded_traj = self._generate_padded_trajectory(
                eta_val, dock_point, p_guess, dt, cruise_speed, reverse_speed
            )

            problem, _ = self._build_qp(eta_val, v_val, u_val, padded_traj,
                                         A_6d, B_6d, x_dot_0, d_6)
            try:
                problem.solve(solver=cp.OSQP)
                if problem.status not in ("optimal", "optimal_inaccurate"):
                    return 1e6
                return problem.value + self._spac_geometric_penalty(p_guess, p_prev)
            except Exception:
                return 1e6

        p_bounds = [(-50.0, -2.0), (0.1, np.pi), (2.0, 50.0), (0.1, np.pi)]
        opt_result = minimize(evaluate_p, p_prev, method="SLSQP",
                              bounds=p_bounds, options={"maxiter": self.config.max_multirate_iterations})
        best_p = opt_result.x

        final_traj = self._generate_padded_trajectory(
            eta_val, dock_point, best_p, dt, cruise_speed, reverse_speed
        )
        u_opt = self.vanilla_solve(eta, v, u_prev, final_traj)
        return u_opt, best_p

    def nonlinear_spac_solver(self, eta, v, u_prev, dock_point, p_prev, dt=0.1,
                               cruise_speed=1.0, reverse_speed=0.5, d_hat=np.zeros(3)):
        import copy
        
        eta_start, v_start, u_prev_val = self._normalize_inputs(eta, v, u_prev)

        # Decision vector: Z = [u_flat (horizon x 2), R_a, Theta_a, R_b, Theta_b]
        Z0 = np.concatenate((np.zeros((self.config.horizon, 2)).flatten(), p_prev))

        thrust_bounds = [(-1.0, 1.0)] * (self.config.horizon * 2)
        path_bounds   = [(-50.0, -2.0), (0.1, np.pi), (2.0, 50.0), (0.1, np.pi)]
        bounds        = thrust_bounds + path_bounds

        def evaluate_nmpc(Z):
            U       = Z[:-4].reshape((self.config.horizon, 2))
            p_guess = Z[-4:]

            padded_traj = self._generate_padded_trajectory(
                eta_start, dock_point, p_guess, dt, cruise_speed, reverse_speed
            )

            cost    = 0.0
            curr_eta = copy.deepcopy(eta_start)
            curr_v   = copy.deepcopy(v_start)
            prev_u   = u_prev_val

            for k in range(self.config.horizon):
                u_k = U[k]

                # RK4 integration with disturbance
                def dyn(v_, u_): return self.model.dynamics(v_, u_) + d_hat

                k1 = dyn(curr_v, u_k)
                k2 = dyn(curr_v + 0.5 * dt * k1, u_k)
                k3 = dyn(curr_v + 0.5 * dt * k2, u_k)
                k4 = dyn(curr_v + dt * k3, u_k)
                
                curr_v   = curr_v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                curr_eta = curr_eta + self.model.kinematics(curr_eta[2], curr_v) * dt
                curr_eta[2] = (curr_eta[2] + np.pi) % (2 * np.pi) - np.pi

                t_eta = self._wrap_target_heading(
                    np.asarray(padded_traj[k].eta), curr_eta[2]
                )
                t_v  = np.asarray(padded_traj[k].v)
                du   = u_k - prev_u

                eta_error = t_eta - curr_eta
                v_error   = t_v   - curr_v

                cost += np.dot(eta_error, self.Q_eta @ eta_error)
                cost += np.dot(v_error,   self.Q_v   @ v_error)
                cost += np.dot(du,        self.Q_u   @ du)

                # Soft surge constraints
                if curr_v[0] > 3.0:  cost += 1000 * (curr_v[0] - 3.0)  ** 2
                if curr_v[0] < -2.0: cost += 1000 * (-2.0 - curr_v[0]) ** 2

                prev_u = u_k

            # Geometric penalties with nonlinear_spac_solver-specific weights
            R_a, Theta_a, R_b, Theta_b = p_guess
            cost += self.W_length * (abs(R_a) * Theta_a + abs(R_b) * Theta_b)
            cost += np.sum(self.W_p_diff * (p_guess - p_prev) ** 2)
            cost += self.W_collapse * max(0, 15.0 - abs(R_b) * Theta_b) ** 2
            cost += self.W_collapse * max(0, 0.5  - Theta_b)            ** 2

            return cost

        opt_result = minimize(evaluate_nmpc, Z0, method="SLSQP",
                              bounds=bounds, options={"maxiter": self.config.max_singleshot_iterations})
        best_Z = opt_result.x
        return best_Z[0:2], best_Z[-4:]