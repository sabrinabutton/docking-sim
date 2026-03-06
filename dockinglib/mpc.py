import numpy as np
from scipy.optimize import minimize
from .path import ManeuverGenerator
from .data import OtterState, Pose, PointType, Status

NUM_PATH_PARAMS = 4 # Constant and baked in, just made into a constant for readability

class MPCController: 
    
    def __init__(self, config):
        self.config = config.mpc
        self.mgen = ManeuverGenerator(config); 
        
        self.u_guess_flattened = np.ones(self.config.lookahead_steps * 2) * 10.0
        self.path_params_guess = np.zeros(4)
        
        self.m_u, self.m_v, self.I_z = 60.0, 110.0, 18.0
        self.d_u, self.d_v, self.d_r = 50.0, 180.0, 40.0
        self.B = 0.85
    
    def _otter_model(self, state: OtterState, u, Vcx=0, Vcy=0):
        tr, tl = u
        
        thrust_surge = tl + tr
        thrust_yaw = 0.5 * self.B * (tr - tl)
        
        du = (1.0 / self.m_u) * (thrust_surge - self.d_u * state.surge + self.m_v * state.sway * state.yaw_rate)
        dv = (1.0 / self.m_v) * (-self.d_v * state.sway - self.m_u * state.surge * state.yaw_rate)
        dr = (1.0 / self.I_z) * (thrust_yaw - self.d_r * state.yaw_rate)

        dx = state.surge * np.cos(state.psi) - state.sway * np.sin(state.psi)
        dy = state.surge * np.sin(state.psi) + state.sway * np.cos(state.psi)
        dpsi = state.yaw_rate

        return OtterState(dx, dy, dpsi, du, dv, dr)
    
    def _u_only_objective(self, u_flattened, state:OtterState, maneuver):
        u_traj = u_flattened.reshape((self.config.lookahead_steps, 2))
        temp_state = state.copy()
        total_cost = 0
        
        dt = self.config.dt
        
        for i in range(self.config.lookahead_steps):
            k1 = self._otter_model(temp_state, u_traj[i])
            temp_state_mid = temp_state + k1 * 0.5 * dt
            k2 = self._otter_model(temp_state_mid, u_traj[i])
            temp_state = temp_state + k2 * dt 

            # Heavy penalty on Cross-Track Error (Y deviation)
            total_cost += 500.0 * np.sqrt((temp_state.y - maneuver[i].y)**2 + (temp_state.x - maneuver[i].x)**2)

            # Penalty on Heading relative to the path
            psi_err = (temp_state.psi - maneuver[i].psi + np.pi) % (2 * np.pi) - np.pi
            total_cost += 50.0 * (psi_err**2)

            # Penalty on high angular velocity to prevent spinning
            total_cost += 10.0 * (temp_state.yaw_rate)**2

            # Control effort
            total_cost += 0.01 * np.sum(u_traj[i]**2)
            
        return total_cost
        
        
    # TODO: PROBLEM: It cannot actually solve this!
    def _objective(self, path_u_flattened, state:OtterState, maneuver, dock_point, status):
        path_params, u_flattened = path_u_flattened[:(NUM_PATH_PARAMS )], path_u_flattened[(NUM_PATH_PARAMS):]
        u_traj = u_flattened.reshape((self.config.lookahead_steps, 2))
        
        temp_state = state.copy()
        total_cost = 0
        
        dt = self.config.dt
        
        # Length, similarity/delta, end position, terminal point
        # penalize phases differently 
        # For delta: Deadband as you get far away or Median vs achievement bubble to penalize 
        
        new_maneuver = self.mgen.get_docking_maneuver(Pose(state.x, state.y, state.psi, PointType.SETUP), dock_point, path_params, status)
        
        for i in range(self.config.lookahead_steps):
            k1 = self._otter_model(temp_state, u_traj[i])
            temp_state_mid = temp_state + k1 * 0.5 * dt
            k2 = self._otter_model(temp_state_mid, u_traj[i])
            temp_state = temp_state + k2 * dt 
            
            # Heavy penalty on Cross-Track Error (Y deviation)
            total_cost += 500.0 * np.sqrt((temp_state.y - new_maneuver[0].y)**2 + (temp_state.x - new_maneuver[0].x)**2)
            # WE NEED TO TARGET DIFFERENT POINTS AS WE MOVE FORWARD

            # Penalty on Heading relative to the path
            psi_err = (temp_state.psi - new_maneuver[0].psi + np.pi) % (2 * np.pi) - np.pi
            total_cost += 50.0 * (psi_err**2)

            # Penalty on high angular velocity to prevent spinning
            total_cost += 10.0 * (temp_state.yaw_rate)**2

            # Control effort
            total_cost += 0.01 * np.sum(u_traj[i]**2)
            
        return total_cost
    
    def solve_with_fixed_maneuver(self, state: OtterState, maneuver):
        bounds = [(-100, 100)] * (self.config.lookahead_steps * 2)
        optimum = minimize(self._u_only_objective, self.u_guess_flattened, args=(state, maneuver), method='SLSQP', bounds=bounds)
        
        if optimum.success: # If we achieve convergence to a local minima
            self.u_guess_flattened = np.roll(optimum.x, -2) # Warm start
            return optimum.x.reshape((-1, 2))[0]
        else:
            return np.array([0.0, 0.0])
        
    def solve(self, state: OtterState, dock_point:Pose, maneuver, status):
        guess = np.append(self.path_params_guess, self.u_guess_flattened)

        path_bounds = [(1.0, 20.0), (1.0, 20.0), (0.1, 2.0), (0.1, 2.0)]  # l_a, l_b, c_a, c_b
        control_bounds = [(-100, 100)] * (self.config.lookahead_steps * 2)
        bounds = path_bounds + control_bounds
        optimum = minimize(self._objective, guess, args=(state, maneuver, dock_point, status), method='SLSQP', bounds=bounds)
        
        if optimum.success: # If we achieve convergence to a local minima
            # Seperate the path and the thruster inputs
            best_path_params, best_u_flattened = optimum.x[:(NUM_PATH_PARAMS)], optimum.x[(NUM_PATH_PARAMS):]
            best_u = best_u_flattened.reshape((-1, 2))[0]
            
            # Setup warm start
            self.u_guess_flattened = np.roll(best_u_flattened, -2) # next timestep in array
            self.path_params_guess = best_path_params
            
        else: # Default output, this should (hopefully) never happen
            best_path_params = [10, 10, 1, 1]
            best_u = [0, 0]
            
        best_maneuver =  self.mgen.get_docking_maneuver(Pose(state.x, state.y, state.psi, PointType.SETUP), dock_point, best_path_params, status)
            
        return best_u, best_maneuver


