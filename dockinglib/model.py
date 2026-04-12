"""
    model.py
    --------
    Holds the OtterModel class, reflecting the equations of motions in the associated paper.
"""

import numpy as np
import types
from .data import Velocity

# Translation from CG to body frame
H_r_g = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0.2, 1]
])

# Body to world
def R(psi:float):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1.0]
    ])

# Factory function instead of class, since everything is static once loaded
def OtterModel(config):
    m = config.mass # boat mass
    L = config.length # total boat length
    
    T_surge = config.surge_period # surge period
    T_yaw = config.yaw_period # yaw period
    
    l = config.cg_to_pontoon # CG to pontoon
    k = config.thrust_coeff # thrust coefficient
    
    R_33 = 0.25 * L
    
    # Parameters derived from Strand 
    X_udot = -0.1 * m
    Y_vdot = -1.5 * m
    N_rdot = -1.7 * R_33**2
    
    # Rigid body m, CG
    M_CG_RB = np.array([
        [m, 0, 0],
        [0, m, 0.2 * m],
        [0, 0.2 * m, 0.04 * m + R_33**2]
    ])

    # Added m 
    M_A = -np.array([
        [X_udot, 0, 0],
        [0, Y_vdot, 0],
        [0, 0, N_rdot]
    ])
    
    M = M_CG_RB + M_A
    M_inv = np.linalg.inv(M) 

    # Drag 
    D = np.array([
        [(m - X_udot)/T_surge, 0, 0],
        [0, -Y_vdot, 0],
        [0, 0, (0.04*m + R_33**2 - N_rdot)/T_yaw]
    ])
    
    # Actuator configuration matrix
    T = np.array([
        [1, 1],
        [-l, l]
    ])

    # Thrust coefficient matrix
    K = np.array([
        [k, 0],
        [0, k]
    ])

    # Map tau x and tau psi to 3 DoF
    B_act = np.array([
        [1, 0],
        [0, 0],
        [0, 1]
    ])
    
    # Coriolis, rigid body, CG
    def C_CG_RB(_v:Velocity):
        u, v, r = _v
        return np.array([
            [0, -m*r, -0.2*m*r],
            [m*r, 0, 0.2*m*u],
            [0.2*m*r,-0.2*m*u,0]
        ])
    
    # Coriolis, added
    def C_A(_v:Velocity):
        u, v, r = _v
        return np.array([
            [0,0, Y_vdot*v],
            [0,0, -X_udot*u],
            [-Y_vdot *v, X_udot * u, 0]
        ]) 
    
    # Jacobian 
    def J_C(_v:Velocity):
        u, v, r = _v
        return np.array([
            [0.0, -(m - Y_vdot)*r, -(m - Y_vdot)*v - 0.4*m*r],
            [(m - X_udot + 0.2*m)*r, 0.0, (m - X_udot + 0.2*m)*u],
            [0.2*m*r - (0.2*m + Y_vdot - X_udot)*v, 
             -(0.2*m + Y_vdot - X_udot)*u, 
             0.2*m*u]
        ])
        
    def kinematics(psi:float, v:Velocity):
        return R(psi) @ v
    
    def dynamics(v:Velocity, u):
        tau = B_act @ T @ K @ u
        C_tot = C_CG_RB(v) + C_A(v)
        return M_inv @ (tau - (C_tot @ v) - (D @ v))
        
    def get_linearized_dynamics(v:Velocity):
        A_cont = -M_inv @ (J_C(v) + D)
        B_cont = M_inv @ B_act @ T @ K
        return A_cont, B_cont

    # Returns the generated static functions for use as model.func
    return types.SimpleNamespace(
        dynamics=dynamics,
        get_linearized_dynamics=get_linearized_dynamics,
        kinematics=kinematics,
        C_CG_RB=C_CG_RB,
        C_A=C_A,
        J_C=J_C,
        M=M,
        M_inv=M_inv,
        D=D,
        B_act=B_act,
        T=T,
        K=K,
        H_r_g=H_r_g
    )
    