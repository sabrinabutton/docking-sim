import yaml 
from dataclasses import dataclass
from typing import List
from .data import OtterState, Pose
      
@dataclass
class ManeuverConfig:
    interpolation_count: int
    max_curvature: float
    min_arc_length: float
    
@dataclass
class MPCConfig:
    lookahead_steps: int
    dt: float
    max_solver_iterations: int

@dataclass
class WeightConfig:
    maneuver_length: float
    curvature_violation: float
    cross_track: float
    heading: float
    angular_velocity: float
    control_effort: float
     
@dataclass
class SimulationConfig:
    total_timesteps: int
    integration_step: float
    enable_noise: bool
    start_state: OtterState
    dock_pose: Pose
    disturbance_amplitude_x: float
    disturbance_amplitude_y: float
    waypoint_tolerance: float
    
@dataclass
class VisualizerConfig:
    timestep_skip: int
    boat_length: float
    boat_width: float

@dataclass
class SystemConfig:
    maneuver_generator: ManeuverConfig
    mpc: MPCConfig
    weights: WeightConfig
    simulation: SimulationConfig
    viz: VisualizerConfig

    @classmethod
    def from_yaml(cls, path:str):
        with open(path, 'r') as f:
                data = yaml.safe_load(f)
                
        # Sim config needs special unpacking because it uses other dataclasses
        sim_data = data['simulation']
    
        # Convert start state mapping from YAML into OtterState dataclass
        if isinstance(sim_data['start_state'], dict):
            sim_data['start_state'] = OtterState(**sim_data['start_state'])
            
        if isinstance(sim_data['dock_pose'], dict):
            sim_data['dock_pose'] = Pose(**sim_data['dock_pose'])
        
        # Auto unpack everything else and return
        return cls(
            maneuver_generator=ManeuverConfig(**data['maneuver_generator']),
            mpc=MPCConfig(**data['mpc']),
            weights=WeightConfig(**data['weights']),
            simulation=SimulationConfig(**sim_data),
            viz=VisualizerConfig(**data['visualizer'])
        )