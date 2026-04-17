import yaml 
import numpy as np
from dataclasses import dataclass
from typing import List, Union
      
@dataclass
class ManeuverConfig:
    capture_radius: float
    
@dataclass
class ModelConfig:
    mass: float
    length: float
    width: float
    surge_period: float
    yaw_period: float
    cg_to_pontoon: float
    thrust_coeff: float
       
@dataclass
class MPCConfig:
    horizon: int
    dt: float
    max_bilevel_iterations: int
    max_monolevel_iterations: int


@dataclass
class WeightConfig:
    Q_eta_pos: float
    Q_eta_psi: float
    Q_v_surge: float
    Q_v_sway: float
    Q_v_yaw: float
    Q_u: float
    W_length: float
    W_p_diff_R: float
    W_p_diff_Theta: float
    W_collapse: float
     
@dataclass
class SimulationConfig:
  initial_maneuver_params: Union[List[float], np.ndarray]
  initial_position: Union[List[float], np.ndarray]
  target_dock: Union[List[float], np.ndarray]
  max_steps: 500
  bilevel_replanning_interval: 10
  dt: 0.1
    
@dataclass
class DisturbanceConfig:
    preset: str = 'semilinear_winds'
    active: bool = True
    magnitude: float = 0.4
    base_dir: float = np.pi/2
    sweep_angle: float = np.pi/8
    yaw_mag: float = 0.0
    
@dataclass
class WindConfig:
    freq: float
    yaw_mag: float

@dataclass
class SystemConfig:
    maneuver_generator: ManeuverConfig
    model: ModelConfig
    mpc: MPCConfig
    weights: WeightConfig
    simulation: SimulationConfig
    disturbances: DisturbanceConfig
    
    @classmethod
    def from_yaml(cls, path:str):
        with open(path, 'r') as f:
                data = yaml.safe_load(f)
   
        return cls(
            maneuver_generator=ManeuverConfig(**data['maneuver_generator']),
            mpc=MPCConfig(**data['mpc']),
            model=ModelConfig(**data['model']),
            weights=WeightConfig(**data['weights']),
            simulation=SimulationConfig(**data['simulation']),
            disturbances=DisturbanceConfig(**data['disturbances'])
        )