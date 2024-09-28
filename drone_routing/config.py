from typing import  Tuple
from typing_extensions import Self
from pydantic import BaseModel, field_validator, model_validator, Field
import yaml
import numpy as np
from pprint import pprint

class BatteryConsumption(BaseModel):
    """
    Battery consumption:

    \Delta battery = \alpha * velocity + \beta * time + \gamma * weight 
    """
    alpha: float = 0.1
    beta: float = 0.1
    gamma: float = 0.1

class DroneEnvConfig(BaseModel):
    n_points: int = 8
    coordinates: list[Tuple[int, int]] = [
        (1,2),
        (3,1),
        (5,4),
        (6,1),
        (2,5),
        (4,3),
        (7,2),
        (8,4),
    ]
    base_coordinates: Tuple[int, int] = (0,0)
    demands: np.ndarray = Field(default_factory=lambda: np.array([10, 15, 20, 25, 10, 15, 5, 10, 15], dtype=np.float32))
    start_battery: float = 100.0
    max_battery: float = 100.0
    max_carry_weight: float = 100.0
    max_velocity: float = 1.0
    max_mission_time: float = 100.0
    battery_consumption: BatteryConsumption = BatteryConsumption()
    drone_weight: float = 1.0

    class Config:
        arbitrary_types_allowed = True


    @staticmethod
    def from_yaml(file: str) -> Self:
        with open(file, "r") as f:
            config = yaml.safe_load(f)
            pprint(config)
            return DroneEnvConfig(**config)

    @field_validator("coordinates", mode="before")
    @classmethod
    def parse_coordinates_from_list_to_tuple(cls, v: list) -> list:
        return [tuple(coord) for coord in v]
    
    @field_validator("base_coordinates", mode="before")
    @classmethod
    def parse_base_coordinates_from_list_to_tuple(cls, v: list) -> tuple:
        return tuple(v)

    @field_validator("start_battery", mode="before")
    @classmethod
    def check_start_battery(cls, v) -> float:
        if v > cls.max_battery:
            raise ValueError("Start battery must be less than max battery")
        return v

    @field_validator("demands", mode="before")
    @classmethod
    def transform_array_to_np(cls, v) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        if not isinstance(v, list):
            raise ValueError("Demands must be a list")
        return np.array(v, dtype=np.float32)

    @model_validator(mode="after")
    def check_coordinates(self) -> Self:
        if len(self.coordinates) != self.n_points:
            raise ValueError("Number of coordinates must be equal to n_points")
        return self
    
    @model_validator(mode="after")
    def check_demands(self) -> Self:
        if len(self.demands) != self.n_points:
            raise ValueError("Number of demands must be equal to n_points")
        return self
    
    @model_validator(mode="after")
    def check_start_battery(self) -> Self:
        if self.start_battery > self.max_battery:
            raise ValueError("Start battery must be less than max battery")
        return self