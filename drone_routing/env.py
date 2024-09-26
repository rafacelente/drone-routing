from typing import Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .config import DroneEnvConfig

"""
Observation space: Demands + Current Position + Battery + Carried weight
Action space: Sites to visit + Visit base + fetch weight from base
"""

"""
Battery consumption:

\Delta battery = \alpha * velocity + \beta * time + \gamma * weight 
"""

class DroneRoutingEnv(gym.Env):
    def __init__(
            self,
            n_points: int,
            coordinates:  list[Tuple[int, int]],
            base_coordinates: Tuple[int, int],
            demands: np.ndarray,
            start_battery: int,
            max_battery: int,
            max_carry_weight: int,
            max_velocity: float,
            max_mission_time: float,
            battery_consumption: dict[str, float],
            drone_weight: float
        ):
        # Action space: [site_to_visit, recharge_amount, fetch_amount]
        self.action_space = spaces.Dict({
            'site': spaces.Discrete(n_points + 1),  # n_points + base
            'velocity': spaces.Box(low=0, high=max_velocity, shape=(1,)),
            'recharge': spaces.Box(low=0, high=max_battery, shape=(1,)),
            'fetch': spaces.Box(low=0, high=max_carry_weight, shape=(1,))
        })
        # Observation space: [demands, battery, carried_weight, current_position]
        self.observation_space = spaces.Dict({
            'demands': spaces.Box(low=0, high=np.inf, shape=(n_points,)),
            'mission_time': spaces.Box(low=0, high=max_mission_time, shape=(1,)),
            'current_position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'battery': spaces.Box(low=0, high=max_battery, shape=(1,)),
            'carried_weight': spaces.Box(low=0, high=max_carry_weight, shape=(1,)),
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        })

        self.n_points = n_points
        self.coordinates = coordinates
        self.base_coordinates = base_coordinates
        self.demands = demands
        self.start_battery = start_battery
        self.max_battery = max_battery
        self.max_carry_weight = max_carry_weight
        self.battery_consumption = battery_consumption
        self.drone_weight = drone_weight
        self.max_velocity = max_velocity
        self.max_mission_time = max_mission_time

        self.reset()


    @classmethod
    def from_config(cls, config: DroneEnvConfig):
        return cls(
            n_points=config.n_points,
            n_observation=config.n_observation,
            coordinates=config.coordinates,
            base_coordinates=config.base_coordinates,
            demands=config.demands,
            start_position=config.start_position,
            start_battery=config.start_battery,
            max_battery=config.max_battery,
            max_carry_weight=config.max_carry_weight,
            battery_consumption=config.battery_consumption,
            drone_weight=config.drone_weight,
        )

    def step(self, action):
        site = action['site']
        velocity = action['velocity'][0]
        recharge_amount = action['recharge'][0]
        fetch_amount = action['fetch'][0]

        reward = 0
        done = False
        info = {}

        if site == 0:
            # TODO: Recharge should take time
            self.current_position = self.base_coordinates
            self.current_battery = min(self.current_battery + recharge_amount, self.max_battery)
            self.carried_weight = min(self.carried_weight + fetch_amount, self.max_carry_weight)
            reward -= 1
        else:
            distance = np.linalg.norm(np.array(self.current_position) - np.array(self.coordinates[site - 1]))
            velocity = min(velocity, self.max_velocity)
            time_spent = distance / velocity
            battery_cost = self.battery_consumption['alpha'] * velocity + self.battery_consumption['beta'] * time_spent + self.battery_consumption['gamma'] * (self.carried_weight + self.drone_weight)
            if self.current_battery < battery_cost:
                reward = -100
                done = True
            else:
                self.current_battery -= battery_cost
                self.total_distance += distance
                self.current_position = self.coordinates[site - 1]

                demand_met = min(self.current_demands[site - 1], self.carried_weight)
                self.current_demands[site - 1] -= demand_met
                self.carried_weight -= demand_met
                reward += demand_met
        
        if np.all(self.current_demands <= 0):
            done = True
            reward += 1000

        observation = {
            'demands': self.current_demands,
            'battery': np.array([self.current_battery]),
            'carried_weight': np.array([self.carried_weight]),
            'position': np.array(self.current_position)
        }

        return observation, reward, done, False, info

    def reset(self):
        self.current_position = self.base_coordinates
        self.current_battery = self.start_battery
        self.carried_weight = 0.0
        self.total_distance = 0
        self.visited_points = np.zeros(self.n_points + 1)
        self.visited_points[0] = 1
        self.current_demands = self.demands.copy()
        self.route_history = [self.base_coordinates]
        
        observation = {
            'demands': self.current_demands,
            'battery': np.array([self.current_battery]),
            'carried_weight': np.array([self.carried_weight]),
            'position': np.array(self.current_position)
        }
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __del__(self):
        pass

    def __str__(self):
        return "DroneRoutingEnv"