from typing import Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .config import DroneEnvConfig

"""
Observation space: Demands + Current Position + Battery + Carried weight
Action space: Sites to visit + Visit base + fetch weight from base
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
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        # Observation space: [demands, battery, carried_weight, current_position]
        obs_dim = n_points + 5  # demands + mission_time + battery + carried_weight + position(x,y)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.n_points = n_points
        self.base_coordinates = base_coordinates
        self.coordinates = [base_coordinates] + coordinates
        self.demands = demands
        self.start_battery = start_battery
        self.max_battery = max_battery
        self.max_carry_weight = max_carry_weight
        self.battery_consumption = battery_consumption
        self.drone_weight = drone_weight
        self.max_velocity = max_velocity
        self.max_mission_time = max_mission_time
        self.mission_time = 0

        self.reset()


    @classmethod
    def from_config(cls, config: DroneEnvConfig):
        return cls(
            n_points=config.n_points,
            coordinates=config.coordinates,
            base_coordinates=config.base_coordinates,
            demands=config.demands,
            start_battery=config.start_battery,
            max_battery=config.max_battery,
            max_carry_weight=config.max_carry_weight,
            battery_consumption=config.battery_consumption,
            drone_weight=config.drone_weight,
            max_velocity=config.max_velocity,
            max_mission_time=config.max_mission_time
        )

    def step(self, action):
        site = int(action[0] * (self.n_points + 1)) if action[0] < 1 else self.n_points
        velocity = action[1] * self.max_velocity
        recharge_amount = action[2] * self.max_battery
        fetch_amount = action[3] * self.max_carry_weight
        self.route_history.append(self.coordinates[site - 1])
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if site == 0:
            # TODO: Recharge should take time
            self.current_position = self.base_coordinates
            distance = np.linalg.norm(np.array(self.current_position) - np.array(self.coordinates[site - 1]))
            velocity = min(velocity, self.max_velocity)
            time_spent = distance / (velocity + 0.0001)
            self.current_battery = min(self.current_battery + recharge_amount, self.max_battery)
            self.carried_weight = min(self.carried_weight + fetch_amount, self.max_carry_weight)
            reward -= 1
        else:
            distance = np.linalg.norm(np.array(self.current_position) - np.array(self.coordinates[site - 1]))
            velocity = min(velocity, self.max_velocity)
            time_spent = distance / (velocity + 0.0001)
            battery_cost = self.battery_consumption.alpha * velocity + self.battery_consumption.alpha * time_spent + self.battery_consumption.gamma * (self.carried_weight + self.drone_weight)
            if self.current_battery < battery_cost:
                reward = -100
                terminated = True
            else:
                self.current_battery -= battery_cost
                self.total_distance += distance
                self.current_position = self.coordinates[site - 1]

                demand_met = min(self.current_demands[site - 1], self.carried_weight)
                self.current_demands[site - 1] -= demand_met
                self.carried_weight -= demand_met
                reward += demand_met

        self.mission_time += time_spent
        reward -= 2

        if self.mission_time > self.max_mission_time:
            truncated = True
            reward -= 100
        
        if np.all(self.current_demands <= 0):
            terminated = True
            reward += 1000

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            self.current_demands,
            [self.mission_time],
            [self.current_battery],
            [self.carried_weight],
            self.current_position
        ]).astype(np.float32)

    def reset(self, seed=42):
        self.current_position = self.base_coordinates
        self.current_battery = self.start_battery
        self.carried_weight = 0.0
        self.total_distance = 0
        self.mission_time = 0
        self.current_demands = self.demands.copy()
        self.route_history = [self.base_coordinates]
        super().reset(seed=42)
        
        return self._get_obs(), {}

    def render(self, mode='human'):
        print("-"*100)
        print(f"Current position: {self.current_position}")
        print(f"Current battery: {self.current_battery}")
        print(f"Carried weight: {self.carried_weight}")
        print(f"Current demands: {self.current_demands}")
        print(f"Total distance: {self.total_distance}")
        print(f"Mission time: {self.mission_time}")
        print(f"Is charging: {self.current_position == self.base_coordinates}")
        print(f"Visited points: {set(self.route_history)}")
        print(f"Route history: {self.route_history}")
        print("-"*100)

    def print_results(self):
        print("-"*100)
        print("Results:")
        print(f"Total distance: {self.total_distance}")
        print(f"Mission time: {self.mission_time}")
        print(f"Route history: {self.route_history}")
        print("-"*100)