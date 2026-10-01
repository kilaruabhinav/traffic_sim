"""Configuration for the traffic model and RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class TrafficConfig:
    """All tunable parameters for one traffic environment."""

    num_lanes: int = 4
    arrival_probabilities: Tuple[float, float, float, float] = (0.35, 0.35, 0.35, 0.35)
    queue_limits: Tuple[int, int, int, int] = (20, 20, 20, 20)
    max_queue_lengths: Tuple[int, int, int, int] = (40, 40, 40, 40)
    vehicles_per_step: int = 2
    min_green_time: int = 3
    max_green_time: int = 12
    switch_time: int = 1
    episode_length: int = 600
    waiting_time_normalizer: float = 30.0

    throughput_reward_weight: float = 1.0
    queue_penalty_weight: float = 0.5
    waiting_penalty_weight: float = 0.5
    overflow_penalty_weight: float = 1.0
    starvation_penalty_weight: float = 0.5
    progress_reward_weight: float = 0.3
    switch_penalty: float = -0.02
    starvation_threshold: float = 15.0

    def __post_init__(self) -> None:
        if self.num_lanes != 4:
            raise ValueError("TrafficEnv supports exactly four lanes")
        vectors = (self.arrival_probabilities, self.queue_limits, self.max_queue_lengths)
        if any(len(values) != self.num_lanes for values in vectors):
            raise ValueError("lane configuration values must contain exactly four entries")
        if any(not 0.0 <= probability <= 1.0 for probability in self.arrival_probabilities):
            raise ValueError("arrival probabilities must be between 0 and 1")
        if any(limit <= 0 for limit in self.queue_limits):
            raise ValueError("queue limits must be positive")
        if any(maximum <= limit for maximum, limit in zip(self.max_queue_lengths, self.queue_limits)):
            raise ValueError("each maximum queue length must be greater than its queue limit")
        if self.vehicles_per_step <= 0 or self.episode_length <= 0:
            raise ValueError("service capacity and episode length must be positive")
        if not 0 <= self.min_green_time <= self.max_green_time:
            raise ValueError("green times must satisfy 0 <= minimum <= maximum")
        if self.switch_time < 0:
            raise ValueError("switch time cannot be negative")
        if self.waiting_time_normalizer <= 0 or self.starvation_threshold <= 0:
            raise ValueError("waiting normalization and starvation threshold must be positive")


@dataclass(frozen=True)
class DQNConfig:
    """Small, deliberately conservative Stable-Baselines3 DQN defaults."""

    total_timesteps: int = 100_000
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.4
    exploration_final_eps: float = 0.05
    model_path: Path = Path("models/dqn_traffic")
    log_dir: Path = Path("logs")


@dataclass(frozen=True)
class PPOConfig:
    """Stable-Baselines3 PPO defaults for comparison with DQN."""

    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2_048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    policy: str = "MlpPolicy"
    device: str = "cpu"
    model_path: Path = Path("models/ppo_traffic")
    log_dir: Path = Path("logs/ppo")
