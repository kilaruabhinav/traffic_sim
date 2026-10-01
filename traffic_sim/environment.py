"""Gymnasium environment for numerical four-lane signal control."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from traffic_sim.config import TrafficConfig
from traffic_sim.scenarios import TrafficScenario
from traffic_sim.traffic import TrafficSystem


class TrafficEnv(gym.Env[np.ndarray, int]):
    """A four-lane queueing environment with one active green signal.

    Observation (15 floats, all in ``[0, 1]``): queue utilization for lanes
    1-4, normalized mean wait for lanes 1-4, a four-value one-hot green lane,
    normalized green duration, a switching flag, and normalized pending lane.
    The pending lane value is zero when no switch is underway; the switching
    flag disambiguates that value from lane 1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: TrafficConfig | None = None,
        scenario: TrafficScenario | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TrafficConfig()
        if scenario is not None and len(scenario.arrivals) < self.config.episode_length:
            raise ValueError("scenario is shorter than the configured episode")
        self.scenario = scenario
        self.action_space = spaces.Discrete(self.config.num_lanes)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(15,), dtype=np.float32)
        self.traffic = TrafficSystem(
            self.config.arrival_probabilities,
            self.config.max_queue_lengths,
            np.random.default_rng(),
        )
        self.current_green_lane = 0
        self.green_time = 0
        self.pending_lane: int | None = None
        self.switch_remaining = 0
        self.elapsed_steps = 0
        self.cumulative_vehicles_passed = 0
        self.cumulative_violations = 0
        self.episode_max_queue = 0

    @property
    def green_lanes(self) -> np.ndarray:
        """Boolean one-hot representation of the active signal."""
        result = np.zeros(self.config.num_lanes, dtype=bool)
        result[self.current_green_lane] = True
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.traffic.reset(self.np_random)
        self.current_green_lane = int((options or {}).get("initial_green_lane", 0))
        if not self.action_space.contains(self.current_green_lane):
            raise ValueError("initial_green_lane must be in [0, 3]")
        self.green_time = 0
        self.pending_lane = None
        self.switch_remaining = 0
        self.elapsed_steps = 0
        self.cumulative_vehicles_passed = 0
        self.cumulative_violations = 0
        self.episode_max_queue = 0
        return self._observation(), self._info(0, np.zeros(4, dtype=np.int64), self._zero_reward())

    def _advance_signal(self, requested_lane: int) -> tuple[bool, bool]:
        """Return ``(can_discharge, switch_initiated)`` after advancing the signal."""
        if self.switch_remaining > 0:
            self.switch_remaining -= 1
            if self.switch_remaining == 0:
                assert self.pending_lane is not None
                self.current_green_lane = self.pending_lane
                self.pending_lane = None
                self.green_time = 0
            return False, False

        must_switch = self.green_time >= self.config.max_green_time
        wants_switch = requested_lane != self.current_green_lane
        may_switch = self.green_time >= self.config.min_green_time

        if may_switch and (wants_switch or must_switch):
            target = requested_lane if wants_switch else (self.current_green_lane + 1) % 4
            if self.config.switch_time > 0:
                self.pending_lane = target
                self.switch_remaining = self.config.switch_time
                return False, True
            self.current_green_lane = target
            self.green_time = 1
            return True, True

        self.green_time += 1
        return True, False

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.config.num_lanes - 1}]")
        previous_queue_pressure = self._queue_pressure()
        can_discharge, switch_initiated = self._advance_signal(int(action))
        attempted_arrivals = (
            None if self.scenario is None else self.scenario.arrivals[self.elapsed_steps]
        )
        arrivals = self.traffic.generate_arrivals(attempted_arrivals)
        self.traffic.increment_waiting()
        vehicles_passed = (
            self.traffic.discharge(self.current_green_lane, self.config.vehicles_per_step)
            if can_discharge
            else 0
        )
        self.elapsed_steps += 1
        self.cumulative_vehicles_passed += vehicles_passed

        queues = self.traffic.queue_lengths
        violations = int(np.count_nonzero(queues > np.asarray(self.config.queue_limits)))
        self.cumulative_violations += violations
        self.episode_max_queue = max(self.episode_max_queue, int(queues.max()))
        components = self._reward_components(
            vehicles_passed, previous_queue_pressure, switch_initiated
        )
        reward = components["total_reward"]
        truncated = self.elapsed_steps >= self.config.episode_length
        info = self._info(vehicles_passed, arrivals, components)
        info["attempted_arrivals"] = self.traffic.last_attempted_arrivals.copy()
        return self._observation(), reward, False, truncated, info

    def _normalized_pressures(self) -> tuple[float, float, float, float]:
        """Return bounded queue, waiting, overflow, and starvation pressures."""
        queues = self.traffic.queue_lengths.astype(np.float64)
        hard_limits = np.asarray(self.config.max_queue_lengths, dtype=np.float64)
        limits = np.asarray(self.config.queue_limits, dtype=np.float64)
        queue_pressure = float(np.mean(queues / hard_limits))
        wait_ratios = np.clip(
            self.traffic.average_waiting_by_lane / self.config.waiting_time_normalizer,
            0.0,
            1.0,
        )
        waiting_pressure = float(np.mean(wait_ratios))
        excess = np.maximum(queues - limits, 0.0)
        overflow_ratios = excess / (hard_limits - limits)
        overflow_pressure = float(np.mean(np.square(overflow_ratios)))
        average_waits = self.traffic.average_waiting_by_lane
        starvation_ratios = np.clip(
            np.maximum(average_waits - self.config.starvation_threshold, 0.0)
            / self.config.starvation_threshold,
            0.0,
            1.0,
        )
        starvation_pressure = float(np.mean(starvation_ratios))
        return queue_pressure, waiting_pressure, overflow_pressure, starvation_pressure

    def _queue_pressure(self) -> float:
        queues = self.traffic.queue_lengths.astype(np.float64)
        hard_limits = np.asarray(self.config.max_queue_lengths, dtype=np.float64)
        return float(np.mean(queues / hard_limits))

    def _reward_components(
        self,
        vehicles_passed: int,
        previous_queue_pressure: float,
        switch_initiated: bool,
    ) -> dict[str, float]:
        queue_pressure, waiting_pressure, overflow_pressure, starvation_pressure = (
            self._normalized_pressures()
        )
        throughput = vehicles_passed / self.config.vehicles_per_step
        queue_progress = previous_queue_pressure - queue_pressure
        components = {
            "throughput": self.config.throughput_reward_weight * throughput,
            "queue_penalty": -self.config.queue_penalty_weight * queue_pressure,
            "waiting_penalty": -self.config.waiting_penalty_weight * waiting_pressure,
            "overflow_penalty": -self.config.overflow_penalty_weight * overflow_pressure,
            "starvation_penalty": -self.config.starvation_penalty_weight * starvation_pressure,
            "progress_reward": self.config.progress_reward_weight * queue_progress,
            "switch_penalty": self.config.switch_penalty if switch_initiated else 0.0,
        }
        components["total_reward"] = float(sum(components.values()))
        return components

    @staticmethod
    def _zero_reward() -> dict[str, float]:
        return {
            "throughput": 0.0,
            "queue_penalty": 0.0,
            "waiting_penalty": 0.0,
            "overflow_penalty": 0.0,
            "starvation_penalty": 0.0,
            "progress_reward": 0.0,
            "switch_penalty": 0.0,
            "total_reward": 0.0,
        }

    def _observation(self) -> np.ndarray:
        queues = self.traffic.queue_lengths / np.asarray(self.config.max_queue_lengths)
        waits = np.clip(
            self.traffic.average_waiting_by_lane / self.config.waiting_time_normalizer, 0.0, 1.0
        )
        green = self.green_lanes.astype(np.float64)
        green_time = min(self.green_time / max(self.config.max_green_time, 1), 1.0)
        switching = float(self.switch_remaining > 0)
        pending = 0.0 if self.pending_lane is None else self.pending_lane / 3.0
        return np.concatenate((queues, waits, green, [green_time, switching, pending])).astype(np.float32)

    def _info(
        self, vehicles_passed: int, arrivals: np.ndarray, reward_components: dict[str, float]
    ) -> dict[str, Any]:
        queues = self.traffic.queue_lengths
        violations = int(np.count_nonzero(queues > np.asarray(self.config.queue_limits)))
        queue_pressure, waiting_pressure, overflow_pressure, starvation_pressure = (
            self._normalized_pressures()
        )
        return {
            "vehicles_passed": vehicles_passed,
            "cumulative_vehicles_passed": self.cumulative_vehicles_passed,
            "arrivals": arrivals.copy(),
            "total_waiting": float(self.traffic.total_waiting_by_lane.sum()),
            "waiting_by_lane": self.traffic.total_waiting_by_lane.copy(),
            "queue_lengths": queues.copy(),
            "max_queue": int(queues.max()),
            "episode_max_queue": self.episode_max_queue,
            "queue_limit_violations": violations,
            "cumulative_queue_limit_violations": self.cumulative_violations,
            "current_green_lane": self.current_green_lane,
            "green_time": self.green_time,
            "is_switching": self.switch_remaining > 0,
            "switch_remaining": self.switch_remaining,
            "pending_lane": self.pending_lane,
            "elapsed_steps": self.elapsed_steps,
            "reward_components": reward_components.copy(),
            "queue_pressure": queue_pressure,
            "waiting_pressure": waiting_pressure,
            "overflow_pressure": overflow_pressure,
            "starvation_pressure": starvation_pressure,
            "dropped_arrivals": self.traffic.dropped_arrivals,
        }

    def close(self) -> None:
        """Release resources (none are held by the numerical environment)."""
