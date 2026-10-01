"""Shared experiment, statistics, and reporting helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from traffic_sim.config import TrafficConfig
from traffic_sim.environment import TrafficEnv
from traffic_sim.scenarios import TrafficScenario, generate_scenario


class Controller(Protocol):
    def act(self, observation: np.ndarray) -> int: ...


@dataclass(frozen=True)
class EpisodeResult:
    scenario_index: int
    controller: str
    reward: float
    total_waiting: float
    max_queue: int
    vehicles_passed: int
    violations: int
    any_violation: bool
    exceeded_soft_limit: bool
    hit_hard_limit: bool

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationStats:
    controller: str
    episodes: int
    mean_total_reward: float
    std_total_reward: float
    median_total_reward: float
    reward_ci95_low: float
    reward_ci95_high: float
    min_reward: float
    max_reward: float
    mean_waiting_time: float
    std_waiting_time: float
    median_waiting_time: float
    waiting_ci95_low: float
    waiting_ci95_high: float
    worst_total_waiting: float
    mean_max_queue: float
    maximum_observed_queue: int
    worst_max_queue: int
    mean_vehicles_passed: float
    std_vehicles_passed: float
    mean_queue_limit_violations: float
    episodes_with_any_violation: int
    percent_exceeding_soft_limit: float
    percent_hitting_hard_limit: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PairedDifference:
    metric: str
    mean: float
    std: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def mean_std_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return mean, sample standard deviation, and normal 95% CI endpoints."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("at least one value is required")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    margin = 1.96 * std / np.sqrt(array.size)
    return mean, std, mean - margin, mean + margin


def summarize_results(controller: str, results: Sequence[EpisodeResult]) -> EvaluationStats:
    if not results:
        raise ValueError("at least one episode result is required")
    rewards = np.asarray([item.reward for item in results], dtype=np.float64)
    waits = np.asarray([item.total_waiting for item in results], dtype=np.float64)
    passed = np.asarray([item.vehicles_passed for item in results], dtype=np.float64)
    max_queues = np.asarray([item.max_queue for item in results], dtype=np.int64)
    violations = np.asarray([item.violations for item in results], dtype=np.float64)
    reward_mean, reward_std, reward_low, reward_high = mean_std_ci95(rewards)
    wait_mean, wait_std, wait_low, wait_high = mean_std_ci95(waits)
    passed_mean, passed_std, _, _ = mean_std_ci95(passed)
    count = len(results)
    return EvaluationStats(
        controller, count, reward_mean, reward_std, float(np.median(rewards)),
        reward_low, reward_high, float(np.min(rewards)), float(np.max(rewards)),
        wait_mean, wait_std, float(np.median(waits)), wait_low, wait_high,
        float(np.max(waits)), float(np.mean(max_queues)), int(np.max(max_queues)),
        int(np.max(max_queues)), passed_mean, passed_std, float(np.mean(violations)),
        sum(item.any_violation for item in results),
        100.0 * sum(item.exceeded_soft_limit for item in results) / count,
        100.0 * sum(item.hit_hard_limit for item in results) / count,
    )


def run_controller_on_scenario(
    name: str,
    config: TrafficConfig,
    scenario: TrafficScenario,
    controller: Controller,
    scenario_index: int,
) -> EpisodeResult:
    """Run one controller against an already-generated arrival scenario."""
    env = TrafficEnv(config, scenario=scenario)
    try:
        observation, _ = env.reset()
        total_reward = 0.0
        waiting_samples: list[float] = []
        exceeded_soft_limit = False
        hit_hard_limit = False
        done = False
        info: dict = {}
        while not done:
            observation, reward, terminated, truncated, info = env.step(controller.act(observation))
            total_reward += reward
            waiting_samples.append(info["total_waiting"])
            queues = np.asarray(info["queue_lengths"])
            exceeded_soft_limit |= bool(np.any(queues > np.asarray(config.queue_limits)))
            hit_hard_limit |= bool(np.any(queues >= np.asarray(config.max_queue_lengths)))
            done = terminated or truncated
        max_queue = int(info["episode_max_queue"])
        violations = int(info["cumulative_queue_limit_violations"])
        return EpisodeResult(
            scenario_index, name, float(total_reward), float(np.mean(waiting_samples)),
            max_queue, int(info["cumulative_vehicles_passed"]), violations,
            violations > 0, exceeded_soft_limit, hit_hard_limit,
        )
    finally:
        env.close()


def run_controller_episodes(
    name: str,
    env_factory: Callable[[], TrafficEnv],
    controller_factory: Callable[[], Controller],
    episodes: int = 10,
) -> EvaluationStats:
    """Evaluate one controller on fresh, independently generated scenarios."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    probe = env_factory()
    config = probe.config
    probe.close()
    episode_results = [
        run_controller_on_scenario(name, config, generate_scenario(config), controller_factory(), index)
        for index in range(episodes)
    ]
    return summarize_results(name, episode_results)


def paired_differences(
    dqn_results: Sequence[EpisodeResult],
    ppo_results: Sequence[EpisodeResult],
) -> list[PairedDifference]:
    """Summarize PPO minus DQN per-scenario differences."""
    dqn_by_scenario = {item.scenario_index: item for item in dqn_results}
    ppo_by_scenario = {item.scenario_index: item for item in ppo_results}
    if dqn_by_scenario.keys() != ppo_by_scenario.keys():
        raise ValueError("DQN and PPO results must cover the same scenarios")
    fields = {"reward": "reward", "waiting": "total_waiting", "vehicles_passed": "vehicles_passed", "violations": "violations"}
    comparisons: list[PairedDifference] = []
    for label, field in fields.items():
        differences = [
            float(getattr(ppo_by_scenario[index], field)) - float(getattr(dqn_by_scenario[index], field))
            for index in sorted(dqn_by_scenario)
        ]
        mean, std, low, high = mean_std_ci95(differences)
        comparisons.append(PairedDifference(label, mean, std, low, high))
    return comparisons


def format_results(results: Sequence[EvaluationStats]) -> str:
    """Format a compact overview followed by readable uncertainty details."""
    header = f"{'Controller':<12} {'Reward mean±sd':>22} {'Wait mean±sd':>22} {'MaxQ mean':>10} {'Passed mean±sd':>20} {'Viol mean':>11}"
    lines = [header, "-" * len(header)]
    for item in results:
        lines.append(
            f"{item.controller:<12} {item.mean_total_reward:>9.2f} ± {item.std_total_reward:<8.2f} "
            f"{item.mean_waiting_time:>9.2f} ± {item.std_waiting_time:<8.2f} "
            f"{item.mean_max_queue:>10.2f} {item.mean_vehicles_passed:>8.2f} ± {item.std_vehicles_passed:<7.2f} "
            f"{item.mean_queue_limit_violations:>11.2f}"
        )
    lines.append("")
    for item in results:
        lines.extend((
            f"{item.controller} ({item.episodes} scenarios)",
            f"  Reward: median {item.median_total_reward:.2f}, 95% CI [{item.reward_ci95_low:.2f}, {item.reward_ci95_high:.2f}], range [{item.min_reward:.2f}, {item.max_reward:.2f}]",
            f"  Waiting: median {item.median_waiting_time:.2f}, 95% CI [{item.waiting_ci95_low:.2f}, {item.waiting_ci95_high:.2f}], worst {item.worst_total_waiting:.2f}",
            f"  Queues: max observed {item.maximum_observed_queue}, worst episode {item.worst_max_queue}, soft-limit episodes {item.percent_exceeding_soft_limit:.1f}%, hard-limit episodes {item.percent_hitting_hard_limit:.1f}%",
            f"  Violations: {item.episodes_with_any_violation}/{item.episodes} episodes had at least one",
        ))
    return "\n".join(lines)


def format_paired_results(results: Sequence[PairedDifference]) -> str:
    lines = ["PPO - DQN paired differences (same scenarios)"]
    for item in results:
        lines.append(
            f"  {item.metric:<16} mean {item.mean:>9.2f}, sd {item.std:>9.2f}, 95% CI [{item.ci95_low:.2f}, {item.ci95_high:.2f}]"
        )
    return "\n".join(lines)
