"""Compare controllers on fresh, paired traffic-arrival scenarios."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import DQN, PPO

from traffic_sim.config import TrafficConfig
from traffic_sim.controllers import RandomController, RuleBasedController
from traffic_sim.scenarios import TrafficScenario, generate_scenario
from traffic_sim.utils import (
    Controller,
    EpisodeResult,
    EvaluationStats,
    PairedDifference,
    format_paired_results,
    format_results,
    paired_differences,
    run_controller_on_scenario,
    summarize_results,
)


class DQNController:
    def __init__(self, model: DQN) -> None:
        self.model = model

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


class PPOController:
    def __init__(self, model: PPO) -> None:
        self.model = model

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


@dataclass(frozen=True)
class EvaluationReport:
    scenarios: list[TrafficScenario]
    episodes: list[EpisodeResult]
    summaries: list[EvaluationStats]
    paired: list[PairedDifference]


def evaluate_controller_factories(
    controller_factories: list[tuple[str, Callable[[], Controller]]],
    episodes: int = 10,
    traffic_config: TrafficConfig | None = None,
    rng: np.random.Generator | None = None,
) -> EvaluationReport:
    """Generate each scenario once, then replay it for every controller."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    config = traffic_config or TrafficConfig()
    generator = rng if rng is not None else np.random.default_rng()
    scenarios = [generate_scenario(config, generator) for _ in range(episodes)]
    episode_results: list[EpisodeResult] = []
    for scenario_index, scenario in enumerate(scenarios):
        for name, factory in controller_factories:
            episode_results.append(
                run_controller_on_scenario(name, config, scenario, factory(), scenario_index)
            )
    summaries = [
        summarize_results(
            name, [item for item in episode_results if item.controller == name]
        )
        for name, _ in controller_factories
    ]
    dqn = [item for item in episode_results if item.controller == "DQN"]
    ppo = [item for item in episode_results if item.controller == "PPO"]
    paired = paired_differences(dqn, ppo) if dqn and ppo else []
    return EvaluationReport(scenarios, episode_results, summaries, paired)


def evaluate_baselines(
    episodes: int = 10,
    traffic_config: TrafficConfig | None = None,
) -> EvaluationReport:
    return evaluate_controller_factories(
        [("Random", RandomController), ("Rule-based", RuleBasedController)],
        episodes,
        traffic_config,
    )


def evaluate_all(
    model_path: Path,
    episodes: int = 10,
    traffic_config: TrafficConfig | None = None,
) -> EvaluationReport:
    dqn_model = DQN.load(str(model_path))
    return evaluate_controller_factories(
        [
            ("Random", RandomController),
            ("Rule-based", RuleBasedController),
            ("DQN", lambda: DQNController(dqn_model)),
        ],
        episodes,
        traffic_config,
    )


def evaluate_with_ppo(
    dqn_model_path: Path,
    ppo_model_path: Path,
    episodes: int = 10,
    traffic_config: TrafficConfig | None = None,
) -> EvaluationReport:
    dqn_model = DQN.load(str(dqn_model_path))
    ppo_model = PPO.load(str(ppo_model_path))
    return evaluate_controller_factories(
        [
            ("Random", RandomController),
            ("Rule-based", RuleBasedController),
            ("DQN", lambda: DQNController(dqn_model)),
            ("PPO", lambda: PPOController(ppo_model)),
        ],
        episodes,
        traffic_config,
    )


def save_report(report: EvaluationReport, path: Path) -> None:
    """Save raw results plus exact attempted-arrival scenarios as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenarios": [
            {"scenario_index": index, "attempted_arrivals": scenario.arrivals.astype(int).tolist()}
            for index, scenario in enumerate(report.scenarios)
        ],
        "episode_results": [item.to_dict() for item in report.episodes],
        "summaries": [item.to_dict() for item in report.summaries],
        "ppo_minus_dqn": [item.to_dict() for item in report.paired],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/dqn_traffic"))
    parser.add_argument("--ppo-model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None, help="save raw results and scenarios as JSON")
    parser.add_argument("--baselines-only", action="store_true", help="run without loading learned models")
    args = parser.parse_args()
    if args.baselines_only:
        report = evaluate_baselines(args.episodes)
    elif args.ppo_model_path is not None:
        report = evaluate_with_ppo(args.model_path, args.ppo_model_path, args.episodes)
    else:
        report = evaluate_all(args.model_path, args.episodes)
    print(format_results(report.summaries))
    if report.paired:
        print("\n" + format_paired_results(report.paired))
    if args.output is not None:
        save_report(report, args.output)
        print(f"\nSaved raw results and scenarios to {args.output}")


if __name__ == "__main__":
    main()
