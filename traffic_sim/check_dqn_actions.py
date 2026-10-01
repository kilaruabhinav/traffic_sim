"""Report deterministic DQN action frequencies over fresh evaluation episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from traffic_sim.config import TrafficConfig
from traffic_sim.environment import TrafficEnv


def count_dqn_actions(
    model_path: Path = Path("models/dqn_traffic_bounded"),
    episodes: int = 10,
    traffic_config: TrafficConfig | None = None,
) -> np.ndarray:
    """Count deterministic actions on fresh random traffic episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    model = DQN.load(str(model_path))
    config = traffic_config or TrafficConfig()
    counts = np.zeros(config.num_lanes, dtype=np.int64)

    for _ in range(episodes):
        env = TrafficEnv(config)
        try:
            observation, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                action_index = int(action)
                counts[action_index] += 1
                observation, _, terminated, truncated, _ = env.step(action_index)
                done = terminated or truncated
        finally:
            env.close()

    return counts


def format_action_distribution(counts: np.ndarray) -> str:
    """Format action counts and percentages for human-readable diagnostics."""
    total = int(counts.sum())
    lines = ["## DQN Action Distribution", ""]
    for lane, count in enumerate(counts, start=1):
        percentage = 100.0 * int(count) / total if total else 0.0
        lines.append(f"Lane {lane}: {int(count)} actions ({percentage:.2f}%)")
    lines.extend(("", f"Total actions: {total}"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/dqn_traffic_bounded")
    )
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    print(format_action_distribution(count_dqn_actions(args.model_path, args.episodes)))


if __name__ == "__main__":
    main()
