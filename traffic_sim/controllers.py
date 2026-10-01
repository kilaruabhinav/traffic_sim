"""Non-learning controller baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RandomController:
    """Choose each lane uniformly using an independent entropy-backed generator."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, observation: np.ndarray) -> int:
        del observation
        return int(self.rng.integers(0, 4))


@dataclass
class RuleBasedController:
    """Transparent baseline that prioritizes queue size and mean waiting."""

    queue_weight: float = 1.0
    waiting_weight: float = 0.7

    def act(self, observation: np.ndarray) -> int:
        queue_utilization = observation[0:4]
        normalized_waiting = observation[4:8]
        scores = self.queue_weight * queue_utilization + self.waiting_weight * normalized_waiting
        return int(np.argmax(scores))
