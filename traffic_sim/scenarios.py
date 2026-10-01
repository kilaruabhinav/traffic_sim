"""Internal traffic-arrival scenarios used for fair controller comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_sim.config import TrafficConfig


@dataclass(frozen=True)
class TrafficScenario:
    """Attempted lane arrivals for every step of one episode.

    Values describe arrival attempts, before the environment rejects an arrival
    at a full hard-limit queue.  This is the random input to the traffic model.
    """

    arrivals: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.arrivals, dtype=bool).copy()
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("scenario arrivals must have shape (steps, 4)")
        values.setflags(write=False)
        object.__setattr__(self, "arrivals", values)


def generate_scenario(
    config: TrafficConfig,
    rng: np.random.Generator | None = None,
) -> TrafficScenario:
    """Generate a fresh scenario from OS entropy unless an internal RNG is supplied."""
    generator = rng if rng is not None else np.random.default_rng()
    probabilities = np.asarray(config.arrival_probabilities, dtype=np.float64)
    attempts = generator.random((config.episode_length, config.num_lanes)) < probabilities
    return TrafficScenario(attempts)
