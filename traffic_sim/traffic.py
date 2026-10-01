"""Traffic queue mechanics, independent of Gymnasium and RL code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class LaneTraffic:
    """Aggregate queue state for one incoming lane."""

    queue_length: int = 0
    total_waiting: float = 0.0

    @property
    def average_waiting(self) -> float:
        return self.total_waiting / self.queue_length if self.queue_length else 0.0


class TrafficSystem:
    """Owns arrivals, waiting accumulation, and vehicle discharge."""

    def __init__(
        self,
        arrival_probabilities: Sequence[float],
        max_queue_lengths: Sequence[int],
        rng: np.random.Generator,
    ) -> None:
        self.arrival_probabilities = np.asarray(arrival_probabilities, dtype=np.float64)
        self.max_queue_lengths = np.asarray(max_queue_lengths, dtype=np.int64)
        self.rng = rng
        self.lanes = [LaneTraffic() for _ in arrival_probabilities]
        self.dropped_arrivals = 0
        self.last_attempted_arrivals = np.zeros(len(self.lanes), dtype=np.int64)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.lanes = [LaneTraffic() for _ in self.arrival_probabilities]
        self.dropped_arrivals = 0
        self.last_attempted_arrivals = np.zeros(len(self.lanes), dtype=np.int64)

    @property
    def queue_lengths(self) -> np.ndarray:
        return np.asarray([lane.queue_length for lane in self.lanes], dtype=np.int64)

    @property
    def total_waiting_by_lane(self) -> np.ndarray:
        return np.asarray([lane.total_waiting for lane in self.lanes], dtype=np.float64)

    @property
    def average_waiting_by_lane(self) -> np.ndarray:
        return np.asarray([lane.average_waiting for lane in self.lanes], dtype=np.float64)

    def generate_arrivals(self, attempted_arrivals: Sequence[bool] | None = None) -> np.ndarray:
        """Generate at most one arrival per lane and return accepted arrivals."""
        if attempted_arrivals is None:
            attempted = self.rng.random(len(self.lanes)) < self.arrival_probabilities
        else:
            attempted = np.asarray(attempted_arrivals, dtype=bool)
            if attempted.shape != (len(self.lanes),):
                raise ValueError("attempted arrivals must contain one value per lane")
        self.last_attempted_arrivals = attempted.astype(np.int64)
        accepted = np.zeros(len(self.lanes), dtype=np.int64)
        for index, arrives in enumerate(attempted):
            if not arrives:
                continue
            lane = self.lanes[index]
            if lane.queue_length < self.max_queue_lengths[index]:
                lane.queue_length += 1
                accepted[index] = 1
            else:
                self.dropped_arrivals += 1
        return accepted

    def increment_waiting(self) -> None:
        """Add one vehicle-step of waiting for every currently queued vehicle."""
        for lane in self.lanes:
            lane.total_waiting += lane.queue_length

    def discharge(self, lane_index: int, capacity: int) -> int:
        """Remove vehicles from one lane, preserving its aggregate mean wait."""
        lane = self.lanes[lane_index]
        departed = min(capacity, lane.queue_length)
        if departed == 0:
            return 0
        remaining_fraction = (lane.queue_length - departed) / lane.queue_length
        lane.total_waiting *= remaining_fraction
        lane.queue_length -= departed
        if lane.queue_length == 0:
            lane.total_waiting = 0.0
        return departed
