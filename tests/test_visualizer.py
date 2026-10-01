from pathlib import Path

import numpy as np
import pytest

from traffic_sim.visualizer import (
    DQN,
    PPO,
    QueuedVehicle,
    TrafficVisualizer,
    load_controller_model,
)


def test_visualizer_loads_matching_model_class(monkeypatch):
    monkeypatch.setattr(DQN, "load", lambda path: ("dqn", path))
    monkeypatch.setattr(PPO, "load", lambda path: ("ppo", path))

    assert load_controller_model("dqn", Path("dqn_model")) == ("dqn", "dqn_model")
    assert load_controller_model("ppo", Path("ppo_model")) == ("ppo", "ppo_model")
    with pytest.raises(ValueError, match="unsupported algorithm"):
        load_controller_model("unknown", Path("model"))


def test_every_reported_departure_gets_distinct_animation():
    visualizer = TrafficVisualizer.__new__(TrafficVisualizer)
    visualizer.visual_queues = [
        [
            QueuedVehicle(0, "blue", 0, 0, 0, 0),
            QueuedVehicle(0, "orange", 0, 0, 0, 0),
        ],
        [],
        [],
        [],
    ]
    visualizer.moving_vehicles = []
    visualizer.next_vehicle_color = 0
    info = {
        "arrivals": np.zeros(4, dtype=np.int64),
        "current_green_lane": 0,
        "vehicles_passed": 2,
        "queue_lengths": np.zeros(4, dtype=np.int64),
    }

    visualizer._apply_visual_events(info)

    # Queue-origin identities persist until their animations finish, but are
    # excluded from the stationary queue drawing while in motion.
    assert len(visualizer.visual_queues[0]) == 2
    assert not visualizer._waiting_vehicles(0)
    assert len(visualizer.moving_vehicles) == 2
    assert [vehicle.color for vehicle in visualizer.moving_vehicles] == ["blue", "orange"]
    assert [vehicle.departure_offset for vehicle in visualizer.moving_vehicles] == [0, 1]
    assert len({(vehicle.start_x, vehicle.start_y) for vehicle in visualizer.moving_vehicles}) == 2

    for vehicle in visualizer.moving_vehicles:
        vehicle.progress = 1.0
    visualizer._remove_finished_departures()
    assert not visualizer.visual_queues[0]
    assert not visualizer.moving_vehicles
