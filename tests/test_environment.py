import numpy as np
from stable_baselines3.common.env_checker import check_env

from traffic_sim.config import TrafficConfig
from traffic_sim.environment import TrafficEnv


def small_config(**changes):
    values = dict(episode_length=20, arrival_probabilities=(0.0, 0.0, 0.0, 0.0))
    values.update(changes)
    return TrafficConfig(**values)


def test_reset_and_spaces_are_valid():
    env = TrafficEnv(small_config())
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert env.action_space.contains(0)
    assert sum(env.green_lanes) == 1
    assert info["current_green_lane"] == 0
    assert info["green_time"] == 0
    assert info["elapsed_steps"] == 0
    assert info["pending_lane"] is None
    check_env(env, warn=True)


def test_seed_reproduces_arrivals():
    config = small_config(arrival_probabilities=(0.5, 0.5, 0.5, 0.5))
    sequences = []
    for _ in range(2):
        env = TrafficEnv(config)
        env.reset(seed=123)
        sequences.append([env.step(0)[4]["arrivals"].tolist() for _ in range(10)])
    assert sequences[0] == sequences[1]


def test_vehicles_arrive_and_only_green_lane_discharges():
    env = TrafficEnv(small_config(arrival_probabilities=(1.0, 1.0, 1.0, 1.0), vehicles_per_step=1))
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    assert info["queue_lengths"].tolist() == [0, 1, 1, 1]
    assert info["vehicles_passed"] == 1


def test_waiting_increases_and_queues_never_negative():
    env = TrafficEnv(small_config(arrival_probabilities=(0.0, 1.0, 0.0, 0.0)))
    env.reset(seed=1)
    first = env.step(0)[4]["waiting_by_lane"][1]
    second = env.step(0)[4]["waiting_by_lane"][1]
    assert second > first
    for _ in range(20):
        info = env.step(0)[4]
        assert np.all(info["queue_lengths"] >= 0)


def test_queue_violations_and_reward_components():
    config = small_config(
        arrival_probabilities=(1.0, 1.0, 1.0, 1.0),
        queue_limits=(1, 1, 1, 1),
        max_queue_lengths=(5, 5, 5, 5),
        vehicles_per_step=1,
    )
    env = TrafficEnv(config)
    env.reset(seed=2)
    info = {}
    for _ in range(3):
        info = env.step(0)[4]
    assert info["queue_limit_violations"] == 3
    assert info["reward_components"]["overflow_penalty"] < 0
    assert set(info["reward_components"]) == {
        "throughput", "queue_penalty", "waiting_penalty", "overflow_penalty",
        "starvation_penalty", "progress_reward", "switch_penalty", "total_reward",
    }
    assert info["reward_components"]["total_reward"] == sum(
        value
        for name, value in info["reward_components"].items()
        if name != "total_reward"
    )


def test_signal_timing_minimum_switch_and_maximum():
    config = small_config(min_green_time=2, max_green_time=3, switch_time=1)
    env = TrafficEnv(config)
    env.reset(seed=1)
    env.step(1)
    assert env.current_green_lane == 0
    env.step(1)
    assert env.current_green_lane == 0
    env.step(1)
    assert env.switch_remaining == 1
    env.step(1)
    assert env.current_green_lane == 1
    assert sum(env.green_lanes) == 1

    env = TrafficEnv(config)
    env.reset(seed=1)
    for _ in range(3):
        env.step(0)
    env.step(0)
    assert env.switch_remaining == 1
    env.step(0)
    assert env.current_green_lane == 1


def test_complete_episode_truncates_not_terminates():
    env = TrafficEnv(small_config(episode_length=5))
    observation, _ = env.reset(seed=4)
    for index in range(5):
        observation, _, terminated, truncated, _ = env.step(0)
        assert env.observation_space.contains(observation)
        assert not terminated
        assert truncated is (index == 4)
