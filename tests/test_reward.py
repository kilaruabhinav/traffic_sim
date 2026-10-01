import pytest

from traffic_sim.config import TrafficConfig
from traffic_sim.environment import TrafficEnv


def reward_env(**changes) -> TrafficEnv:
    values = {
        "arrival_probabilities": (0.0, 0.0, 0.0, 0.0),
        "episode_length": 20,
    }
    values.update(changes)
    env = TrafficEnv(TrafficConfig(**values))
    env.reset(seed=1)
    return env


def set_lane(env: TrafficEnv, lane: int, queue: int, mean_wait: float = 0.0) -> None:
    env.traffic.lanes[lane].queue_length = queue
    env.traffic.lanes[lane].total_waiting = queue * mean_wait


def step_info(env: TrafficEnv, action: int = 0) -> dict:
    return env.step(action)[4]


def test_empty_traffic_has_no_pressure_penalties():
    info = step_info(reward_env())
    components = info["reward_components"]
    assert components["queue_penalty"] == 0.0
    assert components["waiting_penalty"] == 0.0
    assert components["overflow_penalty"] == 0.0
    assert components["starvation_penalty"] == 0.0
    assert info["queue_pressure"] == 0.0
    assert info["waiting_pressure"] == 0.0


def test_passing_vehicles_produces_normalized_throughput_reward():
    env = reward_env()
    set_lane(env, 0, 2)
    components = step_info(env)["reward_components"]
    assert components["throughput"] == 1.0
    assert components["progress_reward"] > 0.0


def test_queue_pressure_increases_with_queue_length():
    low = reward_env()
    high = reward_env()
    set_lane(low, 1, 5)
    set_lane(high, 1, 10)
    assert step_info(high)["queue_pressure"] > step_info(low)["queue_pressure"]


def test_waiting_pressure_increases_with_mean_wait():
    low = reward_env()
    high = reward_env()
    set_lane(low, 1, 2, mean_wait=5)
    set_lane(high, 1, 2, mean_wait=20)
    assert step_info(high)["waiting_pressure"] > step_info(low)["waiting_pressure"]


@pytest.mark.parametrize("queue", [0, 10, 20])
def test_overflow_penalty_is_zero_at_or_below_soft_limit(queue: int):
    env = reward_env()
    set_lane(env, 1, queue)
    assert step_info(env)["reward_components"]["overflow_penalty"] == 0.0


def test_overflow_penalty_grows_quadratically_to_hard_limit():
    penalties = []
    for queue in (25, 30, 40):
        env = reward_env()
        set_lane(env, 1, queue)
        penalties.append(step_info(env)["reward_components"]["overflow_penalty"])
    assert abs(penalties[0]) < abs(penalties[1]) < abs(penalties[2])
    assert penalties == pytest.approx([-0.015625, -0.0625, -0.25])


def test_starvation_penalty_starts_above_threshold_and_increases():
    at_threshold = reward_env()
    above = reward_env()
    farther_above = reward_env()
    # Waiting increments once before reward calculation.
    set_lane(at_threshold, 1, 2, mean_wait=14)
    set_lane(above, 1, 2, mean_wait=16)
    set_lane(farther_above, 1, 2, mean_wait=24)
    penalties = [
        step_info(env)["reward_components"]["starvation_penalty"]
        for env in (at_threshold, above, farther_above)
    ]
    assert penalties[0] == 0.0
    assert abs(penalties[1]) < abs(penalties[2])


def test_queue_progress_is_positive_for_improvement_and_negative_for_worsening():
    improving = reward_env()
    set_lane(improving, 0, 2)
    assert step_info(improving)["reward_components"]["progress_reward"] > 0.0

    worsening = reward_env(arrival_probabilities=(1.0, 1.0, 1.0, 1.0))
    assert step_info(worsening)["reward_components"]["progress_reward"] < 0.0


def test_switch_penalty_only_applies_when_switch_starts():
    env = reward_env(min_green_time=0, switch_time=1)
    assert step_info(env, 0)["reward_components"]["switch_penalty"] == 0.0
    assert step_info(env, 1)["reward_components"]["switch_penalty"] == -0.02
    assert step_info(env, 1)["reward_components"]["switch_penalty"] == 0.0


def test_reward_stays_in_small_range_for_severe_realistic_state():
    env = reward_env()
    for lane in range(4):
        set_lane(env, lane, 40, mean_wait=60)
    _, reward, _, _, info = env.step(0)
    assert -3.0 <= reward <= 1.5
    assert reward == info["reward_components"]["total_reward"]
