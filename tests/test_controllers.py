from traffic_sim.config import TrafficConfig
from traffic_sim.controllers import RandomController, RuleBasedController
from traffic_sim.environment import TrafficEnv
from traffic_sim.utils import run_controller_episodes


def env_factory():
    return TrafficEnv(TrafficConfig(episode_length=12))


def test_random_controller_runs():
    result = run_controller_episodes("Random", env_factory, RandomController, episodes=2)
    assert result.controller == "Random"
    assert result.mean_vehicles_passed >= 0


def test_rule_based_controller_selects_most_congested_lane_and_runs():
    env = env_factory()
    observation, _ = env.reset(seed=1)
    observation[2] = 1.0
    assert RuleBasedController().act(observation) == 2
    result = run_controller_episodes(
        "Rule-based", env_factory, RuleBasedController, episodes=2
    )
    assert result.mean_vehicles_passed >= 0
