import inspect
from pathlib import Path

import numpy as np
import pytest

import traffic_sim.evaluate as evaluation
from traffic_sim.config import DQNConfig, PPOConfig, TrafficConfig
from traffic_sim.controllers import RuleBasedController
from traffic_sim.environment import TrafficEnv
from traffic_sim.scenarios import TrafficScenario, generate_scenario
from traffic_sim.utils import EpisodeResult, mean_std_ci95, paired_differences, summarize_results
from traffic_sim.visualizer import TrafficVisualizer


def result(index: int, controller: str, reward: float, waiting: float, passed: int, violations: int):
    return EpisodeResult(index, controller, reward, waiting, 7 + index, passed, violations,
                         violations > 0, violations > 0, False)


def test_unseeded_reset_continues_rng_stream_instead_of_replaying_episode():
    config = TrafficConfig(episode_length=6, arrival_probabilities=(0.5,) * 4)
    env = TrafficEnv(config)
    env.reset(seed=123)  # controlled starting state for a non-probabilistic assertion
    first = [env.step(0)[4]["attempted_arrivals"].tolist() for _ in range(6)]
    env.reset()
    second = [env.step(0)[4]["attempted_arrivals"].tolist() for _ in range(6)]
    expected_rng = np.random.default_rng(123)
    expected = (expected_rng.random((12, 4)) < 0.5).astype(int).tolist()
    assert first == expected[:6]
    assert second == expected[6:]
    assert first != second


def test_training_defaults_do_not_force_an_sb3_seed():
    assert not hasattr(DQNConfig(), "seed")
    assert not hasattr(PPOConfig(), "seed")


def test_scenario_replay_uses_exact_attempted_arrivals():
    config = TrafficConfig(episode_length=3, arrival_probabilities=(0.0,) * 4)
    attempts = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=bool)
    env = TrafficEnv(config, TrafficScenario(attempts))
    env.reset()
    observed = [env.step(0)[4]["attempted_arrivals"].tolist() for _ in range(3)]
    assert observed == attempts.astype(int).tolist()


def test_evaluation_generates_new_scenarios_and_shares_each_one(monkeypatch):
    config = TrafficConfig(episode_length=4)
    seen: dict[int, list[int]] = {}

    def fake_run(name, passed_config, scenario, controller, scenario_index):
        assert passed_config is config
        seen.setdefault(scenario_index, []).append(id(scenario))
        return result(scenario_index, name, 1.0, 2.0, 3, 0)

    monkeypatch.setattr(evaluation, "run_controller_on_scenario", fake_run)
    report = evaluation.evaluate_controller_factories(
        [("A", RuleBasedController), ("B", RuleBasedController)],
        episodes=3,
        traffic_config=config,
        rng=np.random.default_rng(987),
    )
    assert all(len(set(ids)) == 1 for ids in seen.values())
    assert len({scenario.arrivals.tobytes() for scenario in report.scenarios}) == 3
    assert len(report.episodes) == 6


def test_summary_statistics_and_confidence_intervals():
    episodes = [result(0, "X", 1, 10, 5, 0), result(1, "X", 3, 20, 7, 2)]
    stats = summarize_results("X", episodes)
    mean, std, low, high = mean_std_ci95([1, 3])
    assert (mean, std) == pytest.approx((2.0, np.sqrt(2)))
    assert (stats.reward_ci95_low, stats.reward_ci95_high) == pytest.approx((low, high))
    assert stats.median_total_reward == 2
    assert stats.mean_waiting_time == 15
    assert stats.episodes_with_any_violation == 1
    assert stats.percent_exceeding_soft_limit == 50


def test_paired_ppo_minus_dqn_differences():
    dqn = [result(0, "DQN", 1, 10, 5, 3), result(1, "DQN", 2, 20, 7, 4)]
    ppo = [result(0, "PPO", 4, 8, 6, 1), result(1, "PPO", 6, 17, 10, 2)]
    paired = {item.metric: item for item in paired_differences(dqn, ppo)}
    assert paired["reward"].mean == 3.5
    assert paired["waiting"].mean == -2.5
    assert paired["vehicles_passed"].mean == 2.0
    assert paired["violations"].mean == -2.0


def test_dqn_and_ppo_evaluation_paths_load_and_run(monkeypatch):
    class FakeModel:
        def predict(self, observation, deterministic):
            assert deterministic
            return 0, None

    monkeypatch.setattr(evaluation.DQN, "load", lambda path: FakeModel())
    monkeypatch.setattr(evaluation.PPO, "load", lambda path: FakeModel())
    report = evaluation.evaluate_with_ppo(
        Path("dqn"), Path("ppo"), episodes=2, traffic_config=TrafficConfig(episode_length=4)
    )
    assert [item.controller for item in report.summaries] == ["Random", "Rule-based", "DQN", "PPO"]
    assert len(report.paired) == 4


def test_visualizer_constructor_has_no_seed_parameter():
    assert "seed" not in inspect.signature(TrafficVisualizer).parameters
