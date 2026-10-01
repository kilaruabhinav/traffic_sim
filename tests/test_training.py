from pathlib import Path

from stable_baselines3 import PPO

from traffic_sim.config import DQNConfig, PPOConfig, TrafficConfig
from traffic_sim.environment import TrafficEnv
from traffic_sim.train import train_dqn, train_ppo


def test_dqn_short_training_run(tmp_path: Path):
    settings = DQNConfig(
        total_timesteps=32,
        buffer_size=100,
        learning_starts=8,
        batch_size=8,
        target_update_interval=10,
        model_path=tmp_path / "model",
        log_dir=tmp_path / "logs",
    )
    model = train_dqn(settings, TrafficConfig(episode_length=16))
    observation, _ = TrafficEnv(TrafficConfig(episode_length=16)).reset(seed=2)
    action, _ = model.predict(observation, deterministic=True)
    assert 0 <= int(action) < 4
    assert (tmp_path / "model.zip").exists()
    assert (tmp_path / "logs" / "run_config.json").exists()


def test_ppo_short_training_run(tmp_path: Path):
    settings = PPOConfig(
        total_timesteps=128,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        model_path=tmp_path / "ppo_model",
        log_dir=tmp_path / "ppo_logs",
    )
    model = train_ppo(settings, TrafficConfig(episode_length=16))
    observation, _ = TrafficEnv(TrafficConfig(episode_length=16)).reset(seed=2)
    action, _ = model.predict(observation, deterministic=True)
    assert isinstance(model, PPO)
    assert 0 <= int(action) < 4
    assert (tmp_path / "ppo_model.zip").exists()
    assert (tmp_path / "ppo_logs" / "run_config.json").exists()


def test_ppo_resume_adds_training_timesteps(tmp_path: Path):
    traffic = TrafficConfig(episode_length=16)
    initial = PPOConfig(
        total_timesteps=64,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        model_path=tmp_path / "initial_ppo",
        log_dir=tmp_path / "initial_logs",
    )
    initial_model = train_ppo(initial, traffic)
    initial_timesteps = initial_model.num_timesteps

    resumed = PPOConfig(
        total_timesteps=64,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        model_path=tmp_path / "resumed_ppo",
        log_dir=tmp_path / "resumed_logs",
    )
    resumed_model = train_ppo(resumed, traffic, resume_from=initial.model_path)

    assert resumed_model.num_timesteps == initial_timesteps + resumed.total_timesteps
    assert resumed_model.learning_rate == resumed.learning_rate
    assert resumed_model.n_steps == resumed.n_steps
    assert resumed_model.batch_size == resumed.batch_size
    assert resumed_model.n_epochs == resumed.n_epochs
    assert (tmp_path / "resumed_ppo.zip").exists()
