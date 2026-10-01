"""Train and save a Stable-Baselines3 DQN or PPO traffic controller."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from traffic_sim.config import DQNConfig, PPOConfig, TrafficConfig
from traffic_sim.environment import TrafficEnv


def train_dqn(
    dqn_config: DQNConfig | None = None,
    traffic_config: TrafficConfig | None = None,
) -> DQN:
    """Validate the environment, train DQN, and persist model plus settings."""
    settings = dqn_config or DQNConfig()
    environment_settings = traffic_config or TrafficConfig()
    check_env(TrafficEnv(environment_settings), warn=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(TrafficEnv(environment_settings), filename=str(settings.log_dir / "training"))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=settings.learning_rate,
        buffer_size=settings.buffer_size,
        learning_starts=settings.learning_starts,
        batch_size=settings.batch_size,
        gamma=settings.gamma,
        train_freq=settings.train_freq,
        gradient_steps=settings.gradient_steps,
        target_update_interval=settings.target_update_interval,
        exploration_fraction=settings.exploration_fraction,
        exploration_final_eps=settings.exploration_final_eps,
        verbose=1,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=settings.total_timesteps, progress_bar=False)
    model.save(settings.model_path)
    run_settings = {
        "dqn": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
        "traffic": asdict(environment_settings),
    }
    (settings.log_dir / "run_config.json").write_text(json.dumps(run_settings, indent=2) + "\n")
    env.close()
    return model


def train_ppo(
    ppo_config: PPOConfig | None = None,
    traffic_config: TrafficConfig | None = None,
    resume_from: Path | None = None,
) -> PPO:
    """Train a fresh PPO model or continue one for additional timesteps."""
    settings = ppo_config or PPOConfig()
    environment_settings = traffic_config or TrafficConfig()
    check_env(TrafficEnv(environment_settings), warn=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(TrafficEnv(environment_settings), filename=str(settings.log_dir / "training"))
    ppo_parameters = {
        "learning_rate": settings.learning_rate,
        "n_steps": settings.n_steps,
        "batch_size": settings.batch_size,
        "n_epochs": settings.n_epochs,
        "gamma": settings.gamma,
        "gae_lambda": settings.gae_lambda,
        "clip_range": settings.clip_range,
        "ent_coef": settings.ent_coef,
        "verbose": 1,
        "tensorboard_log": None,
    }
    if resume_from is None:
        model = PPO(settings.policy, env, device=settings.device, **ppo_parameters)
    else:
        model = PPO.load(
            resume_from,
            env=env,
            device=settings.device,
            **ppo_parameters,
        )
    model.learn(
        total_timesteps=settings.total_timesteps,
        reset_num_timesteps=resume_from is None,
        progress_bar=False,
    )
    model.save(settings.model_path)
    run_settings = {
        "ppo": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
        "resume_from": str(resume_from) if resume_from is not None else None,
        "traffic": asdict(environment_settings),
    }
    (settings.log_dir / "run_config.json").write_text(json.dumps(run_settings, indent=2) + "\n")
    env.close()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="existing PPO model to continue training; timesteps are additional",
    )
    args = parser.parse_args()
    if args.algorithm == "ppo":
        timesteps = PPOConfig.total_timesteps if args.timesteps is None else args.timesteps
        model_path = PPOConfig.model_path if args.model_path is None else args.model_path
        train_ppo(
            PPOConfig(total_timesteps=timesteps, model_path=model_path),
            resume_from=args.resume_from,
        )
        print(f"Saved PPO model to {model_path}.zip")
    else:
        if args.resume_from is not None:
            parser.error("--resume-from is supported only with --algorithm ppo")
        timesteps = DQNConfig.total_timesteps if args.timesteps is None else args.timesteps
        model_path = DQNConfig.model_path if args.model_path is None else args.model_path
        train_dqn(DQNConfig(total_timesteps=timesteps, model_path=model_path))
        print(f"Saved DQN model to {model_path}.zip")


if __name__ == "__main__":
    main()
