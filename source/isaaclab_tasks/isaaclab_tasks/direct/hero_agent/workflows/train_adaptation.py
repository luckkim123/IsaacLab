# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptation module training for Hero Agent Encoder-TDC.

Trains the adaptation module (ProprioAdaptTConv) to estimate the encoder
latent z from proprioception history only, using supervised L2 loss against
the frozen encoder output.

Usage:
    ./isaaclab.sh -p scripts/hero_agent/train_adaptation.py \
        --task Isaac-HeroAgent-Adapt-TDC-v0 \
        --phase1_checkpoint logs/rsl_rl/hero_agent_encoder_tdc/<run>/model_600.pt \
        --num_envs 4096

Optional (overrides HeroAgentAdaptTDCRunnerCfg defaults):
    --adapt_lr 3e-4           Learning rate for adaptation module
    --max_agent_steps 1e8     Total env steps for training
    --save_interval 10000000  Save checkpoint every N agent steps
    --log_interval 10         Log metrics every N iterations
    --headless                Run without GUI
    --logger wandb            Use WandB for logging
    --log_project_name hero_agent_adaptation
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Adaptation module training for Hero Agent Encoder-TDC.")
parser.add_argument("--task", type=str, default="Isaac-HeroAgent-Adapt-TDC-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (default: from env_cfg).")
parser.add_argument("--phase1_checkpoint", type=str, required=True, help="Path to Phase 1 model checkpoint.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="Override cfg max_agent_steps.")
parser.add_argument("--adapt_lr", type=float, default=None, help="Override cfg adapt_lr.")
parser.add_argument("--save_interval", type=int, default=None, help="Override cfg save_interval_steps.")
parser.add_argument("--log_interval", type=int, default=None, help="Override cfg log_interval.")
parser.add_argument("--seed", type=int, default=None, help="Override cfg seed.")
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="Logger.")
parser.add_argument("--log_project_name", type=str, default=None, help="WandB project name.")
parser.add_argument("--resume", type=str, default=None, help="Path to Phase 2 checkpoint to resume from.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.hero_agent.runners import AdaptRunner
from isaaclab_tasks.direct.hero_agent.workflows import (
    apply_cli_overrides,
    build_adapt_policy,
    get_proprio_history_shape,
    load_phase1_checkpoint,
    resolve_task_configs,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Phase 2 adaptation training."""
    # --- Config resolution ---
    env_cfg, agent_cfg = resolve_task_configs(args_cli.task)

    # --- CLI overrides (only override if explicitly provided) ---
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    device = apply_cli_overrides(env_cfg, num_envs, seed, args_cli.device)

    if args_cli.adapt_lr is not None:
        agent_cfg.adapt_lr = args_cli.adapt_lr
    if args_cli.max_agent_steps is not None:
        agent_cfg.max_agent_steps = args_cli.max_agent_steps
    if args_cli.save_interval is not None:
        agent_cfg.save_interval_steps = args_cli.save_interval
    if args_cli.log_interval is not None:
        agent_cfg.log_interval = args_cli.log_interval

    # --- Logging directory ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Log directory: {log_dir}")

    # --- WandB setup (before env creation) ---
    if args_cli.logger == "wandb":
        try:
            import wandb

            project_name = args_cli.log_project_name or agent_cfg.experiment_name
            wandb.init(
                project=project_name,
                dir=log_dir,
                config={
                    "task": args_cli.task,
                    "phase1_checkpoint": args_cli.phase1_checkpoint,
                    "num_envs": num_envs,
                    "max_agent_steps": agent_cfg.max_agent_steps,
                    "adapt_lr": agent_cfg.adapt_lr,
                    "save_interval_steps": agent_cfg.save_interval_steps,
                    "log_interval": agent_cfg.log_interval,
                    "seed": seed,
                },
            )
        except ImportError:
            print("[WARN] wandb not available, falling back to tensorboard")
            args_cli.logger = "tensorboard"

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Create policy + load Phase 1 ---
    obs_dict = env.get_observations()
    num_actions = env.unwrapped.cfg.action_space
    policy = build_adapt_policy(obs_dict, agent_cfg, num_actions)
    load_phase1_checkpoint(policy, args_cli.phase1_checkpoint, device)

    # --- Create AdaptRunner ---
    proprio_history_shape = get_proprio_history_shape(env_cfg)

    runner = AdaptRunner(
        env=env,
        policy=policy,
        runner_cfg=agent_cfg,
        device=device,
        log_dir=log_dir,
        proprio_history_shape=proprio_history_shape,
        logger_type=args_cli.logger,
    )

    # --- Resume from Phase 2 checkpoint if provided ---
    if args_cli.resume is not None:
        runner.load(args_cli.resume)

    # --- Save configs ---
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # --- Train ---
    runner.learn()

    # --- Cleanup ---
    env.close()
    if args_cli.logger == "wandb":
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
    simulation_app.close()
