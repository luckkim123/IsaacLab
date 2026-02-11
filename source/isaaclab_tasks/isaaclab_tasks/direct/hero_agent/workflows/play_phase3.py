# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 evaluation and deployment export for Hero Agent Encoder-TDC.

Loads a Phase 2 adaptation checkpoint (adapt_tconv + frozen actor) and runs
inference in simulation. Optionally exports the model as JIT/ONNX and the
TDC config as JSON for real robot deployment.

Usage:
    ./isaaclab.sh -p scripts/hero_agent/play_phase3.py \
        --task Isaac-HeroAgent-Adapt-TDC-v0 \
        --checkpoint logs/rsl_rl/hero_agent_adapt_tdc/<run>/model_final.pt \
        --num_envs 16

    # With export:
    ./isaaclab.sh -p scripts/hero_agent/play_phase3.py \
        --task Isaac-HeroAgent-Adapt-TDC-v0 \
        --checkpoint <phase2_model.pt> \
        --export-jit --export-onnx --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3 evaluation and deploy export for Hero Agent Encoder-TDC.")
parser.add_argument("--task", type=str, default="Isaac-HeroAgent-Adapt-TDC-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to Phase 2 model checkpoint.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run at real-time speed.")
parser.add_argument("--export-jit", action="store_true", default=False, help="Export TorchScript model.")
parser.add_argument("--export-onnx", action="store_true", default=False, help="Export ONNX model.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch

from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.hero_agent.utils.env_utils import connect_encoder_to_env
from isaaclab_tasks.direct.hero_agent.workflows import (
    apply_cli_overrides,
    build_adapt_policy,
    get_proprio_history_shape,
    load_phase2_checkpoint,
    resolve_task_configs,
)


def _export_deploy(env_cfg, policy, hist_normalizer, proprio_history_shape: tuple[int, int]) -> None:
    """Export deployment models (JIT/ONNX) and TDC config JSON."""
    from isaaclab_tasks.direct.hero_agent.deploy import (
        HeroAgentDeployModule,
        export_deploy_config,
        export_deploy_module_jit,
        export_deploy_module_onnx,
    )

    export_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "deployed")
    cfg = env_cfg

    deploy_module = HeroAgentDeployModule(
        adapt_tconv=policy.adapt_tconv,
        actor=policy.actor,
        actor_obs_normalizer=policy.actor_obs_normalizer,
        hist_normalizer=hist_normalizer,
        z_min=policy.z_min,
        kp_min=cfg.kp_range[0] if hasattr(cfg, "kp_range") else 10.0,
        kp_max=cfg.kp_range[1] if hasattr(cfg, "kp_range") else 100.0,
        kd_min=cfg.kd_range[0] if hasattr(cfg, "kd_range") else 2.0,
        kd_max=cfg.kd_range[1] if hasattr(cfg, "kd_range") else 30.0,
    )

    if args_cli.export_jit:
        export_deploy_module_jit(deploy_module, export_dir)

    if args_cli.export_onnx:
        export_deploy_module_onnx(
            deploy_module,
            export_dir,
            policy_obs_dim=cfg.observation_space,
            history_len=proprio_history_shape[0],
            history_dim=proprio_history_shape[1],
        )

    # Always export config alongside model
    if hasattr(cfg, "tdc"):
        export_deploy_config(cfg, cfg.tdc, export_dir)

    print(f"[INFO] Deploy models exported to: {export_dir}")


def main():
    """Phase 3 evaluation and optional deploy export."""
    # --- Config resolution ---
    env_cfg, agent_cfg = resolve_task_configs(args_cli.task)
    device = apply_cli_overrides(env_cfg, args_cli.num_envs, args_cli.seed, args_cli.device)

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Video recording
    if args_cli.video:
        log_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "phase3")
        video_kwargs = {
            "video_folder": log_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Build policy + load Phase 2 checkpoint ---
    obs_dict = env.get_observations()
    num_actions = env.unwrapped.cfg.action_space
    policy = build_adapt_policy(obs_dict, agent_cfg, num_actions)

    proprio_history_shape = get_proprio_history_shape(env_cfg)
    hist_normalizer, _checkpoint = load_phase2_checkpoint(
        policy,
        args_cli.checkpoint,
        device,
        proprio_history_shape,
    )

    # --- Connect encoder policy to env for M_hat transfer ---
    connect_encoder_to_env(env, policy, caller_name="Phase3")

    # --- Optional: Export deploy module ---
    if args_cli.export_jit or args_cli.export_onnx:
        _export_deploy(env_cfg, policy, hist_normalizer, proprio_history_shape)

    # --- Inference loop ---
    dt = env.unwrapped.step_dt
    obs_dict = env.get_observations()
    timestep = 0

    print(f"[INFO] Starting Phase 3 inference (num_envs={args_cli.num_envs}, dt={dt:.4f}s)")

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Normalize proprioception history
            obs_normalized = dict(obs_dict)
            if "proprio_hist" in obs_dict:
                obs_normalized["proprio_hist"] = hist_normalizer(obs_dict["proprio_hist"])

            # Deterministic action from actor
            actions = policy.act_inference(obs_normalized)
            actions = actions.clamp(-1.0, 1.0)

            # Step environment
            obs_dict, _, dones, _ = env.step(actions)

            # Reset hidden states for terminated episodes
            policy.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # Real-time pacing
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
