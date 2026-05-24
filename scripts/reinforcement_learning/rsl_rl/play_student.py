#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Play a student policy (student encoder + frozen teacher actor).

Mirrors play.py's structure but swaps the policy for StudentInLoopPolicy.
HardDR is forced (teacher's training DR); DORAEMON is disabled. play_mode
fixes zero commands for hovering.

Usage (GPU1, livestream):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_student.py \
        --task Isaac-FullDOF-TRPO-v0 --num_envs 4 --livestream 2 \
        --teacher_ckpt logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/model_4999.pt \
        --student_ckpt logs/rsl_rl/student_policy/<run>/models/student_999.pt \
        --encoder_type tcn
"""
import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a student policy with livestream.")
parser.add_argument("--task", type=str, default="Isaac-FullDOF-TRPO-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--teacher_ckpt", type=str, required=True)
parser.add_argument("--student_ckpt", type=str, required=True)
parser.add_argument("--encoder_type", type=str, choices=["tcn", "gru"], required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hover_cascade", action=argparse.BooleanOptionalAction, default=True,
    help="Outer-loop position-hold: xyz=0, yaw=0 PID drives vel_cmd / yaw_rate_cmd.")
parser.add_argument("--kp_pos", type=float, default=0.5)
parser.add_argument("--kp_yaw", type=float, default=0.5)
parser.add_argument("--vel_sat", type=float, default=0.25)
parser.add_argument("--yaw_rate_sat", type=float, default=0.25)
parser.add_argument("--play_init_attitude_deg", type=float, default=None)
parser.add_argument("--play_init_yaw_deg", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""
import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.constrained_full_albc.config import HardDomainRandomizationCfg
from isaaclab_tasks.direct.constrained_full_albc.student.eval import build_student_policy_fn


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Force HardDR (teacher's training distribution) and disable DORAEMON curriculum.
    env_cfg.randomization = HardDomainRandomizationCfg()
    if hasattr(env_cfg, "doraemon") and env_cfg.doraemon is not None:
        env_cfg.doraemon.enable = False

    # play_mode: fixed zero commands, fixed target attitude, no cmd resampling.
    if hasattr(env_cfg, "play_mode"):
        env_cfg.play_mode = True
        if args_cli.play_init_attitude_deg is not None and hasattr(env_cfg, "play_init_attitude_noise_deg"):
            env_cfg.play_init_attitude_noise_deg = args_cli.play_init_attitude_deg
        if args_cli.play_init_yaw_deg is not None and hasattr(env_cfg, "play_init_yaw_noise_deg"):
            env_cfg.play_init_yaw_noise_deg = args_cli.play_init_yaw_deg
    if hasattr(env_cfg, "randomize_target_attitude"):
        env_cfg.randomize_target_attitude = False
        env_cfg.target_attitude = (0.0, 0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped

    device = raw_env.device
    print(f"[PLAY_STUDENT] device={device}, num_envs={raw_env.num_envs}, encoder={args_cli.encoder_type}", flush=True)

    policy = build_student_policy_fn(
        teacher_ckpt=args_cli.teacher_ckpt,
        student_ckpt=args_cli.student_ckpt,
        encoder_type=args_cli.encoder_type,
        num_envs=raw_env.num_envs,
        device=str(device),
    )

    obs_td, _ = env.reset()

    use_cascade = args_cli.hover_cascade and hasattr(raw_env, "_vel_cmd_lin") and hasattr(raw_env, "_ang_cmd")
    if use_cascade:
        _env_origins = raw_env.scene.env_origins
        print(f"[PLAY_STUDENT] Cascade PID: target xyz=0 yaw=0, Kp_pos={args_cli.kp_pos}, Kp_yaw={args_cli.kp_yaw}",
              flush=True)

    step = 0
    log_interval = 200
    with torch.inference_mode():
        while simulation_app.is_running():
            if use_cascade:
                pos_w = raw_env._robot.data.root_pos_w
                quat_w = raw_env._robot.data.root_quat_w
                pos_err_w = _env_origins - pos_w
                pos_err_b = quat_rotate_inverse(quat_w, pos_err_w)
                vel_cmd = torch.clamp(args_cli.kp_pos * pos_err_b, -args_cli.vel_sat, args_cli.vel_sat)
                raw_env._vel_cmd_lin[:, 0] = vel_cmd[:, 0]
                raw_env._vel_cmd_lin[:, 1] = vel_cmd[:, 1]
                raw_env._vel_cmd_lin[:, 2] = vel_cmd[:, 2]
                _, _, yaw_w = euler_xyz_from_quat(quat_w)
                yaw_err = torch.atan2(torch.sin(-yaw_w), torch.cos(-yaw_w))
                yaw_rate_cmd = torch.clamp(args_cli.kp_yaw * yaw_err, -args_cli.yaw_rate_sat, args_cli.yaw_rate_sat)
                raw_env._ang_cmd[:, 0] = 0.0
                raw_env._ang_cmd[:, 1] = 0.0
                raw_env._ang_cmd[:, 2] = yaw_rate_cmd

            actions = policy(obs_td)
            obs_td, _, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated

            if dones.any():
                policy.reset(dones)

            step += 1
            if step % log_interval == 0 and hasattr(raw_env, "get_eval_snapshot"):
                snap = raw_env.get_eval_snapshot()
                print(f"[step {step}] att={snap['attitude_error_deg']:.2f}° "
                      f"lv_err={snap['lin_vel_error']:.3f} thr_util={snap['thruster_utilization']:.2f}",
                      flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
