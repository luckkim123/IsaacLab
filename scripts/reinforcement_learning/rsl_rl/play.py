# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _load_runner_and_policy(env, agent_cfg, resume_path):
    """Create runner, load checkpoint, extract inference policy and nn module."""
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "SACMPCRunner":
        from isaaclab_tasks.direct.hero_agent_mpc.runners import SACMPCRunner

        runner = SACMPCRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    if hasattr(runner, "get_inference_policy"):
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        # SAC-MPC: actor.act_inference accepts TensorDict directly
        runner.actor.eval()
        policy = runner.actor.act_inference

    # extract the neural network module
    if hasattr(runner, "alg"):
        # RSL-RL: try 2.3+ first, then 2.2 fallback
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
    else:
        # SAC-MPC: actor is the policy module
        policy_nn = runner.actor

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    # Skip export for policies that can't be JIT-traced (e.g., MPC with iterative solver)
    if not hasattr(policy_nn, "reset_mpc"):
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    return policy, policy_nn


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy, policy_nn = _load_runner_and_policy(env, agent_cfg, resume_path)

    dt = env.unwrapped.step_dt

    # Wire encoder policy to env for M_hat extraction (one-time setup).
    # The env's _pre_physics_step() will call policy.get_last_z() automatically.
    if hasattr(policy_nn, "get_last_z"):
        raw_env = env.unwrapped
        if hasattr(raw_env, "set_encoder_policy"):
            raw_env.set_encoder_policy(policy_nn)

    # Hero Agent evaluation setup
    raw_env = env.unwrapped
    has_eval = hasattr(raw_env, "get_eval_snapshot")
    eval_interval = 200  # Print eval every N steps
    eval_episode_count = 0
    eval_episode_errors: list[float] = []

    # SAC-MPC: initialize prediction error buffer before first inference call
    if hasattr(policy_nn, "_ensure_pred_error_buf"):
        policy_nn._ensure_pred_error_buf(env.num_envs, env.unwrapped.device)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # SAC-MPC: store dynamics prediction for error feedback (before env.step)
            if hasattr(policy_nn, "store_prediction"):
                policy_nn.store_prediction(obs["mpc_state"], actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # SAC-MPC: update prediction error buffer (after env.step)
            if hasattr(policy_nn, "update_pred_error"):
                policy_nn.update_pred_error(obs["mpc_state"])
            # reset policy states for episodes that have terminated
            if hasattr(policy_nn, "reset_mpc"):
                env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if env_ids.numel() > 0:
                    policy_nn.reset_mpc(env_ids)
            elif hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
        timestep += 1

        # Hero Agent: collect episode-end errors and print periodic eval
        if has_eval:
            if dones.any():
                err_deg = torch.rad2deg(torch.linalg.norm(raw_env._attitude_error[dones.squeeze(-1), :2], dim=-1))
                eval_episode_errors.extend(err_deg.tolist())
                eval_episode_count += dones.sum().item()

            if timestep % eval_interval == 0:
                snap = raw_env.get_eval_snapshot()
                ep_info = ""
                if eval_episode_errors:
                    import statistics

                    ep_info = (
                        f" | ep_err={statistics.mean(eval_episode_errors):.1f}"
                        f"+/-{statistics.stdev(eval_episode_errors) if len(eval_episode_errors) > 1 else 0:.1f}deg"
                        f" ({eval_episode_count} eps)"
                    )
                print(
                    f"[Eval @{timestep:5d}] "
                    f"err={snap['attitude_error_deg']:5.1f}deg "
                    f"act={snap['action_magnitude']:.3f} "
                    f"rate={snap['action_rate']:.4f} "
                    f"angvel={snap['angular_velocity_rms']:.3f}"
                    f"{ep_info}"
                )

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Hero Agent: print final evaluation summary
    if has_eval and eval_episode_errors:
        import statistics

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Total steps:    {timestep}")
        print(f"  Episodes:       {eval_episode_count}")
        print(
            f"  Attitude error: {statistics.mean(eval_episode_errors):.1f} +/- "
            f"{statistics.stdev(eval_episode_errors) if len(eval_episode_errors) > 1 else 0:.1f} deg"
        )
        snap = raw_env.get_eval_snapshot()
        print(f"  Action mag:     {snap['action_magnitude']:.4f}")
        print(f"  Action rate:    {snap['action_rate']:.5f}")
        print(f"  Ang vel RMS:    {snap['angular_velocity_rms']:.4f}")
        print("=" * 60)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
