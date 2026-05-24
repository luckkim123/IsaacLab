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
parser.add_argument("--hover_cascade", action=argparse.BooleanOptionalAction, default=True,
    help="Use cascade position-hold controller: outer loop keeps xyz=0, yaw=0; policy sees vel_cmd + yaw_rate_cmd.")
parser.add_argument("--kp_pos", type=float, default=0.5, help="Outer-loop position P-gain (s^-1).")
parser.add_argument("--kp_yaw", type=float, default=0.5, help="Outer-loop yaw P-gain (s^-1).")
parser.add_argument("--vel_sat", type=float, default=0.25, help="Velocity command saturation (m/s).")
parser.add_argument("--yaw_rate_sat", type=float, default=0.25, help="Yaw rate command saturation (rad/s).")
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
parser.add_argument(
    "--dr_scale",
    type=float,
    default=None,
    help="FullDOF-TRPO only: DR scale 0.0 (none) .. 1.0 (hard). e.g. 0.6=medium. Uses eval_dr_fulldof.build_dr_config.",
)
parser.add_argument(
    "--play_init_attitude_deg",
    type=float,
    default=None,
    help="FullDOF-TRPO only: uniform roll/pitch init noise in degrees (override env_cfg.play_init_attitude_noise_deg).",
)
parser.add_argument(
    "--play_init_yaw_deg",
    type=float,
    default=None,
    help="FullDOF-TRPO only: uniform yaw init noise in degrees (override env_cfg.play_init_yaw_noise_deg).",
)
parser.add_argument(
    "--show_payload_viz",
    action="store_true",
    default=False,
    help="FullDOF-TRPO only: render payload CoG sphere (mass-scaled) and attachment->CoG bar.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Override scene.env_spacing (meters). Smaller = envs packed closer.",
)
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

import importlib
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat

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
    # Runner dispatch: custom runners resolved by class_name via lazy import.
    _RUNNER_MAP = {
        "FullDOFConstraintEncoderRunner": (
            "isaaclab_tasks.direct.constrained_full_albc.runners",
            "ConstraintEncoderRunner",
        ),
        "OnPolicyDoraemonRunner": (
            "isaaclab_tasks.direct.constrained_full_albc.runners",
            "OnPolicyDoraemonRunner",
        ),
    }

    if agent_cfg.class_name in _RUNNER_MAP:
        module_path, cls_name = _RUNNER_MAP[agent_cfg.class_name]
        module = importlib.import_module(module_path)
        runner_cls = getattr(module, cls_name)
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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
    elif hasattr(runner, "policy"):
        # AdaptRunner: policy is the nn.Module directly
        policy_nn = runner.policy
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
    # Skip for:
    #   - MPC policies (iterative solver, not traceable)
    #   - encoder policies (internal normalization of o_t + concat z; external normalizer wrap breaks shape)
    is_encoder = hasattr(policy_nn, "encoder_latent_dim") or hasattr(policy_nn, "encoder")
    can_export = not hasattr(policy_nn, "reset_mpc") and not is_encoder
    if can_export:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    else:
        print(f"[PLAY] Skipping JIT/ONNX export (encoder={is_encoder}, mpc={hasattr(policy_nn, 'reset_mpc')}).")

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
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = args_cli.env_spacing
        print(f"[PLAY] env_spacing override = {args_cli.env_spacing}", flush=True)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Play always evaluates at fixed target pose (0, 0, 0)
    if hasattr(env_cfg, "randomize_target_attitude"):
        env_cfg.randomize_target_attitude = False
        env_cfg.target_attitude = (0.0, 0.0, 0.0)

    # FullDOF-TRPO: enable play_mode (fixed zero commands for hovering eval)
    print("[PLAY] entering main() post-startup", flush=True)
    if hasattr(env_cfg, "play_mode"):
        env_cfg.play_mode = True
        # Optional init-attitude overrides for "recovery from random tilt" demo
        if args_cli.play_init_attitude_deg is not None and hasattr(env_cfg, "play_init_attitude_noise_deg"):
            env_cfg.play_init_attitude_noise_deg = args_cli.play_init_attitude_deg
        if args_cli.play_init_yaw_deg is not None and hasattr(env_cfg, "play_init_yaw_noise_deg"):
            env_cfg.play_init_yaw_noise_deg = args_cli.play_init_yaw_deg
    print(f"[PLAY] play_mode set. env_cfg.num_envs={env_cfg.scene.num_envs}", flush=True)

    if args_cli.show_payload_viz and hasattr(env_cfg, "enable_payload_viz"):
        env_cfg.enable_payload_viz = True
        print("[PLAY] payload CoG visualization enabled.", flush=True)

    # FullDOF-TRPO: optional DR scale override (none=0 .. hard=1).
    # Interpolate between true nominal (scale=0) and HardDR (scale=1).
    # Mirrors scripts/analysis/eval_dr_fulldof.build_dr_config (kept inline to
    # avoid re-executing AppLauncher on import).
    if args_cli.dr_scale is not None and hasattr(env_cfg, "randomization"):
        from isaaclab_tasks.direct.constrained_full_albc.config import (
            DomainRandomizationCfg,
            HardDomainRandomizationCfg,
        )

        _DR_TUPLE_FIELDS = [
            "added_mass_scale", "linear_damping_scale", "quadratic_damping_scale", "volume_scale",
            "cob_offset_x", "cob_offset_y", "cob_offset_z",
            "cog_offset_x", "cog_offset_y", "cog_offset_z",
            "inertia_scale", "body_mass_scale",
            "water_density_range", "joint_stiffness_range", "joint_damping_range",
            "yaw_damping_scale", "joint_effort_limit_range",
            "joint_static_friction_range", "joint_viscous_friction_range",
            "payload_mass_range", "payload_cog_offset_z",
            "thrust_coefficient_scale", "time_constant_scale",
            "ocean_current_strength_range",
        ]
        _DR_FLOAT_FIELDS = ["payload_cog_offset_xy_radius", "buoy_moment_arm"]
        _NOMINAL = {
            "added_mass_scale": 1.0, "linear_damping_scale": 1.0, "quadratic_damping_scale": 1.0,
            "volume_scale": 1.0, "inertia_scale": 1.0, "body_mass_scale": 1.0, "yaw_damping_scale": 1.0,
            "joint_effort_limit_range": 1.0, "thrust_coefficient_scale": 1.0, "time_constant_scale": 1.0,
            "cob_offset_x": 0.0, "cob_offset_y": 0.0, "cob_offset_z": 0.0,
            "cog_offset_x": 0.0, "cog_offset_y": 0.0, "cog_offset_z": 0.0,
            "water_density_range": 1000.0, "payload_mass_range": 0.0, "payload_cog_offset_z": 0.0,
            "joint_static_friction_range": 0.0, "joint_viscous_friction_range": 0.0,
            "payload_cog_offset_xy_radius": 0.0, "ocean_current_strength_range": 0.0,
        }

        base = DomainRandomizationCfg()
        nominal = DomainRandomizationCfg()
        for f in _DR_TUPLE_FIELDS:
            v = _NOMINAL.get(f)
            if v is not None:
                setattr(nominal, f, (v, v))
            else:
                lo, hi = getattr(base, f)
                setattr(nominal, f, ((lo + hi) / 2.0, (lo + hi) / 2.0))
        for f in _DR_FLOAT_FIELDS:
            if f in _NOMINAL:
                setattr(nominal, f, _NOMINAL[f])

        scale = float(args_cli.dr_scale)
        if scale <= 0.0:
            scaled = nominal
        else:
            full = HardDomainRandomizationCfg()
            frac = min(scale, 1.0)
            scaled = DomainRandomizationCfg()
            for f in _DR_TUPLE_FIELDS:
                nlo, nhi = getattr(nominal, f)
                flo, fhi = getattr(full, f)
                setattr(scaled, f, (nlo + frac * (flo - nlo), nhi + frac * (fhi - nhi)))
            for f in _DR_FLOAT_FIELDS:
                setattr(scaled, f, getattr(nominal, f) + frac * (getattr(full, f) - getattr(nominal, f)))
        scaled.enable = True
        env_cfg.randomization = scaled
        if hasattr(env_cfg, "doraemon") and env_cfg.doraemon is not None:
            env_cfg.doraemon.enable = False
        print(f"[PLAY] DR scale = {scale} applied; DORAEMON disabled.", flush=True)

    print("[PLAY] creating environment...", flush=True)
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
        # Prefer best_model.pt when no checkpoint is explicitly specified
        best_model_path = os.path.join(os.path.dirname(resume_path), "best_model.pt")
        if os.path.isfile(best_model_path):
            resume_path = best_model_path

    log_dir = os.path.dirname(resume_path)

    # Load run's agent.yaml and override policy fields that affect network shape
    # (e.g., encoder_latent_dim differs between r13_A=9 and r13_B=16). Without this
    # the default repo config is used -> state_dict size mismatch at checkpoint load.
    run_params_path = os.path.join(log_dir, "params", "agent.yaml")
    if os.path.isfile(run_params_path):
        try:
            import yaml
            with open(run_params_path) as f:
                run_dict = yaml.full_load(f)
            pol = run_dict.get("policy", {})
            for k in ["encoder_latent_dim", "encoder_hidden_dims",
                      "actor_hidden_dims", "critic_hidden_dims",
                      "cost_critic_hidden_dims", "activation", "init_noise_std"]:
                if k in pol and hasattr(agent_cfg.policy, k):
                    setattr(agent_cfg.policy, k, pol[k])
            print(f"[PLAY] Applied policy overrides from {run_params_path}: encoder_latent_dim="
                  f"{getattr(agent_cfg.policy, 'encoder_latent_dim', '?')}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not apply run policy overrides: {e}", flush=True)

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

    # Evaluation setup (works for both Hero Agent and FullDOF-TRPO environments)
    raw_env = env.unwrapped
    has_eval = hasattr(raw_env, "get_eval_snapshot")
    eval_interval = 200  # Print eval every N steps
    eval_episode_count = 0
    eval_episode_errors: list[float] = []
    # Detect attitude error attribute: hero_agent uses _attitude_error, ALBCEnv uses _att_rp_err
    _att_err_attr = (
        "_attitude_error" if hasattr(raw_env, "_attitude_error") else "_att_rp_err" if hasattr(raw_env, "_att_rp_err") else None
    )

    # SAC-MPC: initialize prediction error buffer before first inference call
    if hasattr(policy_nn, "_ensure_pred_error_buf"):
        policy_nn._ensure_pred_error_buf(env.num_envs, env.unwrapped.device)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # Cascade-PID setup (FullDOF-TRPO only — requires vel_cmd/ang_cmd interface)
    use_cascade = args_cli.hover_cascade and hasattr(raw_env, "_vel_cmd_lin") and hasattr(raw_env, "_ang_cmd")
    if use_cascade:
        _env_origins = raw_env.scene.env_origins  # (N, 3) world frame
        print(f"[PLAY] Cascade PID: target xyz=0, yaw=0. Kp_pos={args_cli.kp_pos}, Kp_yaw={args_cli.kp_yaw}, "
              f"vel_sat=±{args_cli.vel_sat}, yaw_rate_sat=±{args_cli.yaw_rate_sat}", flush=True)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Cascade PID outer loop: position/yaw error → vel_cmd / yaw_rate_cmd
            if use_cascade:
                pos_w = raw_env._robot.data.root_pos_w
                quat_w = raw_env._robot.data.root_quat_w
                pos_err_w = _env_origins - pos_w  # target(0) - actual = drive back to origin
                pos_err_b = quat_rotate_inverse(quat_w, pos_err_w)
                vel_cmd = torch.clamp(args_cli.kp_pos * pos_err_b, -args_cli.vel_sat, args_cli.vel_sat)
                raw_env._vel_cmd_lin[:, 0] = vel_cmd[:, 0]
                raw_env._vel_cmd_lin[:, 1] = vel_cmd[:, 1]
                raw_env._vel_cmd_lin[:, 2] = vel_cmd[:, 2]
                _, _, yaw_w = euler_xyz_from_quat(quat_w)
                yaw_err = torch.atan2(torch.sin(-yaw_w), torch.cos(-yaw_w))  # wrap (0 - yaw)
                yaw_rate_cmd = torch.clamp(args_cli.kp_yaw * yaw_err, -args_cli.yaw_rate_sat, args_cli.yaw_rate_sat)
                raw_env._ang_cmd[:, 0] = 0.0
                raw_env._ang_cmd[:, 1] = 0.0
                raw_env._ang_cmd[:, 2] = yaw_rate_cmd
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

        # Collect episode-end errors and print periodic eval
        if has_eval:
            done_mask = dones.squeeze(-1).bool()
            if done_mask.any() and _att_err_attr is not None:
                att_err = getattr(raw_env, _att_err_attr)
                err_deg = torch.rad2deg(torch.linalg.norm(att_err[done_mask, :2], dim=-1))
                eval_episode_errors.extend(err_deg.tolist())
                eval_episode_count += done_mask.sum().item()

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
                # Build output from snapshot keys (adapts to any env)
                parts = [f"[Eval @{timestep:5d}]"]
                parts.append(f"err={snap['attitude_error_deg']:5.1f}deg")
                if "lin_vel_error" in snap:
                    parts.append(f"lin_err={snap['lin_vel_error']:.3f}")
                parts.append(f"rate={snap['action_rate']:.4f}")
                parts.append(f"angvel_rp={snap['angular_velocity_rp_rms']:.3f}")
                parts.append(f"angvel_yaw={snap['angular_velocity_yaw_rms']:.3f}")
                if "joint_oscillation_hf_rms" in snap:
                    parts.append(f"jt_osc={snap['joint_oscillation_hf_rms']:.4f}")
                if "thruster_utilization" in snap:
                    parts.append(f"thr={snap['thruster_utilization']:.3f}")
                parts.append(f"jt_pos={snap['joint_pos_mean_abs']:.2f}")
                print(" ".join(parts) + ep_info)

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Print final evaluation summary
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
        print(f"  Action rate:    {snap['action_rate']:.5f}")
        print(f"  Ang vel RP:     {snap['angular_velocity_rp_rms']:.4f}")
        print(f"  Ang vel Yaw:    {snap['angular_velocity_yaw_rms']:.4f}")
        if "lin_vel_error" in snap:
            print(f"  Lin vel err:    {snap['lin_vel_error']:.4f}")
        if "joint_oscillation_hf_rms" in snap:
            print(f"  Joint osc HF:   {snap['joint_oscillation_hf_rms']:.5f}")
        if "thruster_utilization" in snap:
            print(f"  Thruster util:  {snap['thruster_utilization']:.4f}")
        print(f"  Joint pos abs:  {snap['joint_pos_mean_abs']:.3f}")
        print("=" * 60)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
