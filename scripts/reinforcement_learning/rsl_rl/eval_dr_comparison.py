# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate policy robustness across Domain Randomization levels.

All levels use identical +-15 deg step-change tracking for fair comparison.
DR parameters are linearly scaled from 0% (none) to 100% (hard = training DR):
    none   -> 0%   of training DR (nominal physics)
    soft   -> 30%  of training DR
    medium -> 60%  of training DR
    hard   -> 100% of training DR (matches DomainRandomizationCfg defaults)

All levels start from 0 deg initial pose. Each segment is held for 5 s
(configurable via --segment_duration).

Supported tasks:
    Isaac-HeroAgent-v0              (debug, no DR)
    Isaac-HeroAgent-Base-v0         (base RL training)
    Isaac-HeroAgent-Encoder-Base-v0 (HORA Phase 1)
    Isaac-HeroAgent-TDC-v0          (classical TDC)
    Isaac-HeroAgent-Adapt-Base-v0   (HORA Phase 2)
    Isaac-HeroAgent-SAC-MPC-v0      (SAC-MPC)

Usage:
    # Pure TDC baseline
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py \
        --task Isaac-HeroAgent-TDC-v0 --checkpoint none --num_envs 16 --headless

    # Encoder-Base policy
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py \
        --task Isaac-HeroAgent-Encoder-Base-v0 --num_envs 64 --headless

    # SAC-MPC policy
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py \
        --task Isaac-HeroAgent-SAC-MPC-v0 --num_envs 64 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ---- CLI arguments ----
parser = argparse.ArgumentParser(description="Evaluate DR robustness of RL / TDC policies.")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g. Isaac-HeroAgent-Encoder-Base-v0)")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: logs/eval_dr/<task>/<ts>)")
parser.add_argument("--segment_duration", type=float, default=5.0, help="Duration per segment in seconds (default 5).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL config entry point."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch
import rsl_rl.runners.on_policy_runner as _runner_module
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.hero_agent.config import DomainRandomizationCfg
from isaaclab_tasks.direct.hero_agent.utils import unwrap_env, connect_encoder_to_env
from isaaclab_tasks.direct.hero_agent.encoder import ActorCriticEncoder, ActorCriticEncoderAdapt
from isaaclab_tasks.direct.hero_agent.runners import BaseRunner, EncoderRunner

# Register custom classes in RSL-RL runner module namespace so the runner
# can resolve class_name strings (same pattern as rsl_rl_ppo_cfg.py).
_runner_module.ActorCriticEncoder = ActorCriticEncoder
_runner_module.ActorCriticEncoderAdapt = ActorCriticEncoderAdapt
_runner_module.BaseRunner = BaseRunner
_runner_module.EncoderRunner = EncoderRunner

matplotlib.use("Agg")  # non-interactive backend for headless

# ---- Configuration ----
DR_LEVELS = ["none", "soft", "medium", "hard"]
DR_COLORS = {"none": "#2196F3", "soft": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}
DR_ANGLES = {"none": 15.0, "soft": 15.0, "medium": 15.0, "hard": 15.0}

# DR scale factors: 0 = nominal physics, 1 = full training DR
DR_SCALE = {"none": 0.0, "soft": 0.3, "medium": 0.6, "hard": 1.0}


# ============================================================================
# DR Configuration
# ============================================================================

def build_dr_config(scale: float, is_tdc: bool) -> DomainRandomizationCfg:
    """Build a DomainRandomizationCfg by interpolating between nominal and full DR.

    Args:
        scale: 0.0 = nominal physics (fixed_pose), 1.0 = full training DR.
        is_tdc: If True, use TDC-specific joint gains and disable action latency.

    Returns:
        A DomainRandomizationCfg with interpolated values.
    """
    if scale <= 0.0:
        cfg = DomainRandomizationCfg.fixed_pose()
        cfg.enable = True
        if is_tdc:
            cfg.joint_stiffness_range = (160.0, 240.0)
            cfg.joint_damping_range = (8.0, 12.0)
        return cfg

    nominal = DomainRandomizationCfg.fixed_pose()
    full = DomainRandomizationCfg()
    f = min(scale, 1.0)

    # Fields to interpolate: (field_name, type)
    # tuple[float,float] fields lerp element-wise
    float_tuple_fields = [
        "position_x_range", "position_y_range", "position_z_range",
        "yaw_range",
        "inertia_scale", "body_mass_scale", "volume_scale",
        "added_mass_scale", "linear_damping_scale", "quadratic_damping_scale",
        "water_density_range",
        "perturbation_force_range", "perturbation_torque_range",
        "payload_mass_range", "payload_cog_offset_z",
        "cob_offset_x", "cob_offset_y", "cob_offset_z",
        "cog_offset_x", "cog_offset_y", "cog_offset_z",
        "joint_stiffness_range", "joint_damping_range",
        "yaw_damping_scale",
        "joint_effort_limit_range",
        "joint_static_friction_range", "joint_viscous_friction_range",
    ]
    # tuple[int,int] fields lerp and round
    int_tuple_fields = ["action_latency_range"]
    # float fields lerp directly
    float_fields = ["payload_cog_offset_xy_radius"]

    cfg = DomainRandomizationCfg()
    cfg.enable = True

    for field in float_tuple_fields:
        nom_val = getattr(nominal, field)
        full_val = getattr(full, field)
        lo = nom_val[0] + f * (full_val[0] - nom_val[0])
        hi = nom_val[1] + f * (full_val[1] - nom_val[1])
        setattr(cfg, field, (lo, hi))

    for field in int_tuple_fields:
        nom_val = getattr(nominal, field)
        full_val = getattr(full, field)
        lo = int(round(nom_val[0] + f * (full_val[0] - nom_val[0])))
        hi = int(round(nom_val[1] + f * (full_val[1] - nom_val[1])))
        setattr(cfg, field, (lo, hi))

    for field in float_fields:
        nom_val = getattr(nominal, field)
        full_val = getattr(full, field)
        setattr(cfg, field, nom_val + f * (full_val - nom_val))

    cfg.enable_perturbation = f > 0

    # All levels: initial pose at (0, 0), free yaw
    cfg.roll_range = (0.0, 0.0)
    cfg.pitch_range = (0.0, 0.0)

    # TDC-specific overrides
    if is_tdc:
        cfg.joint_stiffness_range = (160.0, 240.0)
        cfg.joint_damping_range = (8.0, 12.0)
        cfg.action_latency_range = (0, 0)

    return cfg


def apply_dr_config(env_cfg, scale: float, is_tdc: bool) -> None:
    """Apply interpolated DR config to the environment config."""
    env_cfg.randomization = build_dr_config(scale, is_tdc)


# ============================================================================
# Trajectory
# ============================================================================

def _interpolate_waypoints(
    waypoints: list[tuple[float, float, str]],
    increment: float,
) -> list[tuple[float, float, str]]:
    """Expand waypoints so each axis moves by exactly *increment* deg per step.

    Each axis advances independently by *increment* and clamps at the target,
    so all intermediate values are clean multiples of *increment*.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    result: list[tuple[float, float, str]] = [waypoints[0]]
    for prev, cur in zip(waypoints[:-1], waypoints[1:]):
        dr = cur[0] - prev[0]
        dp = cur[1] - prev[1]
        nr = int(np.ceil(abs(dr) / increment)) if dr != 0 else 0
        np_ = int(np.ceil(abs(dp) / increment)) if dp != 0 else 0
        n_steps = max(nr, np_, 1)
        r_sign = np.sign(dr)
        p_sign = np.sign(dp)
        r_cur, p_cur = prev[0], prev[1]
        for k in range(1, n_steps):
            r_cur = _clamp(r_cur + r_sign * increment, prev[0], cur[0])
            p_cur = _clamp(p_cur + p_sign * increment, prev[1], cur[1])
            result.append((r_cur, p_cur, f"({r_cur:+.0f}, {p_cur:+.0f})"))
        result.append(cur)
    return result


def _clamp(val: float, a: float, b: float) -> float:
    """Clamp *val* between *a* and *b* regardless of order."""
    lo, hi = min(a, b), max(a, b)
    return max(lo, min(hi, val))


def build_step_trajectory(
    segment_duration: float,
    step_dt: float,
    max_angle_deg: float,
    increment_deg: float = 15.0,
    target_num_segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build step-change target trajectory with *increment_deg* max per segment.

    Original waypoints (6): neutral, +roll, +pitch, combined-neg, mixed, neutral.
    These are expanded so that consecutive segments differ by at most
    *increment_deg* degrees on each axis.  Each segment is held for
    *segment_duration* seconds.

    If *target_num_segments* is given, shorter trajectories are padded with
    hold-at-zero segments so all levels have equal length.
    """
    a = max_angle_deg
    if a > 0:
        waypoints: list[tuple[float, float, str]] = [
            (0.0, 0.0, "neutral"),
            (a, 0.0, f"roll +{a:.0f}"),
            (0.0, a, f"pitch +{a:.0f}"),
            (-a, -a, f"({-a:.0f}, {-a:.0f})"),
            (a, -a, f"({a:.0f}, {-a:.0f})"),
            (0.0, 0.0, "return neutral"),
        ]
        segments = _interpolate_waypoints(waypoints, increment_deg)
    else:
        segments = [(0.0, 0.0, "hold")]

    # Pad with hold segments to match target length
    if target_num_segments and len(segments) < target_num_segments:
        last_r, last_p = segments[-1][0], segments[-1][1]
        while len(segments) < target_num_segments:
            segments.append((last_r, last_p, f"hold ({last_r:+.0f}, {last_p:+.0f})"))

    steps_per_seg = int(segment_duration / step_dt)
    total_steps = steps_per_seg * len(segments)

    time_s = np.arange(total_steps) * step_dt
    target_roll = np.zeros(total_steps)
    target_pitch = np.zeros(total_steps)
    seg_names = []

    for i, (r, p, name) in enumerate(segments):
        s = i * steps_per_seg
        e = (i + 1) * steps_per_seg
        target_roll[s:e] = r
        target_pitch[s:e] = p
        seg_names.append(name)

    return time_s, target_roll, target_pitch, seg_names


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(data: dict) -> dict:
    """Compute summary metrics from collected data."""
    time_s = data["time"]
    error_roll = data["error_roll"]   # (steps, envs) in degrees
    error_pitch = data["error_pitch"]
    terminated = data["terminated"]
    num_envs = error_roll.shape[1]

    error_norm = np.sqrt(error_roll**2 + error_pitch**2)
    alive = ~terminated

    # Overall mean error (alive envs only)
    if alive.any():
        total_mean_error = float(np.nanmean(np.where(alive, error_norm, np.nan)))
    else:
        total_mean_error = float("nan")

    # Survival rate
    survival_rate = float(alive[-1].sum()) / num_envs * 100.0

    # Per-segment metrics
    seg_steps = data["steps_per_segment"]
    num_segments = len(data["segment_names"])
    steady_state_errors = []
    settling_times = []

    for seg_idx in range(num_segments):
        s = seg_idx * seg_steps
        e = (seg_idx + 1) * seg_steps

        seg_error = error_norm[s:e]
        seg_alive = alive[s:e]
        seg_time = time_s[s:e]

        # Steady-state: last 50% of segment (5s out of 10s)
        ss_start = int(seg_steps * 0.5)
        ss_error = seg_error[ss_start:]
        ss_alive = seg_alive[ss_start:]
        if ss_alive.any():
            steady_state_errors.append(float(np.nanmean(np.where(ss_alive, ss_error, np.nan))))
        else:
            steady_state_errors.append(float("nan"))

        # Settling time: first step where mean error < 5 deg
        mean_per_step = np.nanmean(np.where(seg_alive, seg_error, np.nan), axis=1)
        settled = mean_per_step < 5.0
        if settled.any():
            settling_times.append(float(seg_time[np.argmax(settled)] - seg_time[0]))
        else:
            settling_times.append(float(data["segment_duration"]))

    return {
        "total_mean_error": total_mean_error,
        "survival_rate": survival_rate,
        "steady_state_errors": steady_state_errors,
        "settling_times": settling_times,
    }


# ============================================================================
# Plots
# ============================================================================

def _bar_subplot(ax, x, values, colors, xlabels, ylabel, title, ylim=None):
    """Render a single bar chart subplot with consistent styling."""
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, axis="y")


def generate_plots(
    all_data: dict[str, dict],
    all_metrics: dict[str, dict],
    output_dir: str,
) -> None:
    """Generate 4 comparison figures and save as PNG."""
    levels = [lvl for lvl in DR_LEVELS if lvl in all_data]

    # ---- Figure 1: Per-Level Tracking (4x2 grid) ----
    fig1, axes1 = plt.subplots(len(levels), 2, figsize=(16, 3 * len(levels)), sharex=True)
    fig1.suptitle("Tracking Performance per DR Level", fontsize=14, y=0.98)

    for row, lvl in enumerate(levels):
        d = all_data[lvl]
        color = DR_COLORS[lvl]
        time_s = d["time"]
        alive = ~d["terminated"]
        dr_pct = int(DR_SCALE[lvl] * 100)

        for col, (actual_key, target_key, axis_label) in enumerate([
            ("actual_roll_deg", "target_roll_deg", "Roll (deg)"),
            ("actual_pitch_deg", "target_pitch_deg", "Pitch (deg)"),
        ]):
            ax = axes1[row, col] if len(levels) > 1 else axes1[col]
            ax.plot(time_s, d[target_key], "k--", linewidth=1.2, alpha=0.6, label="target")
            vals = np.where(alive, d[actual_key], np.nan)
            mean = np.nanmean(vals, axis=1)
            std = np.nanstd(vals, axis=1)
            ax.plot(time_s, mean, color=color, linewidth=1.0, label="actual (mean)")
            ax.fill_between(time_s, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_ylabel(axis_label, fontsize=9)
            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_title(f"{lvl} (DR {dr_pct}%)", fontsize=10, fontweight="bold", color=color)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
            if row == len(levels) - 1:
                ax.set_xlabel("Time (s)")

    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "tracking.png"), dpi=150)
    plt.close(fig1)

    # ---- Figure 2: Error Time-Series (2x1, all levels overlaid) ----
    fig2, (ax_re, ax_pe) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle("Tracking Error vs DR Level", fontsize=14)

    for lvl in levels:
        d = all_data[lvl]
        time_s = d["time"]
        color = DR_COLORS[lvl]
        alive = ~d["terminated"]
        dr_pct = int(DR_SCALE[lvl] * 100)
        label = f"{lvl} (DR {dr_pct}%)"

        for ax, key in [(ax_re, "error_roll"), (ax_pe, "error_pitch")]:
            vals = np.where(alive, np.abs(d[key]), np.nan)
            mean = np.nanmean(vals, axis=1)
            std = np.nanstd(vals, axis=1)
            ax.plot(time_s, mean, color=color, linewidth=1.2, label=label)
            ax.fill_between(time_s, mean - std, mean + std, color=color, alpha=0.12)

    ax_re.set_ylabel("|Roll Error| (deg)")
    ax_pe.set_ylabel("|Pitch Error| (deg)")
    ax_pe.set_xlabel("Time (s)")
    ax_re.legend(loc="upper right", fontsize=9)
    for _ax in (ax_re, ax_pe):
        _ax.yaxis.set_major_locator(MultipleLocator(15))
        _ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "error.png"), dpi=150)
    plt.close(fig2)

    # ---- Figure 3: Summary Bar Chart (2x2) ----
    fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig3.suptitle("DR Robustness Summary", fontsize=14)
    x = np.arange(len(levels))
    bar_colors = [DR_COLORS[lvl] for lvl in levels]
    xlabels = [f"{lvl}\n(DR {int(DR_SCALE[lvl] * 100)}%)" for lvl in levels]

    ss_errors = [np.nanmean(all_metrics[lvl]["steady_state_errors"]) for lvl in levels]
    _bar_subplot(axes[0, 0], x, ss_errors, bar_colors, xlabels, "Error (deg)", "Steady-State Error (last 5s avg)")

    settle_times = [np.nanmean(all_metrics[lvl]["settling_times"]) for lvl in levels]
    _bar_subplot(axes[0, 1], x, settle_times, bar_colors, xlabels, "Time (s)", "Settling Time (<5 deg)")

    total_errors = [all_metrics[lvl]["total_mean_error"] for lvl in levels]
    _bar_subplot(axes[1, 0], x, total_errors, bar_colors, xlabels, "Error (deg)", "Total Mean Error")

    survivals = [all_metrics[lvl]["survival_rate"] for lvl in levels]
    _bar_subplot(axes[1, 1], x, survivals, bar_colors, xlabels, "Survival (%)", "Survival Rate", ylim=(0, 105))

    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "summary.png"), dpi=150)
    plt.close(fig3)



# ============================================================================
# Evaluation Loop
# ============================================================================

def run_evaluation(
    env,
    policy,
    policy_nn,
    raw_env,
    time_s: np.ndarray,
    target_roll_deg: np.ndarray,
    target_pitch_deg: np.ndarray,
    segment_names: list[str],
    segment_duration: float,
    step_dt: float,
    num_envs: int,
    device: torch.device,
) -> dict:
    """Run one evaluation pass and collect per-step data."""
    total_steps = len(time_s)
    steps_per_seg = int(segment_duration / step_dt)

    actual_roll = np.zeros((total_steps, num_envs))
    actual_pitch = np.zeros((total_steps, num_envs))
    error_roll = np.zeros((total_steps, num_envs))
    error_pitch = np.zeros((total_steps, num_envs))
    terminated = np.zeros((total_steps, num_envs), dtype=bool)

    # Force full reset via throwaway step
    raw_env.episode_length_buf[:] = raw_env.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
        if hasattr(policy_nn, "reset_mpc"):
            policy_nn.reset_mpc(torch.arange(num_envs, device=device))
        elif hasattr(policy_nn, "reset"):
            policy_nn.reset(torch.ones(num_envs, 1, dtype=torch.bool, device=device))

    # Clear episode length counter to prevent timeout during evaluation.
    # Without this, _reset_framework() jitters episode_length_buf to [0, half_ep),
    # causing ~60% of environments to timeout before the trajectory completes.
    raw_env.episode_length_buf[:] = 0

    target_roll_rad = np.deg2rad(target_roll_deg)
    target_pitch_rad = np.deg2rad(target_pitch_deg)
    terminated_ever = np.zeros(num_envs, dtype=bool)

    for step_idx in range(total_steps):
        # Override target attitude
        raw_env._target_euler[:, 0] = target_roll_rad[step_idx]
        raw_env._target_euler[:, 1] = target_pitch_rad[step_idx]
        raw_env._target_euler[:, 2] = 0.0

        with torch.inference_mode():
            actions = policy(obs)
            # SAC-MPC: store dynamics prediction before env.step
            if hasattr(policy_nn, "store_prediction"):
                policy_nn.store_prediction(obs["mpc_state"], actions)
            obs, _, dones, _ = env.step(actions)
            # SAC-MPC: update prediction error after env.step
            if hasattr(policy_nn, "update_pred_error"):
                policy_nn.update_pred_error(obs["mpc_state"])
            # Reset policy states
            if hasattr(policy_nn, "reset_mpc"):
                env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if env_ids.numel() > 0:
                    policy_nn.reset_mpc(env_ids)
            elif hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        # Collect Euler angles
        roll_cur, pitch_cur, _ = euler_xyz_from_quat(raw_env._robot.data.root_quat_w)
        actual_roll[step_idx] = torch.rad2deg(roll_cur).cpu().numpy()
        actual_pitch[step_idx] = torch.rad2deg(pitch_cur).cpu().numpy()

        # Error (signed, degrees)
        att_err = raw_env._attitude_error[:, :2]
        error_roll[step_idx] = torch.rad2deg(att_err[:, 0]).cpu().numpy()
        error_pitch[step_idx] = torch.rad2deg(att_err[:, 1]).cpu().numpy()

        # Track terminations (cumulative)
        dones_np = dones.squeeze(-1).cpu().numpy().astype(bool) if dones.dim() > 1 else dones.cpu().numpy().astype(bool)
        terminated_ever |= dones_np
        terminated[step_idx] = terminated_ever

        # Progress logging
        if (step_idx + 1) % 1000 == 0 or step_idx == total_steps - 1:
            alive_count = num_envs - terminated_ever.sum()
            err_norm = np.sqrt(error_roll[step_idx] ** 2 + error_pitch[step_idx] ** 2)
            alive_mask = ~terminated_ever
            mean_err = np.mean(err_norm[alive_mask]) if alive_mask.any() else float("nan")
            seg_idx = min(step_idx // steps_per_seg, len(segment_names) - 1)
            print(
                f"  [{step_idx + 1:6d}/{total_steps}] "
                f"seg={segment_names[seg_idx]:30s} "
                f"err={mean_err:5.1f}deg "
                f"alive={alive_count}/{num_envs}"
            )

    return {
        "time": time_s,
        "target_roll_deg": target_roll_deg,
        "target_pitch_deg": target_pitch_deg,
        "actual_roll_deg": actual_roll,
        "actual_pitch_deg": actual_pitch,
        "error_roll": error_roll,
        "error_pitch": error_pitch,
        "terminated": terminated,
        "steps_per_segment": steps_per_seg,
        "segment_duration": segment_duration,
        "segment_names": segment_names,
    }


# ============================================================================
# Main
# ============================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main evaluation function."""
    task_name = args_cli.task.split(":")[-1]
    is_tdc = "TDC" in task_name
    is_pure_tdc = task_name == "Isaac-HeroAgent-TDC-v0"
    use_checkpoint = args_cli.checkpoint != "none" if args_cli.checkpoint else True

    # ---- Output directory ----
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # e.g. "Isaac-HeroAgent-Encoder-Base-v0" -> "hero_agent_encoder_base"
        folder_name = task_name.removeprefix("Isaac-").lower().replace("-", "_").removesuffix("_v0")
        output_dir = os.path.join("logs", "eval_dr", folder_name, ts)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # ---- Env config overrides (evaluation mode) ----
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.randomize_target_attitude = False
    if hasattr(env_cfg, "observation_noise_model"):
        env_cfg.observation_noise_model = None
    env_cfg.enable_payload = True
    # Terminate ONLY on excessive attitude deviation
    env_cfg.max_attitude_angle = 2.5  # ~143 deg absolute limit
    env_cfg.debug_vis = False
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "doraemon"):
        env_cfg.doraemon.enable = False
    env_cfg.randomization.enable = True

    # Compute episode_length_s from the longest trajectory across all DR levels
    _max_segs = 0
    for _a in DR_ANGLES.values():
        if _a > 0:
            _wp = [(0, 0, ""), (_a, 0, ""), (0, _a, ""), (-_a, -_a, ""), (_a, -_a, ""), (0, 0, "")]
            _max_segs = max(_max_segs, len(_interpolate_waypoints(_wp, 15.0)))
        else:
            _max_segs = max(_max_segs, 1)
    env_cfg.episode_length_s = _max_segs * args_cli.segment_duration + 10.0  # +10s margin

    # ---- Load checkpoint ----
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    resume_path = None
    if use_checkpoint and not is_pure_tdc:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint and args_cli.checkpoint != "none":
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Checkpoint: {resume_path}")

    # ---- Create env (initial DR = none) ----
    apply_dr_config(env_cfg, DR_SCALE["none"], is_tdc)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    raw_env = unwrap_env(env)
    step_dt = raw_env.step_dt
    num_envs = raw_env.num_envs
    device = raw_env.device

    print(f"[INFO] step_dt={step_dt:.4f}s, num_envs={num_envs}, device={device}")
    print(f"[INFO] Segment duration: {args_cli.segment_duration}s")
    print(f"[INFO] DR scales: {DR_SCALE}")

    # ---- Create runner + load policy ----
    runner_cls_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    is_sac_mpc = runner_cls_name == "SACMPCRunner"

    if use_checkpoint and resume_path:
        if is_sac_mpc:
            from isaaclab_tasks.direct.hero_agent_mpc.runners import SACMPCRunner

            runner = SACMPCRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            runner.actor.eval()
            policy = runner.actor.act_inference
            policy_nn = runner.actor
        elif runner_cls_name == "EncoderRunner":
            runner = EncoderRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            policy_nn = runner.alg.policy
        elif runner_cls_name == "BaseRunner":
            runner = BaseRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            policy_nn = runner.alg.policy
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = runner.alg.actor_critic

        print(f"[INFO] Loaded {runner_cls_name} from {resume_path}")
        connect_encoder_to_env(env, policy_nn, "EvalDR")

        # SAC-MPC: initialize prediction error buffer
        if hasattr(policy_nn, "_ensure_pred_error_buf"):
            policy_nn._ensure_pred_error_buf(num_envs, device)
    else:
        action_dim = env_cfg.action_space
        policy = lambda obs: torch.zeros(num_envs, action_dim, device=device)  # noqa: E731
        policy_nn = type("FakePolicy", (), {"reset": lambda _s, _d: None})()
        print("[INFO] Pure TDC mode (zero-action policy).")

    # ---- Pre-compute max segment count so all levels have equal length ----
    max_num_segs = 0
    for angle in DR_ANGLES.values():
        _, _, _, sn = build_step_trajectory(args_cli.segment_duration, step_dt, angle)
        max_num_segs = max(max_num_segs, len(sn))
    print(f"[INFO] Unified segment count: {max_num_segs} ({max_num_segs * args_cli.segment_duration:.0f}s)")

    # ---- Run evaluation for each DR level ----
    all_data = {}
    all_metrics = {}

    for level in DR_LEVELS:
        dr_pct = int(DR_SCALE[level] * 100)
        print(f"\n{'=' * 60}")
        print(f"  DR Level: {level.upper()} | DR Scale: {dr_pct}% | Target: +-15 deg")
        print(f"{'=' * 60}")

        # Build trajectory for this level (padded to max_num_segs)
        time_s, target_roll_deg, target_pitch_deg, segment_names = build_step_trajectory(
            segment_duration=args_cli.segment_duration,
            step_dt=step_dt,
            max_angle_deg=DR_ANGLES[level],
            target_num_segments=max_num_segs,
        )
        print(f"  Trajectory: {len(segment_names)} segs x {args_cli.segment_duration}s = {len(time_s)} steps")

        # Apply DR preset
        apply_dr_config(raw_env.cfg, DR_SCALE[level], is_tdc)

        # Run evaluation
        data = run_evaluation(
            env=env,
            policy=policy,
            policy_nn=policy_nn,
            raw_env=raw_env,
            time_s=time_s,
            target_roll_deg=target_roll_deg,
            target_pitch_deg=target_pitch_deg,
            segment_names=segment_names,
            segment_duration=args_cli.segment_duration,
            step_dt=step_dt,
            num_envs=num_envs,
            device=device,
        )
        all_data[level] = data

        # Save per-level data
        np.savez_compressed(
            os.path.join(output_dir, f"eval_{level}.npz"),
            **{k: v for k, v in data.items() if isinstance(v, np.ndarray)},
        )

        metrics = compute_metrics(data)
        all_metrics[level] = metrics

        print(f"\n  Results ({level}, DR {dr_pct}%):")
        print(f"    Total mean error:    {metrics['total_mean_error']:.1f} deg")
        print(f"    Survival rate:       {metrics['survival_rate']:.0f}%")
        print(f"    Steady-state (avg):  {np.nanmean(metrics['steady_state_errors']):.1f} deg")
        print(f"    Settling time (avg): {np.nanmean(metrics['settling_times']):.2f} s")

    # ---- Generate plots ----
    print(f"\n[INFO] Generating plots...")
    generate_plots(all_data, all_metrics, output_dir)

    # ---- Print final comparison ----
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Level':<10} {'DR%':>5} {'MeanErr':>8} {'SS Err':>8} {'Settle':>8} {'Survival':>10}")
    print(f"{'-'*10} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for lvl in DR_LEVELS:
        m = all_metrics[lvl]
        print(
            f"{lvl:<10} "
            f"{int(DR_SCALE[lvl] * 100):4d}% "
            f"{m['total_mean_error']:7.1f}d "
            f"{np.nanmean(m['steady_state_errors']):7.1f}d "
            f"{np.nanmean(m['settling_times']):7.2f}s "
            f"{m['survival_rate']:9.0f}%"
        )
    print(f"{'=' * 70}")
    print(f"\nOutput saved to: {output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
