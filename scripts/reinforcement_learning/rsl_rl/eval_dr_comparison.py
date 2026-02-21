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

All levels start from 0 deg initial pose. Each segment is held for 10 s to
ensure at least 5 s of steady-state observation after settling.

Usage:
    # Pure TDC baseline
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py \
        --task Isaac-HeroAgent-TDC-v0 --checkpoint none --num_envs 16 --headless

    # Encoder-Base policy
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py \
        --task Isaac-HeroAgent-Encoder-Base-v0 --num_envs 64 --headless
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
parser.add_argument("--segment_duration", type=float, default=10.0, help="Duration per segment in seconds (default 10).")
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
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

matplotlib.use("Agg")  # non-interactive backend for headless

# ---- Configuration ----
DR_LEVELS = ["none", "soft", "medium", "hard"]
DR_COLORS = {"none": "#2196F3", "soft": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}
DR_ANGLES = {"none": 15.0, "soft": 15.0, "medium": 15.0, "hard": 15.0}

# DR scale factors: 0 = nominal physics, 1 = full training DR
DR_SCALE = {"none": 0.0, "soft": 0.3, "medium": 0.6, "hard": 1.0}


# ============================================================================
# DR Presets
# ============================================================================

def _dr_preset(level: str, is_tdc: bool) -> dict:
    """Return DR parameter overrides via linear interpolation from nominal to training DR.

    Scale factors (DR_SCALE): none=0%, soft=30%, medium=60%, hard=100%.
    "Full" values match DomainRandomizationCfg training defaults.
    Initial pose is always (0,0) for all levels.
    """
    f = DR_SCALE[level]

    def lerp_scale(full_lo: float, full_hi: float) -> tuple[float, float]:
        """Interpolate multiplicative scale range (nominal = 1.0)."""
        return (1.0 - f * (1.0 - full_lo), 1.0 + f * (full_hi - 1.0))

    def lerp_range(full_lo: float, full_hi: float, nominal: float = 0.0) -> tuple[float, float]:
        """Interpolate value range from nominal."""
        return (nominal + f * (full_lo - nominal), nominal + f * (full_hi - nominal))

    def lerp_int(full_lo: int, full_hi: int) -> tuple[int, int]:
        """Interpolate integer range."""
        return (int(f * full_lo), int(round(f * full_hi)))

    # Full training DR values (from DomainRandomizationCfg defaults)
    p = dict(
        enable=True,
        inertia_scale=lerp_scale(0.4, 2.5),
        body_mass_scale=lerp_scale(0.7, 1.3),
        volume_scale=lerp_scale(0.7, 1.3),
        added_mass_scale=lerp_scale(0.8, 1.2),
        linear_damping_scale=lerp_scale(0.7, 1.3),
        quadratic_damping_scale=lerp_scale(0.6, 1.4),
        water_density_range=lerp_range(995.0, 1025.0, 998.0),
        enable_perturbation=f > 0,
        perturbation_force_range=lerp_range(0.0, 10.0),
        perturbation_torque_range=lerp_range(0.0, 1.5),
        payload_mass_range=lerp_range(0.0, 1.5),
        action_latency_range=lerp_int(0, 4),
        cob_offset_x=lerp_range(-0.01, 0.01),
        cob_offset_y=lerp_range(-0.01, 0.01),
        cob_offset_z=lerp_range(-0.04, 0.04),
        cog_offset_x=lerp_range(-0.01, 0.01),
        cog_offset_y=lerp_range(-0.01, 0.01),
        cog_offset_z=lerp_range(-0.06, 0.06),
        payload_cog_offset_x=lerp_range(-0.30, 0.30),
        payload_cog_offset_y=lerp_range(-0.30, 0.30),
        payload_cog_offset_z=lerp_range(-0.20, 0.0),
        joint_static_friction_range=lerp_range(0.0, 0.05),
        joint_viscous_friction_range=lerp_range(0.0, 0.3),
    )

    # All levels: initial pose at (0, 0)
    p["roll_range"] = (0.0, 0.0)
    p["pitch_range"] = (0.0, 0.0)

    # Joint gain centers differ for TDC vs Base RL
    if is_tdc:
        p["joint_stiffness_range"] = (160.0, 240.0)
        p["joint_damping_range"] = (8.0, 12.0)
        p["action_latency_range"] = (0, 0)
    else:
        # Training defaults: stiffness=(80, 120), damping=(2.4, 3.6)
        p["joint_stiffness_range"] = lerp_range(80.0, 120.0, 100.0)
        p["joint_damping_range"] = lerp_range(2.4, 3.6, 3.0)

    return p


def apply_dr_preset(env_cfg, level: str, is_tdc: bool) -> None:
    """Apply a DR preset to the environment config's randomization field."""
    preset = _dr_preset(level, is_tdc)
    rand_cfg = env_cfg.randomization
    for key, val in preset.items():
        if hasattr(rand_cfg, key):
            setattr(rand_cfg, key, val)


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
    axes[0, 0].bar(x, ss_errors, color=bar_colors)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(xlabels, fontsize=9)
    axes[0, 0].set_ylabel("Error (deg)")
    axes[0, 0].set_title("Steady-State Error (last 5s avg)")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    settle_times = [np.nanmean(all_metrics[lvl]["settling_times"]) for lvl in levels]
    axes[0, 1].bar(x, settle_times, color=bar_colors)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(xlabels, fontsize=9)
    axes[0, 1].set_ylabel("Time (s)")
    axes[0, 1].set_title("Settling Time (<5 deg)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    total_errors = [all_metrics[lvl]["total_mean_error"] for lvl in levels]
    axes[1, 0].bar(x, total_errors, color=bar_colors)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(xlabels, fontsize=9)
    axes[1, 0].set_ylabel("Error (deg)")
    axes[1, 0].set_title("Total Mean Error")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    survivals = [all_metrics[lvl]["survival_rate"] for lvl in levels]
    axes[1, 1].bar(x, survivals, color=bar_colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(xlabels, fontsize=9)
    axes[1, 1].set_ylabel("Survival (%)")
    axes[1, 1].set_title("Survival Rate")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

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
        if hasattr(policy_nn, "reset"):
            policy_nn.reset(torch.ones(num_envs, 1, dtype=torch.bool, device=device))

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
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy_nn, "reset"):
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
        folder_name = task_name.removeprefix("Isaac-").lower().replace("-", "_").rstrip("_v0")
        output_dir = os.path.join("logs", "eval_dr", folder_name, ts)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # ---- Env config overrides (evaluation mode) ----
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.randomize_target_attitude = False
    env_cfg.observation_noise_model = None
    env_cfg.enable_payload = True
    # Terminate ONLY on excessive attitude deviation
    env_cfg.max_attitude_angle = 2.5  # ~143 deg absolute limit
    env_cfg.debug_vis = False
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "dr_curriculum"):
        env_cfg.dr_curriculum.enable = False
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
    apply_dr_preset(env_cfg, "none", is_tdc)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    raw_env = env.unwrapped
    step_dt = raw_env.step_dt
    num_envs = raw_env.num_envs
    device = raw_env.device

    print(f"[INFO] step_dt={step_dt:.4f}s, num_envs={num_envs}, device={device}")
    print(f"[INFO] Segment duration: {args_cli.segment_duration}s")
    print(f"[INFO] DR scales: {DR_SCALE}")

    # ---- Create runner + load policy ----
    if use_checkpoint and resume_path:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "get_last_z") and hasattr(raw_env, "set_encoder_policy"):
            raw_env.set_encoder_policy(policy_nn)
            print("[INFO] Encoder policy connected to env.")
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
        apply_dr_preset(raw_env.cfg, level, is_tdc)

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
