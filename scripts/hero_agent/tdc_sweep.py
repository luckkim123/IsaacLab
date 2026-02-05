# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TDC Controller Gain Sweep Script.

Standalone verification of the TDC controller without RL policy.
Sweeps fixed (K_p, K_d) gain combinations and measures attitude error
to find optimal parameters before RL training.

Usage:
    cd /workspace/isaaclab

    # Quick test (3x3 sweep, 100 steps)
    ./isaaclab.sh -p scripts/hero_agent/tdc_sweep.py \
        --num_envs 256 --episode_steps 100 \
        --k_p_values "5,20,50" --k_d_values "0.5,2,10" \
        --headless

    # Full sweep with wandb logging
    ./isaaclab.sh -p scripts/hero_agent/tdc_sweep.py \
        --num_envs 4096 --episode_steps 500 \
        --logger wandb --log_project_name hero_agent_tdc_sweep \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# CLI arguments (must be parsed before AppLauncher)
parser = argparse.ArgumentParser(description="TDC Controller Gain Sweep for Hero Agent.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--episode_steps", type=int, default=500, help="Steps per episode per gain combination.")
parser.add_argument("--k_p_values", type=str, default="1,5,10,20,30,50", help="Comma-separated K_p values to sweep.")
parser.add_argument("--k_d_values", type=str, default="0.1,0.5,1,2,5,10", help="Comma-separated K_d values to sweep.")
parser.add_argument("--initial_roll", type=float, default=0.5, help="Initial roll angle in radians (~29 deg).")
parser.add_argument("--initial_pitch", type=float, default=0.5, help="Initial pitch angle in radians (~29 deg).")
parser.add_argument(
    "--logger", type=str, default="tensorboard", choices=["wandb", "tensorboard", "none"],
    help="Logging backend.",
)
parser.add_argument("--log_project_name", type=str, default="hero_agent_tdc_sweep", help="WandB project name.")

# AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Clear sys.argv for any downstream Hydra usage
sys.argv = [sys.argv[0]]

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
from datetime import datetime

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 -- triggers gym.register for Hero Agent envs
from isaaclab_tasks.utils import parse_env_cfg

# ---------------------------------------------------------------------------
# Gain <-> Action conversion
# ---------------------------------------------------------------------------

def gains_to_actions(
    k_p: float,
    k_d: float,
    k_p_min: float,
    k_p_max: float,
    k_d_min: float,
    k_d_max: float,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Convert desired (K_p, K_d) to action tensor in [-1, 1].

    TDC set_gains(): action_01 = (action + 1) * 0.5; gain = min + action_01 * range.
    Inverse: action = 2 * (gain - min) / range - 1.

    Applies same K_p/K_d to both roll and pitch (symmetric).

    Returns:
        actions: Shape (num_envs, 4) = [kp_roll, kd_roll, kp_pitch, kd_pitch].
    """
    k_p_range = k_p_max - k_p_min
    k_d_range = k_d_max - k_d_min

    a_kp = 2.0 * (k_p - k_p_min) / k_p_range - 1.0
    a_kd = 2.0 * (k_d - k_d_min) / k_d_range - 1.0

    # Clamp to valid range
    a_kp = max(-1.0, min(1.0, a_kp))
    a_kd = max(-1.0, min(1.0, a_kd))

    actions = torch.tensor([[a_kp, a_kd, a_kp, a_kd]], device=device)
    return actions.expand(num_envs, -1).contiguous()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: gym.Env,
    actions: torch.Tensor,
    num_steps: int,
) -> dict[str, float]:
    """Run one episode with fixed gain actions and collect metrics.

    Args:
        env: The gymnasium environment (wrapping HeroAgentTDCEnv).
        actions: Fixed gain actions. Shape: (num_envs, 4).
        num_steps: Number of simulation steps.

    Returns:
        Dictionary of collected metric tensors.
    """
    raw_env = env.unwrapped
    num_envs = raw_env.num_envs
    device = raw_env.device

    # Accumulators (per-env, per-step)
    roll_errors = torch.zeros(num_envs, num_steps, device=device)
    pitch_errors = torch.zeros(num_envs, num_steps, device=device)
    workspace_utils = torch.zeros(num_envs, num_steps, device=device)
    p_ee_x = torch.zeros(num_envs, num_steps, device=device)
    p_ee_y = torch.zeros(num_envs, num_steps, device=device)

    for step in range(num_steps):
        obs, reward, terminated, truncated, info = env.step(actions)

        # Attitude error (recompute from current state)
        attitude_error = raw_env._compute_attitude_error(raw_env._robot.data.root_quat_w)
        roll_errors[:, step] = attitude_error[:, 0].abs()
        pitch_errors[:, step] = attitude_error[:, 1].abs()

        # Workspace utilization from TDC controller
        workspace_utils[:, step] = raw_env._tdc_controller.workspace_utilization

        # EE position from TDC controller
        p_ee_x[:, step] = raw_env._tdc_controller._last_output[:, 0]
        p_ee_y[:, step] = raw_env._tdc_controller._last_output[:, 1]

    # Aggregate metrics across envs and steps
    deg = 180.0 / math.pi
    roll_mean = roll_errors.mean() * deg
    pitch_mean = pitch_errors.mean() * deg
    total_mean = (roll_errors + pitch_errors).mean() * deg

    # Final convergence: mean of last 20% of steps
    tail = max(1, num_steps // 5)
    roll_final = roll_errors[:, -tail:].mean() * deg
    pitch_final = pitch_errors[:, -tail:].mean() * deg
    total_final = (roll_errors[:, -tail:] + pitch_errors[:, -tail:]).mean() * deg

    ws_mean = workspace_utils.mean()
    ws_max = workspace_utils.max()

    return {
        "attitude_error/roll_deg": roll_mean.item(),
        "attitude_error/pitch_deg": pitch_mean.item(),
        "attitude_error/total_deg": total_mean.item(),
        "attitude_error/final_roll_deg": roll_final.item(),
        "attitude_error/final_pitch_deg": pitch_final.item(),
        "attitude_error/final_total_deg": total_final.item(),
        "workspace/utilization_mean": ws_mean.item(),
        "workspace/utilization_max": ws_max.item(),
        "p_ee/x_mean": p_ee_x.mean().item(),
        "p_ee/y_mean": p_ee_y.mean().item(),
    }


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(logger_type: str, project_name: str, log_dir: str) -> dict:
    """Initialize logging backends.

    Returns:
        Dictionary with logger handles: {"wandb": run_or_None, "tb": writer_or_None}.
    """
    loggers = {"wandb": None, "tb": None}

    if logger_type in ("wandb", "tensorboard"):
        os.makedirs(log_dir, exist_ok=True)

    if logger_type == "wandb":
        import wandb

        run = wandb.init(
            project=project_name,
            name=f"tdc_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dir=log_dir,
            config={
                "k_p_values": args_cli.k_p_values,
                "k_d_values": args_cli.k_d_values,
                "num_envs": args_cli.num_envs,
                "episode_steps": args_cli.episode_steps,
                "initial_roll": args_cli.initial_roll,
                "initial_pitch": args_cli.initial_pitch,
            },
        )
        loggers["wandb"] = run

    if logger_type == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        loggers["tb"] = SummaryWriter(log_dir=log_dir)

    return loggers


def log_metrics(loggers: dict, metrics: dict, k_p: float, k_d: float, combo_idx: int) -> None:
    """Log metrics for one (K_p, K_d) combination."""
    full_metrics = {**metrics, "K_p": k_p, "K_d": k_d}

    if loggers["wandb"] is not None:
        import wandb

        wandb.log(full_metrics, step=combo_idx)

    if loggers["tb"] is not None:
        writer = loggers["tb"]
        for key, val in full_metrics.items():
            writer.add_scalar(key, val, combo_idx)
        writer.flush()


def close_loggers(loggers: dict) -> None:
    """Close logging backends."""
    if loggers["wandb"] is not None:
        import wandb

        # Log summary table
        wandb.finish()

    if loggers["tb"] is not None:
        loggers["tb"].close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the TDC gain sweep."""
    # Parse gain values
    k_p_values = [float(v) for v in args_cli.k_p_values.split(",")]
    k_d_values = [float(v) for v in args_cli.k_d_values.split(",")]

    total_combos = len(k_p_values) * len(k_d_values)
    print(f"[INFO] TDC Gain Sweep: {len(k_p_values)} K_p x {len(k_d_values)} K_d = {total_combos} combinations")
    print(f"[INFO] K_p values: {k_p_values}")
    print(f"[INFO] K_d values: {k_d_values}")
    print(f"[INFO] Envs: {args_cli.num_envs}, Steps/combo: {args_cli.episode_steps}")
    print(f"[INFO] Initial attitude: roll={args_cli.initial_roll:.3f} rad ({math.degrees(args_cli.initial_roll):.1f} deg),"
          f" pitch={args_cli.initial_pitch:.3f} rad ({math.degrees(args_cli.initial_pitch):.1f} deg)")

    # --- Environment setup ---
    task_name = "Isaac-HeroAgent-Base-TDC-v0"
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)

    # Override DR: enable but fix all ranges to exact values for controlled experiments
    rand_cfg = env_cfg.randomization
    rand_cfg.enable = True
    rand_cfg.roll_range = (args_cli.initial_roll, args_cli.initial_roll)
    rand_cfg.pitch_range = (args_cli.initial_pitch, args_cli.initial_pitch)
    rand_cfg.yaw_range = (0.0, 0.0)
    rand_cfg.position_x_range = (0.0, 0.0)
    rand_cfg.position_y_range = (0.0, 0.0)
    rand_cfg.position_z_range = (4.5, 4.5)
    # Fix hydro/inertia randomization to nominal values
    rand_cfg.added_mass_scale = (1.0, 1.0)
    rand_cfg.linear_damping_scale = (1.0, 1.0)
    rand_cfg.quadratic_damping_scale = (1.0, 1.0)
    rand_cfg.volume_scale = (1.0, 1.0)
    rand_cfg.inertia_scale = (1.0, 1.0)
    rand_cfg.cob_offset_x = (0.0, 0.0)
    rand_cfg.cob_offset_y = (0.0, 0.0)
    rand_cfg.cob_offset_z = (0.0, 0.0)
    rand_cfg.cog_offset_x = (0.0, 0.0)
    rand_cfg.cog_offset_y = (0.0, 0.0)
    rand_cfg.cog_offset_z = (0.0, 0.0)

    # Disable ocean current
    env_cfg.ocean_current.max_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    env_cfg.ocean_current.noise_scale = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Disable payload
    env_cfg.enable_payload = False

    # Long episode to avoid premature truncation during sweep
    env_cfg.episode_length_s = args_cli.episode_steps * env_cfg.sim.dt * 2.0

    # Disable debug vis for headless sweep
    env_cfg.debug_vis = False

    # Create environment
    env = gym.make(task_name, cfg=env_cfg)
    raw_env = env.unwrapped
    print(f"[INFO] Environment created: {task_name}")
    print(f"[INFO] Action space: {env.action_space}, Obs space: {env.observation_space}")

    # Extract gain bounds from env config for action conversion
    k_p_min = env_cfg.tdc_k_p_min
    k_p_max = env_cfg.tdc_k_p_max
    k_d_min = env_cfg.tdc_k_d_min
    k_d_max = env_cfg.tdc_k_d_max

    # --- Logging setup ---
    log_dir = os.path.join(
        "logs", "tdc_sweep",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    loggers = setup_logger(args_cli.logger, args_cli.log_project_name, log_dir)
    print(f"[INFO] Logging to: {log_dir} (backend: {args_cli.logger})")

    # --- Sweep loop ---
    results = []
    combo_idx = 0

    for k_p in k_p_values:
        for k_d in k_d_values:
            print(f"\n[SWEEP {combo_idx + 1}/{total_combos}] K_p={k_p:.1f}, K_d={k_d:.1f}")

            # Convert gains to actions
            actions = gains_to_actions(
                k_p=k_p, k_d=k_d,
                k_p_min=k_p_min, k_p_max=k_p_max,
                k_d_min=k_d_min, k_d_max=k_d_max,
                num_envs=raw_env.num_envs,
                device=raw_env.device,
            )

            # Verify gain conversion round-trip
            if combo_idx == 0:
                raw_env._tdc_controller.set_gains(actions)
                actual_kp, actual_kd = raw_env._tdc_controller.current_gains
                print(f"  Gain verification: K_p={actual_kp[0].tolist()}, K_d={actual_kd[0].tolist()}")

            # Reset + run must share inference_mode context.
            # env.step() inside inference_mode marks internal buffers as inference tensors,
            # so env.reset() must also be inside inference_mode to do inplace updates on them.
            with torch.inference_mode():
                env.reset()
                metrics = run_episode(env, actions, args_cli.episode_steps)

            # Log results
            log_metrics(loggers, metrics, k_p, k_d, combo_idx)
            results.append({"K_p": k_p, "K_d": k_d, **metrics})

            # Print summary
            print(f"  Total error:  {metrics['attitude_error/total_deg']:.2f} deg (mean)"
                  f"  ->  {metrics['attitude_error/final_total_deg']:.2f} deg (final)")
            print(f"  Roll/Pitch:   {metrics['attitude_error/roll_deg']:.2f} / {metrics['attitude_error/pitch_deg']:.2f} deg")
            print(f"  Workspace:    {metrics['workspace/utilization_mean']:.3f} (mean)"
                  f"  {metrics['workspace/utilization_max']:.3f} (max)")

            combo_idx += 1

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("TDC GAIN SWEEP RESULTS")
    print("=" * 90)
    print(f"{'K_p':>6}  {'K_d':>6}  {'Total(mean)':>12}  {'Total(final)':>13}  {'WS_mean':>8}  {'WS_max':>7}")
    print("-" * 90)

    # Sort by final total error (ascending)
    results_sorted = sorted(results, key=lambda r: r["attitude_error/final_total_deg"])
    for r in results_sorted:
        print(
            f"{r['K_p']:>6.1f}  {r['K_d']:>6.1f}"
            f"  {r['attitude_error/total_deg']:>12.2f}"
            f"  {r['attitude_error/final_total_deg']:>13.2f}"
            f"  {r['workspace/utilization_mean']:>8.3f}"
            f"  {r['workspace/utilization_max']:>7.3f}"
        )

    print("-" * 90)
    best = results_sorted[0]
    print(f"BEST: K_p={best['K_p']:.1f}, K_d={best['K_d']:.1f}"
          f" -> final error {best['attitude_error/final_total_deg']:.2f} deg")
    print("=" * 90)

    # Log wandb summary table
    if loggers["wandb"] is not None:
        import wandb

        columns = ["K_p", "K_d", "total_mean_deg", "total_final_deg", "ws_mean", "ws_max"]
        table = wandb.Table(columns=columns)
        for r in results_sorted:
            table.add_data(
                r["K_p"], r["K_d"],
                r["attitude_error/total_deg"],
                r["attitude_error/final_total_deg"],
                r["workspace/utilization_mean"],
                r["workspace/utilization_max"],
            )
        wandb.log({"sweep_results": table})

    # Cleanup
    close_loggers(loggers)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
