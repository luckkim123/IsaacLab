# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Thruster Layout Visualization + Force Test.

Two modes:
    --mode viz   (default) Spawn robot with arrow markers showing thruster
                 positions and thrust directions. No physics, just visual.
    --mode test  Apply constant thrust per direction (5s each) to verify motion.

Usage:
    ./isaaclab.sh -p scripts/demos/test_hero_thruster.py --num_envs 1 --livestream 2
    ./isaaclab.sh -p scripts/demos/test_hero_thruster.py --mode test --num_envs 2 --livestream 2
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hero Agent Thruster Verification")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", choices=["viz", "test"], default="viz")
parser.add_argument("--thrust_amp", type=float, default=0.3)
parser.add_argument("--phase_seconds", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab imports (after app launch) ---

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from isaaclab_tasks.direct.constrained_albc.albc_env import ALBCEnv
from isaaclab_tasks.direct.constrained_albc.config import ALBCEnvCfg, HeroAgentThrusterCfg

# ---------------------------------------------------------------------------
# Thruster geometry from actuators.xacro (positions and orientations)
# Position: relative to base center, force axis = local X after rotation
# ---------------------------------------------------------------------------
THRUSTER_POSITIONS = [
    (-0.102, -0.102, 0.010),  # T0 rear-left horizontal
    (-0.102, +0.102, 0.010),  # T1 rear-right horizontal
    (+0.102, -0.102, 0.010),  # T2 front-left horizontal
    (+0.102, +0.102, 0.010),  # T3 front-right horizontal
    (-0.145, 0.000, -0.111),  # T4 rear vertical
    (+0.145, 0.000, -0.111),  # T5 front vertical
]
THRUSTER_RPY = [
    (0.0, 0.0, -math.pi / 4),       # T0
    (0.0, 0.0, -3 * math.pi / 4),   # T1
    (0.0, 0.0, +math.pi / 4),       # T2
    (0.0, 0.0, +3 * math.pi / 4),   # T3
    (0.0, -math.pi / 2, 0.0),       # T4
    (0.0, -math.pi / 2, 0.0),       # T5
]
THRUSTER_COLORS = [
    (1.0, 0.3, 0.3),  # T0 red
    (0.3, 1.0, 0.3),  # T1 green
    (0.3, 0.3, 1.0),  # T2 blue
    (1.0, 1.0, 0.3),  # T3 yellow
    (1.0, 0.3, 1.0),  # T4 magenta
    (0.3, 1.0, 1.0),  # T5 cyan
]
THRUSTER_NAMES = ["T0(RL)", "T1(RR)", "T2(FL)", "T3(FR)", "T4(V-R)", "T5(V-F)"]

# TAM-derived pure direction commands: [T0, T1, T2, T3, T4, T5]
# fmt: off
TEST_PHASES = [
    ("Neutral",          0.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    ("Forward  (+X)",   +1.0, -1.0, +1.0, -1.0,  0.0,  0.0),
    ("Backward (-X)",   -1.0, +1.0, -1.0, +1.0,  0.0,  0.0),
    ("Left     (+Y)",   -1.0, -1.0, +1.0, +1.0,  0.0,  0.0),
    ("Right    (-Y)",   +1.0, +1.0, -1.0, -1.0,  0.0,  0.0),
    ("Up       (+Z)",    0.0,  0.0,  0.0,  0.0, +1.0, +1.0),
    ("Down     (-Z)",    0.0,  0.0,  0.0,  0.0, -1.0, -1.0),
    ("Yaw CW   (+Mz)",  +1.0, +1.0, +1.0, +1.0,  0.0,  0.0),
]
# fmt: on


def _create_arrow_markers(num_envs: int) -> list[tuple[VisualizationMarkers, VisualizationMarkers]]:
    """Create shaft (cylinder) + head (cone) marker pair per thruster."""
    marker_pairs = []
    shaft_len = 0.12
    head_len = 0.04
    for i, color in enumerate(THRUSTER_COLORS):
        shaft_cfg = VisualizationMarkersCfg(
            prim_path=f"/Visuals/ThrusterShaft_{i}",
            markers={
                "shaft": sim_utils.CylinderCfg(
                    radius=0.004,
                    height=shaft_len,
                    axis="X",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        head_cfg = VisualizationMarkersCfg(
            prim_path=f"/Visuals/ThrusterHead_{i}",
            markers={
                "head": sim_utils.ConeCfg(
                    radius=0.012,
                    height=head_len,
                    axis="X",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        shaft_m = VisualizationMarkers(shaft_cfg)
        head_m = VisualizationMarkers(head_cfg)
        marker_pairs.append((shaft_m, head_m))
    return marker_pairs


def _update_arrow_markers(
    marker_pairs: list[tuple[VisualizationMarkers, VisualizationMarkers]],
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    device: str,
) -> None:
    """Position shaft+head markers at thruster locations in world frame."""
    num_envs = robot_pos.shape[0]
    shaft_half = 0.06  # half of shaft_len=0.12
    head_offset = 0.12 + 0.02  # shaft_len + half of head_len

    for i in range(6):
        # Thruster orientation in world frame
        r, p, y = THRUSTER_RPY[i]
        local_quat = quat_from_euler_xyz(
            torch.tensor([r], device=device),
            torch.tensor([p], device=device),
            torch.tensor([y], device=device),
        ).expand(num_envs, -1)
        world_quat = _quat_mul(robot_quat, local_quat)

        # Local X direction in world frame (thrust direction)
        local_x = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
        thrust_dir_w = quat_apply(world_quat, local_x)

        # Thruster base position in world frame
        local_pos = torch.tensor(THRUSTER_POSITIONS[i], device=device, dtype=torch.float32)
        base_pos_w = robot_pos + quat_apply(robot_quat, local_pos.unsqueeze(0).expand(num_envs, -1))

        # Shaft center = base + shaft_half * thrust_dir
        shaft_pos = base_pos_w + shaft_half * thrust_dir_w
        # Head center = base + head_offset * thrust_dir
        head_pos = base_pos_w + head_offset * thrust_dir_w

        shaft_m, head_m = marker_pairs[i]
        shaft_m.visualize(translations=shaft_pos, orientations=world_quat)
        head_m.visualize(translations=head_pos, orientations=world_quat)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2. Quaternion format: (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _force_joint_positions(env: ALBCEnv, g1: float, g2: float) -> None:
    """Force ALBC joints to exact positions."""
    joint_pos = env._robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, env._albc_joint_ids[0]] = g1
    joint_pos[:, env._albc_joint_ids[1]] = g2
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env._joint_pos_targets[:] = torch.tensor([g1, g2], device=env.device)


def run_viz():
    """Visual-only mode: show robot + thruster arrows, no physics forces."""
    cfg = ALBCEnvCfg()
    cfg.thrusters = HeroAgentThrusterCfg()
    cfg.action_space = 8
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 300.0
    cfg.randomization.enable = False

    env = ALBCEnv(cfg)
    env.reset()
    _force_joint_positions(env, 0.0, math.pi)

    arrow_markers = _create_arrow_markers(args_cli.num_envs)

    # Draw arrows once at initial position
    _update_arrow_markers(
        arrow_markers,
        env._robot.data.root_pos_w,
        env._robot.data.root_quat_w,
        env.device,
    )

    print("\n" + "=" * 70)
    print("Thruster Layout Visualization (paused -- render only, no physics)")
    print("  Arrows show thruster position and thrust direction (local X axis)")
    for i, name in enumerate(THRUSTER_NAMES):
        r, g, b = THRUSTER_COLORS[i]
        pos = THRUSTER_POSITIONS[i]
        print(f"  {name}: pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})  color=({r},{g},{b})")
    print("  Rotate camera freely. Press Ctrl+C to exit.")
    print("=" * 70)

    # Render loop only -- no physics stepping, robot stays frozen
    sim = env.sim
    while simulation_app.is_running():
        sim.render()

    env.close()


def run_test():
    """Force test: apply constant thrust per direction to verify motion."""
    amp = args_cli.thrust_amp
    phase_seconds = args_cli.phase_seconds

    cfg = ALBCEnvCfg()
    cfg.thrusters = HeroAgentThrusterCfg()
    cfg.action_space = 8
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = phase_seconds * len(TEST_PHASES) + 10.0
    cfg.randomization.enable = False

    env = ALBCEnv(cfg)
    env.reset()
    _force_joint_positions(env, 0.0, math.pi)

    arrow_markers = _create_arrow_markers(args_cli.num_envs)
    env_dt = cfg.decimation * cfg.sim.dt
    phase_steps = int(phase_seconds / env_dt)

    print("\n" + "=" * 95)
    print("Thruster Force Test")
    print(f"  Amp: {amp} | Coeff: {cfg.thrusters.thrust_coefficient} | Max: {cfg.thrusters.max_thrust} N")
    print(f"  Phases: {len(TEST_PHASES)} x {phase_seconds}s")
    print("=" * 95)

    for pi_idx, (name, *cmds) in enumerate(TEST_PHASES):
        print(f"\n--- Phase {pi_idx}: {name} ---")
        thruster_cmd = torch.tensor(cmds, device=env.device, dtype=torch.float32) * amp

        for s in range(phase_steps):
            actions = torch.zeros(args_cli.num_envs, 8, device=env.device)
            actions[:, 2:] = thruster_cmd.unsqueeze(0)
            env.step(actions)

            _update_arrow_markers(
                arrow_markers,
                env._robot.data.root_pos_w,
                env._robot.data.root_quat_w,
                env.device,
            )

            print_interval = max(1, int(1.0 / env_dt))
            if s % print_interval == 0:
                t = (pi_idx * phase_steps + s) * env_dt
                pos = env._robot.data.root_pos_w[0]
                vel = env._robot.data.root_lin_vel_w[0]
                roll, pitch, yaw = euler_xyz_from_quat(env._robot.data.root_quat_w[0:1])

                thr_str = ""
                if env._thruster is not None:
                    tf = env._thruster.body_forces[0]
                    tt = env._thruster.body_torques[0]
                    thr_str = (
                        f" | F=({tf[0]:+6.1f},{tf[1]:+6.1f},{tf[2]:+6.1f})"
                        f" | T=({tt[0]:+5.2f},{tt[1]:+5.2f},{tt[2]:+5.2f})"
                    )

                print(
                    f"  [{t:5.1f}s] "
                    f"pos=({pos[0]:+6.2f},{pos[1]:+6.2f},{pos[2]:+6.2f}) "
                    f"vel=({vel[0]:+5.2f},{vel[1]:+5.2f},{vel[2]:+5.2f}) "
                    f"rpy=({math.degrees(roll[0].item()):+5.1f},"
                    f"{math.degrees(pitch[0].item()):+5.1f},"
                    f"{math.degrees(yaw[0].item()):+5.1f})"
                    f"{thr_str}"
                )

                if torch.isnan(pos).any():
                    print("  [ERROR] NaN detected!")
                    env.close()
                    return

    env.close()
    print("\n" + "=" * 95)
    print("Test complete.")
    print("=" * 95)


def main():
    if args_cli.mode == "viz":
        run_viz()
    else:
        run_test()


if __name__ == "__main__":
    main()
    simulation_app.close()
