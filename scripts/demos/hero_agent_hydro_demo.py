# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo script for Hero Agent with hydrodynamics visualization.

This script demonstrates the Hero Agent underwater vehicle with hydrodynamic
forces applied properly for articulated bodies:
    - Full hydrodynamics (damping, Coriolis, buoyancy) on base (root body)
    - Full hydrodynamics on link3 (buoy) - same as RL environment
    - Simple buoyancy (vertical force) on other child links (gripper, link1, link2)

Note: Added mass (M_A * v_dot) is NOT applied as external force. The physics engine
handles M_RB * v_dot internally, and applying M_A * v_dot as external force would
cause numerical instability. The Coriolis matrix C(v) = C_RB(v) + C_A(v) includes
the added mass coupling effects.

Expected Behavior:
    - link3 (buoy): Net +10.83N upward -> floats up
    - base: Net -9.27N downward -> sinks
    - System rotates until buoy is above base, then stabilizes

Usage:
    ./isaaclab.sh -p scripts/demos/hero_agent_hydro_demo.py
"""

from __future__ import annotations

import argparse

import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Hero Agent Hydrodynamics Demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tasks.models.hydrodynamics import HydrodynamicsModel

from isaaclab_assets.robots.uuv import (
    HERO_AGENT_CFG,
    HeroAgentBuoyHydrodynamicsCfg,
    HeroAgentHydrodynamicsCfg,
)


def create_ground_plane():
    """Create a ground plane (seafloor)."""
    cfg = sim_utils.GroundPlaneCfg(color=(0.3, 0.4, 0.35))
    cfg.func("/World/ground", cfg)


def create_lights():
    """Create lighting for underwater scene."""
    cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.6, 0.7, 0.9))
    cfg.func("/World/DomeLight", cfg)

    cfg2 = sim_utils.DistantLightCfg(intensity=2000.0, color=(0.8, 0.9, 1.0))
    cfg2.func("/World/SunLight", cfg2)


# Body geometry parameters for simple buoyancy (non-root, non-buoy bodies)
# Format: {"body_name": {"mass": kg, "volume": m^3}}
# Note: link3 (buoy) uses full hydrodynamics, not simple buoyancy
CHILD_BODY_BUOYANCY_PARAMS = {
    "gripper": {"mass": 0.3, "volume": 0.00023},    # cylinder R=0.015, L=0.325
    "link1": {"mass": 0.1, "volume": 0.000058},     # box 0.233x0.025x0.01
    "link2": {"mass": 0.1, "volume": 0.000058},     # box 0.233x0.025x0.01
}

WATER_DENSITY = 997.0  # kg/m^3
GRAVITY = 9.81  # m/s^2


def compute_buoyancy_force(mass: float, volume: float) -> float:
    """Compute net buoyancy force (positive = upward)."""
    buoyancy = WATER_DENSITY * volume * GRAVITY
    weight = mass * GRAVITY
    return buoyancy - weight


def main():
    """Main function."""
    # Simulation parameters
    dt = 0.005  # 200Hz for stability
    max_sim_time = 15.0

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=dt, render_interval=4)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 3.0], target=[0.0, 0.0, 2.0])

    # Create scene
    create_ground_plane()
    create_lights()

    # Create Hero Agent
    robot_cfg = HERO_AGENT_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 2.0)
    robot = Articulation(robot_cfg)

    # Play simulation
    sim.reset()
    robot.update(sim.cfg.dt)

    # Get body info
    print("\n" + "=" * 70)
    print("Hero Agent Hydrodynamics Demo (Full Hydro on Base + Buoy)")
    print("=" * 70)
    print(f"\nBodies: {robot.body_names}")

    # Create hydrodynamics model for base (root body)
    hydro_cfg = HeroAgentHydrodynamicsCfg()
    hydro_model = HydrodynamicsModel(
        num_envs=1,
        device=sim.device,
        cfg=hydro_cfg,
        dt=dt,
    )

    # Create hydrodynamics model for buoy (link3)
    buoy_hydro_cfg = HeroAgentBuoyHydrodynamicsCfg()
    buoy_hydro_model = HydrodynamicsModel(
        num_envs=1,
        device=sim.device,
        cfg=buoy_hydro_cfg,
        dt=dt,
    )

    # Print configuration
    print("\n[Hydrodynamics Configuration]")
    print("  Base (full hydro):")
    print(f"    Mass: {hydro_cfg.vehicle_mass} kg")
    print(f"    Volume: {hydro_cfg.volume * 1000:.4f} L")
    buoyancy_base = WATER_DENSITY * hydro_cfg.volume * GRAVITY
    weight_base = hydro_cfg.vehicle_mass * GRAVITY
    print(f"    Buoyancy: {buoyancy_base:.2f} N")
    print(f"    Weight: {weight_base:.2f} N")
    print(f"    Net: {buoyancy_base - weight_base:+.2f} N")

    print("\n  Buoy/link3 (full hydro):")
    print(f"    Mass: {buoy_hydro_cfg.vehicle_mass} kg")
    print(f"    Volume: {buoy_hydro_cfg.volume * 1000:.4f} L")
    buoyancy_buoy = WATER_DENSITY * buoy_hydro_cfg.volume * GRAVITY
    weight_buoy = buoy_hydro_cfg.vehicle_mass * GRAVITY
    print(f"    Buoyancy: {buoyancy_buoy:.2f} N")
    print(f"    Weight: {weight_buoy:.2f} N")
    print(f"    Net: {buoyancy_buoy - weight_buoy:+.2f} N")

    print("\n  Other child bodies (simple buoyancy only):")
    total_net = (buoyancy_base - weight_base) + (buoyancy_buoy - weight_buoy)
    for name, params in CHILD_BODY_BUOYANCY_PARAMS.items():
        net = compute_buoyancy_force(params["mass"], params["volume"])
        total_net += net
        print(f"    {name:10s}: mass={params['mass']:.2f}kg, vol={params['volume']*1000:.4f}L, net={net:+.2f}N")

    print(f"\n  System Total Net Force: {total_net:+.2f} N")
    if abs(total_net) < 1.0:
        print("  Status: NEAR NEUTRAL BUOYANCY")
    elif total_net > 0:
        print("  Status: POSITIVE (system will float)")
    else:
        print("  Status: NEGATIVE (system will sink)")

    # Get body indices
    body_indices = {}
    for body_name in robot.body_names:
        idx = robot.find_bodies(body_name)[0][0]
        body_indices[body_name] = idx

    base_idx = body_indices["base"]
    buoy_idx = body_indices["link3"]
    num_bodies = len(robot.body_names)

    print(f"\n  Body indices: {body_indices}")

    print("\n" + "=" * 70)
    print("Starting simulation... Press Ctrl+C to exit.")
    print("Expected: buoy (link3) rises, base sinks, system rotates to equilibrium")
    print("=" * 70 + "\n")

    sim_time = 0.0
    step_count = 0

    while simulation_app.is_running() and sim_time < max_sim_time:
        # Update robot state
        robot.update(sim.cfg.dt)

        # Check for NaN early
        if torch.isnan(robot.data.root_pos_w).any():
            print(f"\n[t={sim_time:6.2f}s] *** NaN DETECTED - Simulation diverged ***")
            for name, idx in body_indices.items():
                pos = robot.data.body_pos_w[0, idx].cpu().numpy()
                print(f"  {name}: pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
            break

        # Initialize force/torque tensors for all bodies
        forces = torch.zeros(1, num_bodies, 3, device=sim.device)
        torques = torch.zeros(1, num_bodies, 3, device=sim.device)

        # === Step 1: Full hydrodynamics on ROOT BODY (base) ===
        # HydrodynamicsModel outputs forces in BODY FRAME
        # Includes: damping, Coriolis (C_RB + C_A), buoyancy
        # Note: Added mass (M_A * v_dot) is NOT applied as external force
        root_lin_vel = robot.data.root_lin_vel_w
        root_ang_vel = robot.data.root_ang_vel_w
        root_quat = robot.data.root_quat_w

        force_b, torque_b = hydro_model.compute_forces(root_lin_vel, root_ang_vel, root_quat)

        # Store body-frame forces for base
        forces[:, base_idx, :] = force_b
        torques[:, base_idx, :] = torque_b

        # === Step 2: Full hydrodynamics on BUOY (link3) ===
        # Same approach as RL environment - damping, Coriolis, buoyancy
        buoy_lin_vel = robot.data.body_lin_vel_w[:, buoy_idx, :]
        buoy_ang_vel = robot.data.body_ang_vel_w[:, buoy_idx, :]
        buoy_quat = robot.data.body_quat_w[:, buoy_idx, :]

        buoy_force_b, buoy_torque_b = buoy_hydro_model.compute_forces(
            buoy_lin_vel, buoy_ang_vel, buoy_quat
        )

        forces[:, buoy_idx, :] = buoy_force_b
        torques[:, buoy_idx, :] = buoy_torque_b

        # === Step 3: Simple buoyancy on OTHER CHILD BODIES (force only, no torque) ===
        # For gripper, link1, link2 - only vertical buoyancy force
        # This avoids fighting with joint constraints
        for body_name, params in CHILD_BODY_BUOYANCY_PARAMS.items():
            if body_name in body_indices:
                idx = body_indices[body_name]
                net_force = compute_buoyancy_force(params["mass"], params["volume"])

                # Get body orientation to transform world Z to body frame
                body_quat = robot.data.body_quat_w[:, idx, :]

                # World up direction
                world_up = torch.zeros(1, 3, device=sim.device)
                world_up[:, 2] = net_force

                # Transform world force to body frame
                force_body = quat_apply_inverse(body_quat, world_up)

                forces[:, idx, :] = force_body
                # No torques on these child bodies to avoid joint constraint conflicts

        # Apply forces using the modern wrench composer API
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            is_global=False,  # Forces are in body frame
        )
        robot.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim.cfg.dt
        step_count += 1

        # Print status every 0.5 seconds
        if step_count % 100 == 0:
            root_pos = robot.data.root_pos_w[0].cpu().numpy()
            root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
            buoy_pos = robot.data.body_pos_w[0, body_indices["link3"]].cpu().numpy()
            root_ang = robot.data.root_ang_vel_w[0].cpu().numpy()

            # Height difference (buoy relative to base)
            height_diff = buoy_pos[2] - root_pos[2]

            print(
                f"[t={sim_time:5.1f}s] base_z={root_pos[2]:+.3f} | buoy_z={buoy_pos[2]:+.3f} | "
                f"diff={height_diff:+.3f} | vel_z={root_vel[2]:+.3f} | ang={root_ang[0]:+.2f},{root_ang[1]:+.2f}",
                flush=True
            )

    # Final status
    print("\n" + "=" * 70)
    if sim_time >= max_sim_time:
        print("Simulation completed successfully!")
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        buoy_pos = robot.data.body_pos_w[0, body_indices["link3"]].cpu().numpy()
        print(f"Final base Z: {root_pos[2]:.3f}m")
        print(f"Final buoy Z: {buoy_pos[2]:.3f}m")
        print(f"Height difference: {buoy_pos[2] - root_pos[2]:+.3f}m")
    else:
        print("Simulation ended early (NaN or user interrupt)")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
