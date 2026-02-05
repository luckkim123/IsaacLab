# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify physics values (mass, volume, buoyancy) for Hero Agent.

This script loads the Hero Agent and prints the actual physics values
from PhysX and the computed buoyancy values from collision geometry.

Usage:
    ./isaaclab.sh -p scripts/demos/verify_physics_values.py
"""

from __future__ import annotations

import argparse

import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Verify Hero Agent Physics Values")
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
from isaaclab.utils.volume import compute_collision_volume

from isaaclab_assets.robots.uuv import HERO_AGENT_CFG


# Constants
WATER_DENSITY = 998.0  # kg/m^3
GRAVITY = 9.81  # m/s^2


def main():
    """Main function."""
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # Create ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)

    # Create Hero Agent
    robot_cfg = HERO_AGENT_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 2.0)
    robot = Articulation(robot_cfg)

    # Play simulation to initialize
    sim.reset()
    robot.update(sim.cfg.dt)

    # Get body names and indices
    print("\n" + "=" * 80)
    print("Hero Agent Physics Verification")
    print("=" * 80)

    body_names = robot.body_names
    print(f"\nBodies: {body_names}")

    # Get masses from PhysX
    body_masses = robot.root_physx_view.get_masses()[0]  # Shape: (num_bodies,)

    print("\n" + "-" * 80)
    print("1. Mass from PhysX (URDF -> USD -> PhysX)")
    print("-" * 80)
    print(f"{'Body':<12} {'Mass (kg)':<12} {'Weight (N)':<12}")
    print("-" * 40)

    total_mass = 0.0
    total_weight = 0.0
    mass_dict = {}

    for i, name in enumerate(body_names):
        mass = body_masses[i].item()
        weight = mass * GRAVITY
        mass_dict[name] = mass
        total_mass += mass
        total_weight += weight
        print(f"{name:<12} {mass:<12.3f} {weight:<12.2f}")

    print("-" * 40)
    print(f"{'Total':<12} {total_mass:<12.3f} {total_weight:<12.2f}")

    # Compute volumes from collision geometry
    print("\n" + "-" * 80)
    print("2. Volume from Collision Geometry (USD)")
    print("-" * 80)
    print(f"{'Body':<12} {'Volume (m^3)':<14} {'Volume (L)':<12} {'Buoyancy (N)':<14}")
    print("-" * 55)

    volume_dict = {}
    buoyancy_dict = {}

    for name in body_names:
        body_path = f"/World/Robot/{name}"
        try:
            volume = compute_collision_volume(body_path)
        except Exception as e:
            print(f"{name:<12} Error: {e}")
            volume = 0.0

        buoyancy = WATER_DENSITY * volume * GRAVITY
        volume_dict[name] = volume
        buoyancy_dict[name] = buoyancy
        print(f"{name:<12} {volume:<14.6f} {volume*1000:<12.4f} {buoyancy:<14.2f}")

    # Force balance analysis
    print("\n" + "-" * 80)
    print("3. Force Balance Analysis (Buoyancy applied to: base, link3 only)")
    print("-" * 80)
    print(f"{'Body':<12} {'Weight (N)':<12} {'Buoyancy (N)':<14} {'Net (N)':<12} {'Status':<15}")
    print("-" * 70)

    # Bodies that receive buoyancy in HydrodynamicsModel
    buoyancy_bodies = ["base", "link3"]

    total_weight_all = 0.0
    total_buoyancy_applied = 0.0
    total_net = 0.0

    for name in body_names:
        mass = mass_dict[name]
        weight = mass * GRAVITY
        total_weight_all += weight

        if name in buoyancy_bodies:
            buoyancy = buoyancy_dict[name]
            total_buoyancy_applied += buoyancy
        else:
            buoyancy = 0.0  # No buoyancy applied

        net = buoyancy - weight
        total_net += net

        if net > 0.5:
            status = "FLOATS"
        elif net < -0.5:
            status = "SINKS"
        else:
            status = "NEUTRAL"

        print(f"{name:<12} {weight:<12.2f} {buoyancy:<14.2f} {net:<+12.2f} {status:<15}")

    print("-" * 70)
    print(f"{'Total':<12} {total_weight_all:<12.2f} {total_buoyancy_applied:<14.2f} {total_net:<+12.2f}")

    # Summary
    print("\n" + "-" * 80)
    print("4. Summary")
    print("-" * 80)
    print(f"Total Mass (PhysX):        {total_mass:.3f} kg")
    print(f"Total Weight:              {total_weight_all:.2f} N")
    print(f"Total Buoyancy (applied):  {total_buoyancy_applied:.2f} N")
    print(f"Net Force:                 {total_net:+.2f} N")

    if abs(total_net) < 1.0:
        print("\nSystem Status: NEAR NEUTRAL BUOYANCY")
    elif total_net > 0:
        print(f"\nSystem Status: POSITIVE BUOYANCY (will float at {abs(total_net)/total_mass:.2f} m/s^2)")
    else:
        print(f"\nSystem Status: NEGATIVE BUOYANCY (will sink at {abs(total_net)/total_mass:.2f} m/s^2)")

    # Expected values comparison
    print("\n" + "-" * 80)
    print("5. Expected vs Actual Comparison")
    print("-" * 80)
    expected = {
        "base": {"mass": 9.18, "volume": 0.00827},
        "gripper": {"mass": 0.30, "volume": 0.00023},
        "link1": {"mass": 0.10, "volume": 0.000058},
        "link2": {"mass": 0.10, "volume": 0.000058},
        "link3": {"mass": 0.93, "volume": 0.00204},
    }

    print(f"{'Body':<12} {'Exp Mass':<10} {'Act Mass':<10} {'Exp Vol':<12} {'Act Vol':<12}")
    print("-" * 60)
    for name in body_names:
        if name in expected:
            exp_m = expected[name]["mass"]
            exp_v = expected[name]["volume"]
            act_m = mass_dict.get(name, 0)
            act_v = volume_dict.get(name, 0)
            print(f"{name:<12} {exp_m:<10.3f} {act_m:<10.3f} {exp_v:<12.6f} {act_v:<12.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
