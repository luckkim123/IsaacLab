# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple script to check mass values from PhysX."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check Mass Values")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.uuv import HERO_AGENT_CFG

GRAVITY = 9.81
WATER_DENSITY = 998.0

# Expected volumes from URDF collision geometry (manually calculated)
EXPECTED_VOLUMES = {
    "base": 3.14159 * 0.09**2 * 0.325,      # Cylinder R=0.09, L=0.325
    "gripper": 3.14159 * 0.015**2 * 0.325,  # Cylinder R=0.015, L=0.325
    "link1": 0.233 * 0.025 * 0.01,          # Box
    "link2": 0.233 * 0.025 * 0.01,          # Box
    "link3": 3.14159 * 0.085**2 * 0.09,     # Cylinder R=0.085, H=0.09
}


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)

    robot_cfg = HERO_AGENT_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.cfg.dt)

    body_names = robot.body_names
    body_masses = robot.root_physx_view.get_masses()[0]

    print("\n" + "=" * 90)
    print("Hero Agent Physics Values (Mass from PhysX, Volume from URDF specs)")
    print("=" * 90)

    print(f"\n{'Body':<12} {'Mass(kg)':<10} {'Weight(N)':<10} {'Volume(m3)':<12} {'Buoyancy(N)':<12} {'Net(N)':<10} {'Status'}")
    print("-" * 90)

    buoyancy_bodies = ["base", "link3"]  # Only these get buoyancy in HydrodynamicsModel
    total_mass = 0
    total_weight = 0
    total_buoyancy = 0
    total_net = 0

    for i, name in enumerate(body_names):
        mass = body_masses[i].item()
        weight = mass * GRAVITY
        volume = EXPECTED_VOLUMES.get(name, 0)

        if name in buoyancy_bodies:
            buoyancy = WATER_DENSITY * volume * GRAVITY
        else:
            buoyancy = 0.0

        net = buoyancy - weight

        total_mass += mass
        total_weight += weight
        total_buoyancy += buoyancy
        total_net += net

        status = "FLOAT" if net > 0.5 else ("SINK" if net < -0.5 else "NEUTRAL")
        buoy_mark = "*" if name in buoyancy_bodies else " "

        print(f"{name:<12} {mass:<10.3f} {weight:<10.2f} {volume:<12.6f} {buoyancy:<12.2f} {net:<+10.2f} {status}{buoy_mark}")

    print("-" * 90)
    print(f"{'TOTAL':<12} {total_mass:<10.3f} {total_weight:<10.2f} {'':<12} {total_buoyancy:<12.2f} {total_net:<+10.2f}")
    print("\n* = Buoyancy applied by HydrodynamicsModel")

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"Total Weight (all bodies):     {total_weight:.2f} N")
    print(f"Total Buoyancy (base + link3): {total_buoyancy:.2f} N")
    print(f"Net Force:                     {total_net:+.2f} N")

    if total_net > 0:
        print(f"\nSystem will FLOAT (acceleration = {total_net/total_mass:.3f} m/s^2)")
    else:
        print(f"\nSystem will SINK (acceleration = {abs(total_net)/total_mass:.3f} m/s^2)")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
