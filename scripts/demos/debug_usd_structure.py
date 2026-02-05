# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug USD structure to understand collision geometry paths."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug USD Structure")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_assets.robots.uuv import HERO_AGENT_CFG


def print_prim_tree(prim, indent=0):
    """Print USD prim tree structure."""
    prim_type = prim.GetTypeName()
    has_collision = UsdPhysics.CollisionAPI(prim) is not None and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()

    # Check for geometry
    geom_info = ""
    if prim_type == "Sphere":
        sphere = UsdGeom.Sphere(prim)
        r = sphere.GetRadiusAttr().Get()
        geom_info = f" (R={r})"
    elif prim_type == "Cylinder":
        cyl = UsdGeom.Cylinder(prim)
        r = cyl.GetRadiusAttr().Get()
        h = cyl.GetHeightAttr().Get()
        geom_info = f" (R={r}, H={h})"
    elif prim_type == "Cube":
        cube = UsdGeom.Cube(prim)
        s = cube.GetSizeAttr().Get()
        geom_info = f" (S={s})"
    elif prim_type == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points:
            geom_info = f" ({len(points)} verts)"

    collision_str = " [COLLISION]" if has_collision else ""
    print("  " * indent + f"{prim.GetName()} ({prim_type}){geom_info}{collision_str}")

    for child in prim.GetChildren():
        print_prim_tree(child, indent + 1)


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

    # Get stage
    from omni.usd import get_context
    stage = get_context().get_stage()

    print("\n" + "=" * 80)
    print("USD Prim Tree Structure")
    print("=" * 80)

    robot_prim = stage.GetPrimAtPath("/World/Robot")
    print_prim_tree(robot_prim)

    # Check specific body paths
    print("\n" + "=" * 80)
    print("Collision Geometry Analysis")
    print("=" * 80)

    body_names = ["base", "gripper", "link1", "link2", "link3"]

    for body_name in body_names:
        body_path = f"/World/Robot/{body_name}"
        body_prim = stage.GetPrimAtPath(body_path)

        print(f"\n--- {body_name} ---")
        print(f"Path: {body_path}")
        print(f"Valid: {body_prim.IsValid()}")

        if body_prim.IsValid():
            # Find all descendants with collision or geometry
            for desc in Usd.PrimRange(body_prim):
                desc_type = desc.GetTypeName()
                has_collision_api = UsdPhysics.CollisionAPI(desc)

                if has_collision_api or desc_type in ["Sphere", "Cylinder", "Cube", "Capsule", "Cone", "Mesh"]:
                    print(f"  Found: {desc.GetPath()} ({desc_type})")

                    # Check collision API
                    if has_collision_api:
                        enabled = has_collision_api.GetCollisionEnabledAttr().Get()
                        print(f"    CollisionAPI: enabled={enabled}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
