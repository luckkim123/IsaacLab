# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to check USD physics properties for Hero Agent."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check USD Physics Properties")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import pxr
from pxr import Usd, UsdPhysics

USD_PATH = "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/uuv/hero_agent/meshes/Agent.usd"

print("=" * 60)
print("Hero Agent USD Physics Properties Analysis")
print("=" * 60)

stage = Usd.Stage.Open(USD_PATH)

print("\n[All Prims in Stage]")
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    prim_type = prim.GetTypeName()
    print(f"  {prim_path} ({prim_type})")

print("\n[Joint Properties]")
for prim in stage.Traverse():
    if "Joint" in prim.GetTypeName() or "joint" in str(prim.GetPath()).lower():
        print(f"\n  Joint: {prim.GetPath()} (type: {prim.GetTypeName()})")
        for attr in prim.GetAttributes():
            val = attr.Get()
            if val is not None:
                print(f"    {attr.GetName()}: {val}")

print("\n[Mass Properties]")
for prim in stage.Traverse():
    mass_api = UsdPhysics.MassAPI(prim)
    if mass_api:
        mass_attr = mass_api.GetMassAttr()
        if mass_attr.HasValue():
            print(f"  {prim.GetPath()}: mass = {mass_attr.Get()} kg")

        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        if inertia_attr.HasValue():
            print(f"    diagonal_inertia = {inertia_attr.Get()}")

print("\n[Rigid Body Properties]")
for prim in stage.Traverse():
    rb_api = UsdPhysics.RigidBodyAPI(prim)
    if rb_api:
        print(f"  {prim.GetPath()}")
        for attr in prim.GetAttributes():
            if "physx" in attr.GetName().lower() or "rigid" in attr.GetName().lower():
                val = attr.Get()
                if val is not None:
                    print(f"    {attr.GetName()}: {val}")

print("\n" + "=" * 60)

simulation_app.close()
