# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple USD structure check without full simulation."""

import sys
sys.stdout.flush()

print("=" * 60, flush=True)
print("Starting USD Analysis...", flush=True)
print("=" * 60, flush=True)

from pxr import Usd, UsdPhysics, UsdGeom

USD_PATH = "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/uuv/hero_agent/meshes/Agent.usd"

stage = Usd.Stage.Open(USD_PATH)
print(f"\nOpened USD: {USD_PATH}", flush=True)

print("\n[All Prims in Stage]", flush=True)
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    prim_type = prim.GetTypeName()
    print(f"  {prim_path} ({prim_type})", flush=True)

print("\n[Joint Properties]", flush=True)
for prim in stage.Traverse():
    ptype = prim.GetTypeName()
    ppath = str(prim.GetPath())
    if "Joint" in ptype or "joint" in ppath.lower():
        print(f"\n  Joint: {ppath} (type: {ptype})", flush=True)
        for attr in prim.GetAttributes():
            val = attr.Get()
            if val is not None:
                print(f"    {attr.GetName()}: {val}", flush=True)

print("\n[Mass Properties]", flush=True)
for prim in stage.Traverse():
    mass_api = UsdPhysics.MassAPI(prim)
    if mass_api:
        mass_attr = mass_api.GetMassAttr()
        if mass_attr.HasValue():
            print(f"  {prim.GetPath()}: mass = {mass_attr.Get()} kg", flush=True)
        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        if inertia_attr.HasValue():
            print(f"    diagonal_inertia = {inertia_attr.Get()}", flush=True)

print("\n[Articulation Root]", flush=True)
for prim in stage.Traverse():
    artic_api = UsdPhysics.ArticulationRootAPI(prim)
    if artic_api:
        print(f"  ArticulationRoot at: {prim.GetPath()}", flush=True)

print("\n" + "=" * 60, flush=True)
print("Analysis Complete", flush=True)
print("=" * 60, flush=True)
