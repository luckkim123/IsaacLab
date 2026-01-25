#!/usr/bin/env python3
"""Simple script to view Agent USD in Isaac Sim."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import Gf, UsdGeom, UsdLux

USD_PATH = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/uuv/assets/Agent/Agent.usd"

def main():
    from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage

    create_new_stage()
    stage = omni.usd.get_context().get_stage()

    # Add dome light using USD API
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.GetIntensityAttr().Set(1000.0)

    # Add the Agent USD
    add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/Agent")

    print("=" * 50)
    print("Agent USD loaded!")
    print("Use mouse to navigate in viewport")
    print("Stage hierarchy -> select Agent -> press F to focus")
    print("=" * 50)

    while simulation_app.is_running():
        simulation_app.update()

if __name__ == "__main__":
    main()
    simulation_app.close()
