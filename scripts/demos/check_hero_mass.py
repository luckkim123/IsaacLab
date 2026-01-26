"""Check actual Hero Agent mass from PhysX."""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.uuv import HERO_AGENT_CFG

# Create simulation
sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)

# Ground plane
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)

# Create robot
robot_cfg = HERO_AGENT_CFG.copy()
robot_cfg.prim_path = "/World/Robot"
robot = Articulation(robot_cfg)

sim.reset()
robot.update(sim.cfg.dt)

print("\n" + "=" * 60)
print("Hero Agent Actual Mass (from PhysX)")
print("=" * 60)

# Get mass from articulation data
body_names = robot.body_names
print(f"\nBodies: {body_names}")

# Try to get mass info
total_mass = 0.0
for i, name in enumerate(body_names):
    # Mass is available in root_physx_view
    pass

# Use articulation root mass
print(f"\nArticulation data available attributes:")
for attr in dir(robot.data):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Get from PhysX view
view = robot.root_physx_view
print(f"\nPhysX view attributes:")
for attr in dir(view):
    if 'mass' in attr.lower():
        print(f"  {attr}")

# Try to get masses
try:
    masses = view.get_masses()
    print(f"\nBody masses: {masses}")
    total_mass = masses.sum().item()
    print(f"Total mass: {total_mass:.4f} kg")
except Exception as e:
    print(f"Could not get masses: {e}")

# Try inertias
try:
    inertias = view.get_inertias()
    print(f"\nInertias shape: {inertias.shape}")
except Exception as e:
    print(f"Could not get inertias: {e}")

print("\n" + "=" * 60)

simulation_app.close()
