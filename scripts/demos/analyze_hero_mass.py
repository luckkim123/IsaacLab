"""Analyze Hero Agent mass and buoyancy."""

from pxr import Usd, UsdPhysics
import math

usd_path = '/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/uuv/hero_agent/meshes/Agent.usd'
stage = Usd.Stage.Open(usd_path)

print("\n" + "=" * 60)
print("Hero Agent Mass & Buoyancy Analysis")
print("=" * 60)

total_mass = 0.0
print("\n[USD File Mass Properties]")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        if mass_attr.HasValue():
            mass = mass_attr.Get()
            if mass and mass > 0:
                print(f"  {prim.GetPath().name:20s}: {mass:8.4f} kg")
                total_mass += mass

print(f"\n  {'TOTAL':20s}: {total_mass:8.4f} kg")
print(f"  {'Total Weight':20s}: {total_mass * 9.81:8.2f} N")

# Current hydrodynamic config
print("\n" + "-" * 60)
print("[Current Hydrodynamic Config]")
print("-" * 60)

# Main body
main_mass = 9.18  # kg
main_volume = 0.00826  # m^3
rho = 997.0  # kg/m^3
g = 9.81  # m/s^2

main_buoyancy = rho * main_volume * g
main_weight = main_mass * g

print(f"\nMain Body (base):")
print(f"  Mass:     {main_mass:.2f} kg")
print(f"  Volume:   {main_volume:.5f} m^3")
print(f"  Buoyancy: {main_buoyancy:.2f} N")
print(f"  Weight:   {main_weight:.2f} N")
print(f"  Net:      {main_buoyancy - main_weight:+.2f} N {'(sinks)' if main_buoyancy < main_weight else '(floats)'}")

# Buoy
buoy_mass = 0.01  # kg
buoy_volume = 0.00136  # m^3

buoy_buoyancy = rho * buoy_volume * g
buoy_weight = buoy_mass * g

print(f"\nBuoy (link3):")
print(f"  Mass:     {buoy_mass:.2f} kg")
print(f"  Volume:   {buoy_volume:.5f} m^3")
print(f"  Buoyancy: {buoy_buoyancy:.2f} N")
print(f"  Weight:   {buoy_weight:.4f} N")
print(f"  Net:      {buoy_buoyancy - buoy_weight:+.2f} N {'(sinks)' if buoy_buoyancy < buoy_weight else '(floats)'}")

# Total
print("\n" + "-" * 60)
print("[TOTAL (Config)]")
print("-" * 60)
total_buoyancy = main_buoyancy + buoy_buoyancy
total_weight_config = main_weight + buoy_weight

print(f"  Total Buoyancy: {total_buoyancy:.2f} N")
print(f"  Total Weight:   {total_weight_config:.2f} N")
print(f"  Net Force:      {total_buoyancy - total_weight_config:+.2f} N")
print(f"  Result:         {'SINKS' if total_buoyancy < total_weight_config else 'FLOATS'}")

# With actual USD mass
print("\n" + "-" * 60)
print("[With Actual USD Mass]")
print("-" * 60)
actual_weight = total_mass * g
print(f"  Total Buoyancy: {total_buoyancy:.2f} N")
print(f"  Actual Weight:  {actual_weight:.2f} N")
print(f"  Net Force:      {total_buoyancy - actual_weight:+.2f} N")
print(f"  Result:         {'SINKS' if total_buoyancy < actual_weight else 'FLOATS'}")

# Recommendation
print("\n" + "=" * 60)
print("[Recommendation for Neutral Buoyancy]")
print("=" * 60)
neutral_mass = total_buoyancy / g
print(f"  Required total mass for neutral buoyancy: {neutral_mass:.2f} kg")
print(f"  Current config mass: {main_mass + buoy_mass:.2f} kg")
print(f"  Difference: {neutral_mass - (main_mass + buoy_mass):+.2f} kg")

# Or adjust volume
neutral_volume = (main_mass + buoy_mass) * g / (rho * g)
print(f"\n  OR Required total volume: {neutral_volume:.5f} m^3")
print(f"  Current volume: {main_volume + buoy_volume:.5f} m^3")
print(f"  Difference: {neutral_volume - (main_volume + buoy_volume):+.5f} m^3")

print("\n" + "=" * 60 + "\n")
