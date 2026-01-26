# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common physical models for simulation environments.

This module provides physics models that can be shared across multiple environments:
    - HydrodynamicsModel: Fossen model for 6-DOF hydrodynamic forces (UUV)
    - ThrusterModel: First-order thruster dynamics with allocation matrix (UUV)

Configuration classes are imported from isaaclab_assets.robots.uuv and
re-exported here for convenient access.
"""

from .hydrodynamics import HydrodynamicsModel, HydrodynamicsCfg, OceanCurrentCfg
from .thruster import ThrusterModel, ThrusterCfg

__all__ = [
    "HydrodynamicsModel",
    "HydrodynamicsCfg",
    "OceanCurrentCfg",
    "ThrusterModel",
    "ThrusterCfg",
]
