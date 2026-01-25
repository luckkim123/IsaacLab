# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for underwater vehicles (UUV).

This package provides robot configurations for underwater vehicles including
articulation settings, hydrodynamic parameters, and thruster configurations.
"""

from .uuv_cfg import HydrodynamicsCfg, OceanCurrentCfg, ThrusterCfg
from .bluerov import *
from .hero_agent import *
