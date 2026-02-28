# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for underwater vehicles (UUV).

This package provides robot configurations for underwater vehicles including
articulation settings, hydrodynamic parameters, and thruster configurations.
"""

from .uuv_cfg import HydrodynamicsCfg, OceanCurrentCfg, ThrusterCfg
from .bluerov import BLUEROV_CFG, BlueROVHydrodynamicsCfg, BlueROVThrusterCfg
from .hero_agent import (
    HERO_AGENT_ALBC_HEIGHT_OFFSET,
    HERO_AGENT_ALBC_JOINT_NAMES,
    HERO_AGENT_ALBC_LINK1_LENGTH,
    HERO_AGENT_ALBC_LINK2_LENGTH,
    HERO_AGENT_CFG,
    HERO_AGENT_USD_PATH,
    HeroAgentBuoyHydrodynamicsCfg,
    HeroAgentHydrodynamicsCfg,
)
