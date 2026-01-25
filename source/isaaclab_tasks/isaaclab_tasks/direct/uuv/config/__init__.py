# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration package for UUV environments.

This package organizes UUV environment configurations by robot type,
following Isaac Lab conventions for task/robot separation.

Structure:
    config/
        bluerov/       - BlueROV2 specific configurations
        hero_agent/    - Hero Agent specific configurations
"""

# Import robot-specific configurations to trigger gym.register()
from . import bluerov
from . import hero_agent
