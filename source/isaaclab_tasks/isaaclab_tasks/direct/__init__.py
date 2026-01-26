# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Direct workflow environments.
"""

import gymnasium as gym

# Import UUV environments to register them with gymnasium
from . import bluerov  # noqa: F401
from . import hero_agent  # noqa: F401
