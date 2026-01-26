# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP (Markov Decision Process) functions for BlueROV environments.

This module provides event functions for domain randomization specific to
underwater vehicles, following Isaac Lab's EventTerm pattern.

These functions can be used with EventCfg to configure randomization events
that trigger on "startup" or "reset" modes.
"""

from .events import (
    randomize_hydrodynamics,
    randomize_ocean_current,
    randomize_robot_pose,
    randomize_thruster_params,
)

__all__ = [
    "randomize_hydrodynamics",
    "randomize_ocean_current",
    "randomize_robot_pose",
    "randomize_thruster_params",
]
