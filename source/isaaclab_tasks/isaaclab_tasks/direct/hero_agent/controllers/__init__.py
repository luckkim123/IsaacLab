# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controller modules for Hero Agent ALBC integration.

This module provides:
    - ALBCKinematics: Forward/Inverse kinematics for 2-link ALBC arm
    - TDCController: Time Delay Controller for roll/pitch attitude stabilization
    - TDCControllerCfg: Configuration for TDC controller
"""

from .kinematics import ALBCKinematics
from .tdc import TDCController, TDCControllerCfg, compute_M_bb

__all__ = [
    "ALBCKinematics",
    "TDCController",
    "TDCControllerCfg",
    "compute_M_bb",
]
