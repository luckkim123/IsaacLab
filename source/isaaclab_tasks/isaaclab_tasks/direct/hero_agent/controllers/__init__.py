# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controller modules for Hero Agent TDC integration.

This module provides:
    - ALBCKinematics: Forward/Inverse kinematics for 2-link ALBC arm
    - TDCController: Time Delay Control implementation
    - TDCControllerCfg: TDC configuration class
"""

from .kinematics import ALBCKinematics
from .tdc import TDCController, TDCControllerCfg

__all__ = [
    "ALBCKinematics",
    "TDCController",
    "TDCControllerCfg",
]
