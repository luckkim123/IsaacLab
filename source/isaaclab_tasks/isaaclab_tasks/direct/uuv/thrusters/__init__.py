# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thruster models for UUV environments.

This module provides modular thruster system implementations following
Isaac Lab conventions. The design allows easy extension for different
thruster configurations and dynamics models.
"""

from .thruster_model import ThrusterModel, ThrusterModelCfg

__all__ = [
    "ThrusterModel",
    "ThrusterModelCfg",
]
