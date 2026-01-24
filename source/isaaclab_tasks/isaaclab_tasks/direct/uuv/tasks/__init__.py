# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task abstractions for UUV environments.

This module provides a task abstraction layer for underwater vehicle environments,
enabling pluggable task definitions without modifying the core environment logic.

The design follows Isaac Lab conventions while adapting to DirectRLEnv patterns.
"""

from .task_base import TaskBase, TaskBaseCfg
from .hover_task import HoverTask, HoverTaskCfg

__all__ = [
    "TaskBase",
    "TaskBaseCfg",
    "HoverTask",
    "HoverTaskCfg",
]
