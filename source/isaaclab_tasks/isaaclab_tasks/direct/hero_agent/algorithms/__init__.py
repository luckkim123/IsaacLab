# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL algorithms for Hero Agent environments."""

from .ppo_patch import apply_ppo_patch

# Ensure PPO is patched with encoder support when this module is imported.
# This runs before any PPO instance is created (imported by rsl_rl_ppo_cfg.py).
apply_ppo_patch()

__all__ = ["apply_ppo_patch"]
