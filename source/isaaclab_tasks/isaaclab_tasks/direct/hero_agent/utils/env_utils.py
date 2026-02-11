# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment utility functions for Hero Agent."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def unwrap_env(env: Any) -> Any:
    """Unwrap environment through wrapper chain to get the raw Isaac Lab env.

    Handles the common case where ``gym.Env.unwrapped`` returns ``self``
    (preventing infinite loops).

    Args:
        env: Potentially wrapped environment.

    Returns:
        The innermost (raw) environment.
    """
    raw = env
    while hasattr(raw, "unwrapped") and raw is not raw.unwrapped:
        raw = raw.unwrapped
    return raw


def connect_encoder_to_env(env: Any, policy: Any, caller_name: str = "Runner") -> None:
    """Wire encoder policy to environment for M_hat extraction via get_last_z().

    Args:
        env: Wrapped environment (will be unwrapped).
        policy: Policy with get_last_z() method.
        caller_name: Name for log message.
    """
    raw_env = unwrap_env(env)
    if hasattr(raw_env, "set_encoder_policy"):
        raw_env.set_encoder_policy(policy)
        logger.info("[%s] Connected encoder policy to env for M_hat extraction.", caller_name)
