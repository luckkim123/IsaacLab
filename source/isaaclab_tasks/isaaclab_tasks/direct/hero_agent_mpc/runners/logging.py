# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logging helpers for SAC-MPC runner.

Standalone functions that collect metrics from actor, env, and replay buffer
into a flat dict[str, float]. Called by SACMPCRunner._log().

Target: ~25 well-organized metrics in functional WandB sections:
    Reward/      - episode reward totals + per-term breakdown
    Loss/        - actor, critic, alpha, dynamics
    SAC/         - algorithm diagnostics (Q values, grad norms, alpha)
    Dynamics/    - prediction error, grad norm
    MPC/         - cost params, solve cost, tracking error
    Train/       - fps, episode length, buffer size
    Environment/ - attitude error, action stats
    DR/          - domain randomization (unchanged from base_env)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..encoder.actor_critic_mpc import ActorCriticMPC

# Remap base_env extras prefixes to WandB dashboard sections.
_PREFIX_REMAP: dict[str, str] = {
    "Episode_Reward/": "Reward/",
    "Episode_Termination/": "Train/",
    "Attitude_Error/": "Environment/attitude_error_",
    "Action/": "Environment/",
    "Dynamics/": "Environment/",
}


def collect_mpc_metrics(metrics: dict[str, float], actor: ActorCriticMPC, env) -> None:
    """Collect MPC-specific metrics: Q_diag aggregates, R_diag, solve_cost, state_err."""
    with torch.no_grad():
        obs = env.get_observations()
        policy_obs = obs[actor.policy_obs_key]

        raw_logits = actor.cost_map(policy_obs)
        Q_diag, _, R_diag = actor.decode_cost_params(raw_logits)

        metrics["MPC/Q_diag_mean"] = Q_diag.mean().item()
        metrics["MPC/R_diag_mean"] = R_diag.mean().item()

        # Aggregate Q breakdown: attitude (phi+theta), rate (p+q), joint (q1+q2)
        q_per_dim = Q_diag.mean(dim=(0, 1))
        if q_per_dim.shape[0] >= 6:
            metrics["MPC/Q_attitude"] = ((q_per_dim[0] + q_per_dim[1]) / 2).item()
            metrics["MPC/Q_rate"] = ((q_per_dim[2] + q_per_dim[3]) / 2).item()
            metrics["MPC/Q_joint"] = ((q_per_dim[4] + q_per_dim[5]) / 2).item()

        # MPC state error magnitude (tracking accuracy)
        mpc_state = obs.get("mpc_state")
        mpc_target = obs.get("mpc_target")
        if mpc_state is not None and mpc_target is not None:
            metrics["MPC/state_err_total"] = (mpc_state - mpc_target).abs().mean().item()

        # MPC solve cost from last rollout
        solve_cost = actor.get_last_solve_cost()
        if solve_cost is not None:
            metrics["MPC/solve_cost"] = solve_cost.mean().item()

        # Max |u| after differentiable refinement (verifies soft clamp)
        if actor._mpc_train is not None:
            metrics["MPC/u_max_after_refine"] = actor._mpc_train.last_u_max_after_refine


def _remap_key(key: str) -> str:
    """Remap base_env extras key to WandB dashboard section."""
    for old_prefix, new_prefix in _PREFIX_REMAP.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
    return key


def collect_reward_components(
    metrics: dict[str, float],
    ema_extras: dict[str, float],
) -> None:
    """Copy EMA-smoothed extras into metrics with prefix remapping.

    The runner maintains per-key EMA values updated in real-time as episodes
    complete, using adaptive alpha based on reset batch size.  This function
    simply applies the dashboard prefix remapping (e.g. ``Episode_Reward/`` ->
    ``Reward/``).
    """
    for key, value in ema_extras.items():
        metrics[_remap_key(key)] = value
