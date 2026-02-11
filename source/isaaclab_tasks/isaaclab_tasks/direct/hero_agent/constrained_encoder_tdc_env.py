# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Constrained Encoder-TDC Environment.

Extends HeroAgentEncoderTDCEnv with cost computation for NORBC constrained RL.
Computes per-step cost signals for K constraints and passes them to the
training pipeline via extras["costs"].

Key additions over HeroAgentEncoderTDCEnv:
    - CostManager for computing K cost signals
    - _prev_ee_pos, _prev_z: history buffers for smoothness/rate costs
    - Costs passed to runner via extras["costs"]
"""

from __future__ import annotations

import torch

from .config import HeroAgentConstrainedEncoderTDCEnvCfg
from .constraints.config import ConstraintCfg
from .constraints.cost_functions import (
    angular_velocity_cost,
    control_smoothness_cost,
    inertia_rate_cost,
    joint_velocity_cost,
    workspace_cost,
)
from .constraints.cost_manager import CostManager
from .encoder_tdc_env import HeroAgentEncoderTDCEnv

# Map constraint names to cost functions (config stores names, env binds functions)
_COST_FUNC_REGISTRY: dict = {
    "workspace": workspace_cost,
    "control_smoothness": control_smoothness_cost,
    "inertia_rate": inertia_rate_cost,
    "angular_velocity": angular_velocity_cost,
    "joint_velocity": joint_velocity_cost,
}


class HeroAgentConstrainedEncoderTDCEnv(HeroAgentEncoderTDCEnv):
    """Encoder-TDC environment with constrained RL cost computation.

    Adds K cost signals computed each step and passed via extras["costs"]
    to the ConstrainedPPO algorithm. The costs are:
        1. workspace: EE position exceeding reachable workspace (binary)
        2. control_smoothness: EE position change magnitude (continuous)
        3. inertia_rate: Encoder latent change rate (continuous)
        4. angular_velocity: Body angular velocity exceeding limit (binary)
        5. joint_velocity: Joint velocity exceeding motor limit (binary)
    """

    cfg: HeroAgentConstrainedEncoderTDCEnvCfg

    def __init__(self, cfg: HeroAgentConstrainedEncoderTDCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # History buffers for smoothness and rate costs
        self._prev_ee_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_z = None  # Set after first encoder forward pass
        self._last_z = None  # Cached z from encoder (updated in _pre_physics_step)

        # Bind cost functions to constraint configs at runtime
        # (config stores names + limits, env resolves functions)
        bound_cfgs = []
        for c in cfg.constrained_training.constraints:
            func = _COST_FUNC_REGISTRY.get(c.name)
            if func is None:
                raise ValueError(f"Unknown constraint name: {c.name}. Available: {list(_COST_FUNC_REGISTRY)}")
            bound_cfgs.append(ConstraintCfg(
                name=c.name,
                constraint_type=c.constraint_type,
                cost_func=func,
                limit_D=c.limit_D,
                params=c.params,
            ))

        # Cost manager
        self._cost_manager = CostManager(
            constraints=bound_cfgs,
            num_envs=self.num_envs,
            device=self.device,
        )
        # Store bound configs for algorithm access
        self._constraint_cfgs = bound_cfgs

    def _pre_physics_step(self, actions: torch.Tensor):
        """Extend parent to cache z for rate cost computation."""
        # Cache previous z before parent updates it
        if self._encoder_policy is not None:
            z = self._encoder_policy.get_last_z()
            if z is not None:
                self._prev_z = self._last_z
                self._last_z = z.clone()

        # Cache previous EE position before TDC updates it
        self._prev_ee_pos.copy_(self._tdc._p_EE_prev)

        # Run parent pipeline (updates TDC gains, M_hat, computes EE targets)
        super()._pre_physics_step(actions)

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards and costs.

        Costs are computed alongside rewards and stored in self.extras for
        the constrained PPO to pick up in process_env_step().
        """
        rewards = super()._get_rewards()

        # Compute per-step costs for all constraints
        costs = self._cost_manager.compute(self)

        # Store in extras for the algorithm to consume
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["costs"] = costs

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset including cost tracking.

        Order matters: super()._reset_idx() calls _collect_episode_metrics()
        which reads episode_sums, so cost_manager.reset() must happen AFTER.
        """
        # super() calls _collect_episode_metrics() -> reads episode_sums (still populated)
        super()._reset_idx(env_ids)

        # Now reset cost manager (after metrics have been collected)
        if env_ids is not None and len(env_ids) > 0:
            self._cost_manager.reset(env_ids)

        # Reset history buffers
        if env_ids is not None:
            self._prev_ee_pos[env_ids] = 0.0

    def _collect_episode_metrics(
        self,
        env_ids: torch.Tensor,
        reward_sums: dict[str, torch.Tensor],
    ) -> dict[str, float | torch.Tensor]:
        """Extend parent metrics with constraint diagnostics."""
        log = super()._collect_episode_metrics(env_ids, reward_sums)

        if len(env_ids) == 0:
            return log

        with torch.no_grad():
            # Episode length for mean computation
            ep_len = self.episode_length_buf[env_ids].float().clamp(min=1)

            # Per-constraint episode sums and means
            for name in self._cost_manager.constraint_names:
                ep_sum = self._cost_manager.episode_sums[name][env_ids]
                log[f"Cost/{name}_episode_sum"] = ep_sum.mean().item()
                log[f"Cost/{name}_episode_mean"] = (ep_sum / ep_len).mean().item()

        return log
