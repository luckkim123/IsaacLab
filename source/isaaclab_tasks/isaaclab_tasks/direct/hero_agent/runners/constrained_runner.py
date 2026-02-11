# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ConstrainedEncoderRunner: OnPolicyRunner with ConstrainedPPO algorithm.

Extends OnPolicyRunner to:
    - Override _construct_algorithm() to create ConstrainedPPO instead of PPO
    - Pass constraint configs from the environment to the algorithm
    - Log per-constraint metrics (cost returns, barrier loss, adaptive thresholds)
    - Include encoder metrics logging (from EncoderRunner pattern)
"""

from __future__ import annotations

import logging

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from ..constraints import ConstrainedPPO
from ..encoder.actor_critic_constrained import ActorCriticConstrained  # noqa: F401 - used by eval()
from ..utils.env_utils import connect_encoder_to_env, unwrap_env
from ..utils.logging import flush_metrics, log_encoder_metrics, log_encoder_tdc_metrics

logger = logging.getLogger(__name__)


class ConstrainedEncoderRunner(OnPolicyRunner):
    """OnPolicyRunner with ConstrainedPPO and encoder metrics logging.

    Overrides _construct_algorithm() so that ConstrainedPPO is used instead
    of standard PPO. The constraint configs are read from the unwrapped
    environment's _constraint_cfgs and constrained_training config.
    """

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct ConstrainedPPO with constraint configs from the environment.

        The standard OnPolicyRunner._construct_algorithm() creates a plain PPO.
        This override creates a ConstrainedPPO that includes barrier loss and
        multi-head cost value function training.
        """
        # Build the policy network (same as parent)
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Get constraint configs from the unwrapped environment
        raw_env = unwrap_env(self.env)
        constraint_cfgs = getattr(raw_env, "_constraint_cfgs", [])
        constrained_cfg = getattr(raw_env.cfg, "constrained_training", None)

        if constraint_cfgs and constrained_cfg is not None:
            logger.info(
                "[ConstrainedEncoderRunner] Creating ConstrainedPPO with %d constraints.",
                len(constraint_cfgs),
            )
            # Pop class_name from alg_cfg (PPO doesn't use it)
            alg_cfg = dict(self.alg_cfg)
            alg_cfg.pop("class_name", None)

            alg = ConstrainedPPO(
                actor_critic,
                constraint_cfgs=constraint_cfgs,
                constrained_cfg=constrained_cfg,
                device=self.device,
                **alg_cfg,
                multi_gpu_cfg=self.multi_gpu_cfg,
            )
        else:
            logger.warning(
                "[ConstrainedEncoderRunner] No constraint configs found. Falling back to standard PPO."
            )
            alg_cfg = dict(self.alg_cfg)
            alg_cfg.pop("class_name", None)
            alg = PPO(actor_critic, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._has_encoder = hasattr(self.alg.policy, "encoder")
        self._has_constraints = isinstance(self.alg, ConstrainedPPO)

        if self._has_encoder:
            logger.info("[ConstrainedEncoderRunner] Encoder detected. Metrics logging enabled.")
            connect_encoder_to_env(self.env, self.alg.policy, "ConstrainedEncoderRunner")

        if self._has_constraints:
            logger.info(
                "[ConstrainedEncoderRunner] ConstrainedPPO active with %d constraints.",
                self.alg.num_constraints,
            )

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with encoder + constraint metrics."""
        # Update reward curriculum
        iteration = locs["it"]
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "_reward_manager") and hasattr(raw_env.cfg, "reward"):
            raw_env._reward_manager.update_curriculum(iteration, raw_env.cfg.reward.curriculum_end_iter)

        # Parent log (standard PPO metrics)
        super().log(locs, width, pad)

        if self.log_dir is None or self.disable_logs:
            return

        # Encoder metrics
        if self._has_encoder:
            log_encoder_metrics(
                writer=self.writer,
                policy=self.alg.policy,
                env=self.env,
                iteration=iteration,
                device=self.device,
                logger_type=self.logger_type,
            )
            log_encoder_tdc_metrics(
                writer=self.writer,
                policy=self.alg.policy,
                env=self.env,
                iteration=iteration,
                device=self.device,
                logger_type=self.logger_type,
            )

        # Constraint metrics
        if self._has_constraints:
            self._log_constraint_metrics(iteration, locs)

    def _log_constraint_metrics(self, iteration: int, locs: dict) -> None:
        """Log barrier, adaptive thresholds, and cost return metrics."""
        metrics: dict[str, float] = {}

        # Barrier manager metrics (adaptive limits, slack ratios)
        barrier = self.alg.barrier
        metrics.update(barrier.get_metrics())

        # Cost returns from last rollout (if storage has data)
        storage = self.alg.storage
        if hasattr(storage, "cost_returns") and storage.num_constraints > 0:
            cost_returns = storage.cost_returns  # (T, N, K)
            constraint_cfgs = self.alg.constraint_cfgs
            for k, name in enumerate(barrier.constraint_names):
                mean_cr = cost_returns[:, :, k].mean().item()
                metrics[f"Constraint/{name}_cost_return"] = mean_cr

                if constraint_cfgs[k].constraint_type == "probabilistic":
                    # Binary cost: fraction of steps with violation
                    costs = storage.costs[:, :, k]  # (T, N)
                    metrics[f"Constraint/{name}_violation_rate"] = (costs > 0.5).float().mean().item()
                else:
                    # Average cost: fraction of envs whose cost return exceeds d_k
                    per_env_cr = cost_returns[:, :, k].mean(dim=0)  # (N,)
                    d_k = barrier.original_limits[k].item()
                    metrics[f"Constraint/{name}_violation_rate"] = (per_env_cr > d_k).float().mean().item()

        # Barrier/cost losses from last update (available via locals() in learn())
        loss_dict = locs.get("loss_dict", {})
        if "barrier" in loss_dict:
            metrics["Loss/barrier"] = loss_dict["barrier"]
        if "cost_value" in loss_dict:
            metrics["Loss/cost_value"] = loss_dict["cost_value"]

        flush_metrics(self.writer, metrics, iteration, self.logger_type)
