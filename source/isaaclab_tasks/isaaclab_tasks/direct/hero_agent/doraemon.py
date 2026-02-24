# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DORAEMON: Domain Randomization with Entropy Maximization (ICLR 2024).

Replaces linear DR curriculum with adaptive Beta distribution scheduling.
Maximizes DR entropy subject to policy success rate >= alpha.
When the constraint is infeasible, backs up toward higher success rate.

Reference:
    Tiboni et al., "Domain Randomization via Entropy Maximization", ICLR 2024.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
from scipy.optimize import NonlinearConstraint, minimize

from isaaclab.utils import configclass

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@configclass
class DoraemonCfg:
    """DORAEMON scheduler configuration."""

    enable: bool = True
    alpha: float = 0.5
    """Success rate threshold. Distribution expands only when success >= alpha."""

    kl_ub: float = 0.01
    """Trust region KL divergence upper bound per step."""

    init_concentration: float = 50.0
    """Initial Beta(a, b) concentration (a + b). Higher = tighter initial distribution."""

    success_threshold_deg: float = 15.0
    """Attitude error below this (deg) counts as success. Evaluated over settling window.
    Anneals to success_threshold_deg_final over success_threshold_anneal_steps."""

    success_threshold_deg_final: float = 5.0
    """Final success threshold (deg) after annealing completes."""

    success_threshold_anneal_steps: int = 500
    """Number of DORAEMON steps over which threshold anneals from start to final."""

    buffer_size: int = 2000
    """Maximum episode buffer capacity."""

    min_episodes: int = 200
    """Minimum episodes before first DORAEMON update."""


# =============================================================================
# Parameter Specification
# =============================================================================


class ParamSpec(NamedTuple):
    """Single DR parameter specification."""

    name: str
    min_bound: float
    max_bound: float
    nominal: float


# 14 DORAEMON-managed DR parameters.
# Order matches BetaDistribution dimension indices.
PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("inertia_scale", 0.5, 2.0, 1.0),
    ParamSpec("body_mass_scale", 0.7, 1.3, 1.0),
    ParamSpec("volume_scale", 0.7, 1.3, 1.0),
    ParamSpec("added_mass_scale", 0.5, 2.0, 1.0),
    ParamSpec("linear_damping_scale", 0.5, 2.0, 1.0),
    ParamSpec("quadratic_damping_scale", 0.3, 2.0, 1.0),
    ParamSpec("water_density", 990.0, 1030.0, 1000.0),
    ParamSpec("cog_offset_z", -0.04, 0.04, 0.0),
    ParamSpec("cob_offset_z", -0.04, 0.04, 0.0),
    ParamSpec("joint_stiffness", 40.0, 200.0, 100.0),
    ParamSpec("joint_damping", 1.0, 6.0, 3.0),
    ParamSpec("joint_static_friction", 0.0, 0.1, 0.0),
    ParamSpec("joint_viscous_friction", 0.0, 0.5, 0.0),
    ParamSpec("payload_mass", 0.0, 3.0, 0.0),
]

NDIMS = len(PARAM_SPECS)

# Minimum Beta parameter value to keep distribution well-defined.
_MIN_BETA_PARAM = 1.0
_MAX_BETA_PARAM = 500.0


def _compute_kl(flat_new: np.ndarray, flat_prev: np.ndarray) -> float:
    """Compute KL(new || prev) for independent Beta distributions.

    Args:
        flat_new: [a0, b0, a1, b1, ...] new distribution parameters.
        flat_prev: [a0, b0, a1, b1, ...] previous distribution parameters.

    Returns:
        Sum of per-dimension KL divergences.
    """
    a_b_new = torch.from_numpy(flat_new.copy()).reshape(-1, 2).double()
    a_b_prev = torch.from_numpy(flat_prev.copy()).reshape(-1, 2).double()
    new = torch.distributions.Beta(
        a_b_new[:, 0].clamp(min=_MIN_BETA_PARAM),
        a_b_new[:, 1].clamp(min=_MIN_BETA_PARAM),
    )
    prev = torch.distributions.Beta(
        a_b_prev[:, 0].clamp(min=_MIN_BETA_PARAM),
        a_b_prev[:, 1].clamp(min=_MIN_BETA_PARAM),
    )
    return torch.distributions.kl_divergence(new, prev).sum().item()


# =============================================================================
# Beta Distribution
# =============================================================================


class BetaDistribution:
    """Independent Beta distributions over DR parameter space.

    Each dimension i has Beta(a_i, b_i) on [0, 1], linearly mapped to
    [min_bound_i, max_bound_i] for physical sampling.
    """

    def __init__(self, params: list[ParamSpec], device: torch.device, concentration: float = 200.0) -> None:
        self.params = params
        self.ndims = len(params)
        self.device = device

        # Physical bounds: (ndims,)
        self._mins = torch.tensor([p.min_bound for p in params], dtype=torch.float64)
        self._maxs = torch.tensor([p.max_bound for p in params], dtype=torch.float64)
        self._ranges = self._maxs - self._mins

        # Initialize Beta(a, b) from nominal + concentration
        self._a = torch.zeros(self.ndims, dtype=torch.float64)
        self._b = torch.zeros(self.ndims, dtype=torch.float64)

        for i, p in enumerate(params):
            # Map nominal to [0, 1]
            mu = (p.nominal - p.min_bound) / (p.max_bound - p.min_bound)
            mu = max(0.01, min(0.99, mu))  # Avoid boundary degeneration
            self._a[i] = max(_MIN_BETA_PARAM, mu * concentration)
            self._b[i] = max(_MIN_BETA_PARAM, (1.0 - mu) * concentration)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n parameter vectors and their log probabilities.

        Returns:
            xi_physical: (n, ndims) physical-scale samples on GPU.
            log_probs: (n,) sum of per-dim log probs on GPU.
        """
        a_gpu = self._a.float().to(self.device)
        b_gpu = self._b.float().to(self.device)
        dist = torch.distributions.Beta(a_gpu, b_gpu)

        # Sample in [0, 1] space
        xi_unit = dist.sample((n,))  # (n, ndims)
        log_probs = dist.log_prob(xi_unit).sum(dim=-1)  # (n,)

        # Map to physical space
        mins_gpu = self._mins.float().to(self.device)
        ranges_gpu = self._ranges.float().to(self.device)
        xi_physical = mins_gpu + xi_unit * ranges_gpu

        return xi_physical, log_probs

    def log_prob(self, xi_physical: torch.Tensor) -> torch.Tensor:
        """Compute log probability of physical-scale samples.

        Args:
            xi_physical: (n, ndims) physical-scale values.

        Returns:
            (n,) sum of per-dim log probs.
        """
        mins_gpu = self._mins.float().to(self.device)
        ranges_gpu = self._ranges.float().to(self.device)
        xi_unit = (xi_physical - mins_gpu) / ranges_gpu
        xi_unit = xi_unit.clamp(1e-6, 1.0 - 1e-6)

        a_gpu = self._a.float().to(self.device)
        b_gpu = self._b.float().to(self.device)
        dist = torch.distributions.Beta(a_gpu, b_gpu)
        return dist.log_prob(xi_unit).sum(dim=-1)

    def entropy(self) -> float:
        """Total entropy: sum of per-dim Beta entropies + log(range) scaling."""
        dist = torch.distributions.Beta(self._a, self._b)
        return (dist.entropy() + self._ranges.log()).sum().item()

    def kl_divergence(self, other: BetaDistribution) -> float:
        """KL(self || other): sum of per-dim KL divergences."""
        p = torch.distributions.Beta(self._a, self._b)
        q = torch.distributions.Beta(other._a, other._b)
        return torch.distributions.kl_divergence(p, q).sum().item()

    def get_flat_params(self) -> np.ndarray:
        """Return [a0, b0, a1, b1, ...] as float64 numpy for scipy."""
        flat = torch.stack([self._a, self._b], dim=-1).flatten()
        return flat.numpy()

    def set_flat_params(self, flat: np.ndarray) -> None:
        """Update from scipy result: [a0, b0, a1, b1, ...]."""
        t = torch.from_numpy(flat.copy()).reshape(self.ndims, 2).double()
        self._a = t[:, 0].clamp(min=_MIN_BETA_PARAM, max=_MAX_BETA_PARAM)
        self._b = t[:, 1].clamp(min=_MIN_BETA_PARAM, max=_MAX_BETA_PARAM)

    def clone(self) -> BetaDistribution:
        """Deep copy."""
        new = BetaDistribution.__new__(BetaDistribution)
        new.params = self.params
        new.ndims = self.ndims
        new.device = self.device
        new._mins = self._mins.clone()
        new._maxs = self._maxs.clone()
        new._ranges = self._ranges.clone()
        new._a = self._a.clone()
        new._b = self._b.clone()
        return new

    def get_stats(self) -> dict[str, float]:
        """Return per-dimension mean and std in physical space for logging."""
        stats = {}
        for i, p in enumerate(self.params):
            a, b = self._a[i].item(), self._b[i].item()
            mean_unit = a / (a + b)
            var_unit = (a * b) / ((a + b) ** 2 * (a + b + 1))
            std_unit = var_unit**0.5
            mean_phys = p.min_bound + mean_unit * (p.max_bound - p.min_bound)
            std_phys = std_unit * (p.max_bound - p.min_bound)
            stats[f"mean/{p.name}"] = mean_phys
            stats[f"std/{p.name}"] = std_phys
        return stats


# =============================================================================
# Episode Buffer
# =============================================================================


@dataclass
class EpisodeBuffer:
    """Ring buffer for completed episode statistics."""

    capacity: int
    ndims: int
    device: torch.device

    xi: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)
    success: torch.Tensor = field(init=False)
    log_probs: torch.Tensor = field(init=False)
    _count: int = field(init=False, default=0)
    _write_idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.xi = torch.zeros(self.capacity, self.ndims, device=self.device)
        self.returns = torch.zeros(self.capacity, device=self.device)
        self.success = torch.zeros(self.capacity, device=self.device)
        self.log_probs = torch.zeros(self.capacity, device=self.device)

    def add(
        self,
        xi: torch.Tensor,
        returns: torch.Tensor,
        success: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """Batch insert episodes using vectorized indexing. Wraps around if capacity exceeded."""
        n = xi.shape[0]
        if n == 0:
            return
        # If batch exceeds capacity, only keep the last `capacity` entries
        if n > self.capacity:
            tail = n - self.capacity
            xi = xi[tail:]
            returns = returns[tail:]
            success = success[tail:]
            log_probs = log_probs[tail:]
            n = self.capacity
        # Compute destination indices (handles wrap-around)
        start = self._write_idx % self.capacity
        if start + n <= self.capacity:
            # No wrap: single contiguous slice
            self.xi[start : start + n] = xi
            self.returns[start : start + n] = returns
            self.success[start : start + n] = success
            self.log_probs[start : start + n] = log_probs
        else:
            # Wrap-around: two slices
            first = self.capacity - start
            self.xi[start:] = xi[:first]
            self.returns[start:] = returns[:first]
            self.success[start:] = success[:first]
            self.log_probs[start:] = log_probs[:first]
            self.xi[: n - first] = xi[first:]
            self.returns[: n - first] = returns[first:]
            self.success[: n - first] = success[first:]
            self.log_probs[: n - first] = log_probs[first:]
        self._write_idx += n
        self._count = min(self._count + n, self.capacity)

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return valid entries."""
        n = self._count
        return self.xi[:n], self.returns[:n], self.success[:n], self.log_probs[:n]

    def clear(self) -> None:
        """Reset buffer."""
        self._count = 0
        self._write_idx = 0


# =============================================================================
# DORAEMON Scheduler
# =============================================================================


class DoraemonScheduler:
    """DORAEMON DR distribution scheduler.

    Maintains a Beta distribution over DR parameters, collects episode statistics,
    and optimizes the distribution to maximize entropy subject to success rate >= alpha.
    """

    def __init__(self, cfg: DoraemonCfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        self.dist = BetaDistribution(PARAM_SPECS, device, cfg.init_concentration)

        self.buffer = EpisodeBuffer(cfg.buffer_size, NDIMS, device)

        self._step_count = 0
        self._backup_count = 0
        self._total_episodes = 0

        # Success threshold annealing state
        self._current_threshold_deg = cfg.success_threshold_deg

        logger.info(
            "[DORAEMON] Initialized: alpha=%.2f, kl_ub=%.4f, %d parameters, "
            "concentration=%.0f, threshold=%.1f->%.1f deg over %d steps",
            cfg.alpha,
            cfg.kl_ub,
            NDIMS,
            cfg.init_concentration,
            cfg.success_threshold_deg,
            cfg.success_threshold_deg_final,
            cfg.success_threshold_anneal_steps,
        )

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample DR parameters from current Beta distribution.

        Returns:
            xi_physical: (n, ndims) physical-scale samples.
            log_probs: (n,) log probabilities.
        """
        return self.dist.sample(n)

    def record_episodes(
        self,
        xi: torch.Tensor,
        returns: torch.Tensor,
        success: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """Record completed episodes into the buffer."""
        self.buffer.add(xi, returns, success, log_probs)
        self._total_episodes += xi.shape[0]

    def step(self) -> dict[str, float]:
        """Run one DORAEMON optimization step.

        Returns:
            Metrics dict for logging.
        """
        xi, _returns, success, _log_probs = self.buffer.get_all()
        n = xi.shape[0]

        metrics: dict[str, float] = {}
        metrics["buffer_size"] = float(n)
        metrics["total_episodes"] = float(self._total_episodes)

        if n < self.cfg.min_episodes:
            metrics["skipped"] = 1.0
            metrics["entropy"] = self.dist.entropy()
            metrics["success_rate"] = success.mean().item() if n > 0 else 0.0
            return metrics

        # Anneal success threshold
        self._anneal_threshold()
        metrics["success_threshold_deg"] = self._current_threshold_deg

        success_rate = success.mean().item()
        metrics["success_rate"] = success_rate
        metrics["entropy_before"] = self.dist.entropy()

        # Save current distribution for trust region
        prev_dist = self.dist.clone()

        if success_rate < self.cfg.alpha:
            # Infeasible: backup toward higher success
            self._backup(prev_dist, xi, success)
            # Retry entropy maximization using backup result as starting point
            backup_success = self._estimate_success_rate(xi, success, prev_dist)
            if backup_success >= self.cfg.alpha:
                self._maximize_entropy(self.dist.clone(), xi, success)
                metrics["mode"] = 0.5  # backup-then-expand
            else:
                metrics["mode"] = 0.0  # backup only
            self._backup_count += 1
        else:
            # Feasible: maximize entropy subject to success >= alpha
            self._maximize_entropy(prev_dist, xi, success)
            metrics["mode"] = 1.0  # expand

        metrics["entropy_after"] = self.dist.entropy()
        metrics["kl_step"] = self.dist.kl_divergence(prev_dist)
        metrics["backup_count"] = float(self._backup_count)

        # Per-parameter distribution stats
        param_stats = self.dist.get_stats()
        for k, v in param_stats.items():
            metrics[k] = v

        self.buffer.clear()
        self._step_count += 1
        return metrics

    def _anneal_threshold(self) -> None:
        """Linearly anneal success threshold from start to final value."""
        cfg = self.cfg
        if cfg.success_threshold_anneal_steps <= 0:
            self._current_threshold_deg = cfg.success_threshold_deg_final
            return
        t = min(1.0, self._step_count / cfg.success_threshold_anneal_steps)
        self._current_threshold_deg = cfg.success_threshold_deg + t * (
            cfg.success_threshold_deg_final - cfg.success_threshold_deg
        )

    def _estimate_success_rate(
        self,
        xi: torch.Tensor,
        success: torch.Tensor,
        ref_dist: BetaDistribution,
    ) -> float:
        """Estimate success rate under current dist via IS from ref_dist."""
        new_lp = self.dist.log_prob(xi)
        old_lp = ref_dist.log_prob(xi)
        log_ratio = new_lp - old_lp
        weights = torch.exp(log_ratio - log_ratio.max())
        weights = weights / weights.sum()
        return (weights * success).sum().item()

    def _maximize_entropy(
        self,
        prev_dist: BetaDistribution,
        xi: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        """Maximize entropy subject to success >= alpha and trust region.

        Uses trust-constr optimizer with keep_feasible=True to guarantee
        KL constraint satisfaction at every iteration.
        """
        x0 = self.dist.get_flat_params()
        prev_flat = prev_dist.get_flat_params()
        ranges = self.dist._ranges
        mins = self.dist._mins

        xi_cpu = xi.detach().cpu().numpy().astype(np.float64)
        success_cpu = success.detach().cpu().numpy().astype(np.float64)

        def objective_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
            """Negative entropy (to minimize) with gradient via autograd."""
            a_b = torch.from_numpy(flat.copy()).reshape(NDIMS, 2).double().requires_grad_(True)
            a = a_b[:, 0].clamp(min=_MIN_BETA_PARAM)
            b = a_b[:, 1].clamp(min=_MIN_BETA_PARAM)
            dist = torch.distributions.Beta(a, b)
            entropy = (dist.entropy() + ranges.log()).sum()
            neg_entropy = -entropy
            neg_entropy.backward()
            assert a_b.grad is not None
            return neg_entropy.item(), a_b.grad.flatten().numpy().copy()

        def success_constraint_fun(flat: np.ndarray) -> float:
            """IS-weighted success rate under new distribution."""
            a_b = torch.from_numpy(flat.copy()).reshape(NDIMS, 2).double()
            a_new = a_b[:, 0].clamp(min=_MIN_BETA_PARAM)
            b_new = a_b[:, 1].clamp(min=_MIN_BETA_PARAM)

            xi_t = torch.from_numpy(xi_cpu).double()
            xi_unit = ((xi_t - mins) / ranges).clamp(1e-6, 1 - 1e-6)
            new_dist = torch.distributions.Beta(a_new, b_new)
            new_lp = new_dist.log_prob(xi_unit).sum(dim=-1)

            old_a_b = torch.from_numpy(prev_flat.copy()).reshape(NDIMS, 2).double()
            old_dist = torch.distributions.Beta(old_a_b[:, 0].clamp(min=1.0), old_a_b[:, 1].clamp(min=1.0))
            old_lp = old_dist.log_prob(xi_unit).sum(dim=-1)

            log_ratio = new_lp - old_lp
            weights = torch.exp(log_ratio - log_ratio.max())
            weights = weights / weights.sum()
            success_t = torch.from_numpy(success_cpu).double()
            return (weights * success_t).sum().item()

        def kl_constraint_fun(flat: np.ndarray) -> float:
            """KL(new || prev)."""
            return _compute_kl(flat, prev_flat)

        bounds = [(float(_MIN_BETA_PARAM), float(_MAX_BETA_PARAM))] * (2 * NDIMS)

        success_con = NonlinearConstraint(success_constraint_fun, lb=self.cfg.alpha, ub=np.inf)
        kl_con = NonlinearConstraint(kl_constraint_fun, lb=0.0, ub=self.cfg.kl_ub, keep_feasible=True)

        try:
            result = minimize(
                objective_and_grad,
                x0,
                method="trust-constr",
                jac=True,
                bounds=bounds,
                constraints=[success_con, kl_con],
                options={"maxiter": 50, "gtol": 1e-8},
            )
            if result.success or result.fun < objective_and_grad(x0)[0]:
                self.dist.set_flat_params(result.x)
        except Exception as e:
            logger.warning("[DORAEMON] Entropy maximization failed: %s", e)

    def _backup(
        self,
        prev_dist: BetaDistribution,
        xi: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        """Backup: maximize IS-weighted success rate within trust region.

        Uses trust-constr with keep_feasible=True for KL constraint.
        """
        # Guard: if no successful episodes, objective is constant-zero everywhere.
        # The optimizer would declare trivial convergence and accept arbitrary drift
        # within the KL trust region, causing unintended distribution expansion.
        if success.sum().item() < 1.0:
            logger.debug("[DORAEMON] Backup skipped: no successful episodes in buffer.")
            return

        x0 = self.dist.get_flat_params()
        prev_flat = prev_dist.get_flat_params()
        mins = self.dist._mins
        ranges = self.dist._ranges

        xi_cpu = xi.detach().cpu().numpy().astype(np.float64)
        success_cpu = success.detach().cpu().numpy().astype(np.float64)

        def neg_success_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
            """Negative IS-weighted success rate with gradient."""
            a_b = torch.from_numpy(flat.copy()).reshape(NDIMS, 2).double().requires_grad_(True)
            a = a_b[:, 0].clamp(min=_MIN_BETA_PARAM)
            b = a_b[:, 1].clamp(min=_MIN_BETA_PARAM)

            xi_t = torch.from_numpy(xi_cpu).double()
            xi_unit = ((xi_t - mins) / ranges).clamp(1e-6, 1 - 1e-6)

            new_dist = torch.distributions.Beta(a, b)
            new_lp = new_dist.log_prob(xi_unit).sum(dim=-1)

            old_a_b = torch.from_numpy(prev_flat.copy()).reshape(NDIMS, 2).double()
            old_dist = torch.distributions.Beta(old_a_b[:, 0].clamp(min=1.0), old_a_b[:, 1].clamp(min=1.0))
            old_lp = old_dist.log_prob(xi_unit).sum(dim=-1)

            log_ratio = new_lp - old_lp
            weights = torch.exp(log_ratio - log_ratio.max().detach())
            weights = weights / weights.sum().detach()

            success_t = torch.from_numpy(success_cpu).double()
            neg_success = -(weights * success_t).sum()

            neg_success.backward()
            assert a_b.grad is not None
            return neg_success.item(), a_b.grad.flatten().numpy().copy()

        def kl_constraint_fun(flat: np.ndarray) -> float:
            return _compute_kl(flat, prev_flat)

        bounds = [(float(_MIN_BETA_PARAM), float(_MAX_BETA_PARAM))] * (2 * NDIMS)
        kl_con = NonlinearConstraint(kl_constraint_fun, lb=0.0, ub=self.cfg.kl_ub, keep_feasible=True)

        try:
            result = minimize(
                neg_success_and_grad,
                x0,
                method="trust-constr",
                jac=True,
                bounds=bounds,
                constraints=[kl_con],
                options={"maxiter": 50, "gtol": 1e-8},
            )
            if result.success or result.fun < neg_success_and_grad(x0)[0]:
                self.dist.set_flat_params(result.x)
        except Exception as e:
            logger.warning("[DORAEMON] Backup step failed: %s", e)

    def get_metrics(self) -> dict[str, float]:
        """Current distribution stats for logging."""
        metrics = {
            "entropy": self.dist.entropy(),
            "step_count": float(self._step_count),
            "backup_count": float(self._backup_count),
        }
        metrics.update(self.dist.get_stats())
        return metrics
