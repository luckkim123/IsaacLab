# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent workflow scripts for training, evaluation, and benchmarking.

Shared modules:
    resolve_task_configs:      Resolve env_cfg + agent_cfg from gym.spec
    apply_cli_overrides:       Apply num_envs, seed, device from CLI
    build_adapt_policy:        Construct ActorCriticEncoderTDCAdapt
    load_phase1_checkpoint:    Load Phase 1 model (strict=False)
    load_phase2_checkpoint:    Load Phase 2 model (strict=True) + hist_normalizer
    get_proprio_history_shape: Extract history dimensions from env config

Benchmark modules:
    BenchmarkEntry:            Model entry (task_id + checkpoint)
    BenchmarkRunner:           Evaluate (entry, scenario) pairs
    BenchmarkAggregator:       Aggregate and output results
    EpisodeResult:             Per-episode metrics
"""

from ._config_resolver import apply_cli_overrides, resolve_task_configs
from ._policy_factory import (
    build_adapt_policy,
    get_proprio_history_shape,
    load_phase1_checkpoint,
    load_phase2_checkpoint,
)
from .benchmark_runner import (
    BenchmarkAggregator,
    BenchmarkEntry,
    BenchmarkRunner,
    EpisodeResult,
    ScenarioSummary,
)

__all__ = [
    "resolve_task_configs",
    "apply_cli_overrides",
    "build_adapt_policy",
    "get_proprio_history_shape",
    "load_phase1_checkpoint",
    "load_phase2_checkpoint",
    "BenchmarkAggregator",
    "BenchmarkEntry",
    "BenchmarkRunner",
    "EpisodeResult",
    "ScenarioSummary",
]
