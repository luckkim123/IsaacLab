# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-05]

### Context
Code simplification session for hero_agent codebase (~7,700 lines, 27 Python files).
Focused on dead code removal, duplicate code consolidation, and unused reward function cleanup.
During post-simplification diff analysis against run 2026-03-03_09-24-38, discovered that
`termination_penalty = -10.0` was mistakenly removed as "unused" -- it was actually active
(default -10.0, applied on early termination). Restored immediately.

### Changed
- `base_env.py`: Consolidated `_update_perturbation()` main/buoy logic into `_apply_perturbation_cycle()` helper
- `base_env.py`: Added `_iter_noise_params()` static method; simplified `_pad_noise_cfg_for_tde()` and `_convert_noise_cfg_tuples()` from nested loops to single-line iterations
- `base_env.py`: Replaced verbose termination logging with `_term_rate()` helper
- `config.py`: Removed stale MPC docstring reference, removed redundant `ocean_current` and `enable_payload` overrides that matched parent class
- `config.py`: Observation noise tuples use `[val] * N` pattern for readability
- `controllers/tdc.py`: Extracted `_set_param()` static helper for `update_controller_params()`/`update_gains()` deduplication
- `controllers/tdc.py`: Consolidated 11-buffer `reset()` into `_zero_buffers` list + loop
- `mdp/events.py`: Added `_apply_xyz_offset_with_doraemon()` helper to merge CoB/CoG DORAEMON branches (~16 lines x2 -> 2 calls)
- `mdp/events.py`: Removed unused `_apply_xyz_offset()` function (26 lines)

### Removed
- `base_env.py`: Removed `_cumulative_effort` buffer (logging-only, never used in reward)
- `base_env.py`: Removed `HeroAgentEnvWindow` class and `BaseEnvWindow` import
- `mdp/rewards.py`: Removed `action_rate_penalty()` and `angular_velocity_penalty()` functions (both had weight=0.0 in all configs)
- `mdp/__init__.py`: Removed corresponding imports and `__all__` exports
- `controllers/__init__.py`, `encoder/__init__.py`, `runners/__init__.py`: Removed MPC docstring references
- `direct/__init__.py`: Removed `from . import hero_agent_mpc` (directory was deleted previously)

### Fixed
- `mdp/rewards.py`: Restored `termination_penalty: float = -10.0` field that was incorrectly removed during cleanup (was active, not unused)
- `base_env.py`: Restored termination penalty application code in `_get_rewards()`

### Notes
- `encoder_tdc_env.py` kept as reference code (not registered, not simplified)
- All changes verified with `ruff check` and `ruff format`
- Full step-by-step log: `hero_agent/docs/code-simplification-log.md`
