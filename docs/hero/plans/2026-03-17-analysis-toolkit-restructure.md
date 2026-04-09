# Analysis Toolkit Restructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate all Hero Agent analysis scripts into `scripts/analysis/` with shared constants imported from hero_agent modules, eliminating hardcoded values.

**Architecture:** Create `common.py` as SSOT bridge that imports physics constants, encoder architecture, and DR config from hero_agent. All analysis scripts import from `common.py`. Isaac Sim scripts add `sys.path` for `cli_args` access. One-off debug scripts become proper CLI tools.

**Tech Stack:** Python 3.12, PyTorch, matplotlib, TensorBoard, WandB, Isaac Lab

---

## File Structure

### Create
- `scripts/analysis/__init__.py` — empty package marker
- `scripts/analysis/common.py` — shared constants/utilities from hero_agent imports

### Move + Rewrite
- `scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py` -> `scripts/analysis/eval_dr.py`
- `scripts/analysis/compare_eval_dr.py` -> `scripts/analysis/compare_dr.py`
- `scripts/analysis/encoder_z_sweep.py` -> rewrite in place
- `scripts/analysis/plot_encoder_training.py` -> `scripts/analysis/plot_training.py`
- `scripts/analyze_tb_runs.py` -> `scripts/analysis/compare_tb_runs.py`
- `scripts/debug_encoder.py` -> `scripts/analysis/debug_checkpoint.py`
- `scripts/hero_agent/create_wandb_dashboard.py` -> `scripts/analysis/create_wandb_dashboard.py`

### Delete (after moves)
- `scripts/analyze_tb_runs.py`
- `scripts/debug_encoder.py`
- `scripts/hero_agent/create_wandb_dashboard.py`
- `scripts/analysis/compare_eval_dr.py` (renamed)
- `scripts/analysis/plot_encoder_training.py` (renamed)

---

## Chunk 1: Foundation (common.py + package)

### Task 1: Create `scripts/analysis/__init__.py` and `common.py`

**Files:**
- Create: `scripts/analysis/__init__.py`
- Create: `scripts/analysis/common.py`

- [ ] **Step 1: Create empty `__init__.py`**

```python
# scripts/analysis/__init__.py
```

- [ ] **Step 2: Create `common.py` with hero_agent imports**

Key design decisions:
- Import from `isaaclab_assets` for physics constants (volumes, inertias, CoG/CoB)
- Import from `isaaclab_tasks.direct.hero_agent.config` for DR config
- Import from `isaaclab_tasks.direct.hero_agent.agents.rsl_rl_ppo_cfg` for encoder architecture
- Assemble nominal privileged obs from hydro configs (replaces hardcoded `NOMINAL_19D`)
- Derive sweep ranges from `DomainRandomizationCfg` (replaces hardcoded `SweepParam` bounds)
- Provide fallback for non-Isaac-Sim environments (checkpoint-only analysis)

```python
"""Shared constants and utilities for Hero Agent analysis scripts.

Imports authoritative values from hero_agent modules.
Provides fallback for non-Isaac-Sim environments (pure PyTorch analysis).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# DR constants (no Isaac Sim dependency)
# ---------------------------------------------------------------------------

DR_LEVELS: list[str] = ["none", "soft", "medium", "hard"]
DR_SCALE: dict[str, float] = {"none": 0.0, "soft": 0.3, "medium": 0.6, "hard": 1.0}
DR_COLORS: dict[str, str] = {
    "none": "#2196F3",
    "soft": "#4CAF50",
    "medium": "#FF9800",
    "hard": "#F44336",
}

# ---------------------------------------------------------------------------
# Isaac Lab imports (graceful fallback)
# ---------------------------------------------------------------------------

_ISAAC_AVAILABLE = False

try:
    from isaaclab_assets.robots.uuv.hero_agent.hero_agent import (
        HeroAgentBuoyHydrodynamicsCfg,
        HeroAgentHydrodynamicsCfg,
    )
    from isaaclab_tasks.direct.hero_agent.agents.rsl_rl_ppo_cfg import (
        RslRlPpoActorCriticEncoderCfg,
    )
    from isaaclab_tasks.direct.hero_agent.config import DomainRandomizationCfg

    _ISAAC_AVAILABLE = True
except ImportError:
    pass


def _get_encoder_cfg():
    """Return encoder config. Raises if Isaac Lab not available."""
    if not _ISAAC_AVAILABLE:
        raise RuntimeError(
            "Isaac Lab modules not importable. "
            "Run via ./isaaclab.sh -p or install isaaclab_tasks."
        )
    return RslRlPpoActorCriticEncoderCfg()


def get_encoder_architecture() -> dict:
    """Return encoder architecture constants from config.

    Returns:
        Dict with keys: hidden_dims, latent_dim, output_activation, privileged_dim.
    """
    cfg = _get_encoder_cfg()
    return {
        "hidden_dims": cfg.encoder_hidden_dims,
        "latent_dim": cfg.encoder_latent_dim,
        "output_activation": cfg.encoder_output_activation,
        "privileged_dim": cfg.privileged_dim,
    }


def get_encoder_architecture_from_checkpoint(ckpt_path: str) -> dict:
    """Extract encoder architecture from checkpoint state dict.

    Works without Isaac Lab. Infers dims from weight shapes.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Extract encoder weights
    enc_keys = sorted(k for k in sd if k.startswith("encoder."))
    linear_keys = [k for k in enc_keys if k.endswith(".weight")]

    hidden_dims = []
    for i, k in enumerate(linear_keys[:-1]):
        hidden_dims.append(sd[k].shape[0])

    last_w = sd[linear_keys[-1]]
    latent_dim = last_w.shape[0]
    input_dim = sd[linear_keys[0]].shape[1]

    # Detect activation from layer count
    # tanh: encoder has N*2 layers (Linear+ELU pairs) + Linear + Tanh
    # no-tanh: encoder has N*2 layers + Linear only
    n_modules = max(int(k.split(".")[1]) for k in enc_keys if "." in k.split("encoder.")[1]) + 1
    expected_with_tanh = len(linear_keys) * 2  # each Linear has activation after it
    has_output_activation = n_modules > expected_with_tanh - 1

    return {
        "hidden_dims": hidden_dims,
        "latent_dim": latent_dim,
        "input_dim": input_dim,
        "output_activation": "tanh" if has_output_activation else "none",
        "privileged_dim": input_dim,
    }


def build_nominal_obs() -> np.ndarray:
    """Build nominal privileged observation from hydro configs.

    Returns:
        1D numpy array (19D or 20D depending on config).
    """
    main = HeroAgentHydrodynamicsCfg()
    buoy = HeroAgentBuoyHydrodynamicsCfg()
    cfg = _get_encoder_cfg()

    # 19D structure: main_hydro(5) + buoy_hydro(5) + main_inertia(2) + buoy_inertia(2)
    #              + payload(4) + main_added_mass_surge(1)
    obs = [
        # Main body hydro (5D)
        main.volume,
        main.center_of_gravity[0], main.center_of_gravity[1], main.center_of_gravity[2],
        main.center_of_buoyancy[2],
        # Buoy hydro (5D)
        buoy.volume,
        buoy.center_of_gravity[0], buoy.center_of_gravity[1], buoy.center_of_gravity[2],
        buoy.center_of_buoyancy[2],
        # Main inertia (2D): Ixx, Iyy
        main.rigid_body_inertia[0], main.rigid_body_inertia[1],
        # Buoy inertia (2D): Ixx, Iyy
        buoy.rigid_body_inertia[0], buoy.rigid_body_inertia[1],
        # Payload (4D): mass, cog_offset_xyz
        0.5, 0.0, 0.0, -0.015,
        # Main added mass surge (1D)
        main.added_mass[0],
    ]
    return np.array(obs[:cfg.privileged_dim], dtype=np.float32)


def build_sweep_params() -> list[dict]:
    """Build sweep parameter definitions from DR config ranges.

    Returns:
        List of dicts with keys: name, dim_idx, low, high, unit.
    """
    dr = DomainRandomizationCfg()
    main = HeroAgentHydrodynamicsCfg()
    buoy = HeroAgentBuoyHydrodynamicsCfg()

    return [
        {"name": "Main Volume", "dim_idx": 0,
         "low": main.volume * dr.volume_scale[0], "high": main.volume * dr.volume_scale[1], "unit": "m^3"},
        {"name": "Buoy Volume", "dim_idx": 5,
         "low": buoy.volume * dr.volume_scale[0], "high": buoy.volume * dr.volume_scale[1], "unit": "m^3"},
        {"name": "Main CoG Z", "dim_idx": 3,
         "low": main.center_of_gravity[2] + dr.cog_offset_z[0],
         "high": main.center_of_gravity[2] + dr.cog_offset_z[1], "unit": "m"},
        {"name": "Main Inertia Ixx", "dim_idx": 10,
         "low": main.rigid_body_inertia[0] * dr.inertia_scale[0],
         "high": main.rigid_body_inertia[0] * dr.inertia_scale[1], "unit": "kg*m^2"},
        {"name": "Main Inertia Iyy", "dim_idx": 11,
         "low": main.rigid_body_inertia[1] * dr.inertia_scale[0],
         "high": main.rigid_body_inertia[1] * dr.inertia_scale[1], "unit": "kg*m^2"},
        {"name": "Buoy Inertia Ixx", "dim_idx": 12,
         "low": buoy.rigid_body_inertia[0] * dr.inertia_scale[0],
         "high": buoy.rigid_body_inertia[0] * dr.inertia_scale[1], "unit": "kg*m^2"},
        {"name": "Buoy Inertia Iyy", "dim_idx": 13,
         "low": buoy.rigid_body_inertia[1] * dr.inertia_scale[0],
         "high": buoy.rigid_body_inertia[1] * dr.inertia_scale[1], "unit": "kg*m^2"},
        {"name": "Payload Mass", "dim_idx": 14,
         "low": dr.payload_mass_range[0], "high": dr.payload_mass_range[1], "unit": "kg"},
        {"name": "Payload CoG Z", "dim_idx": 17,
         "low": dr.payload_cog_offset_z[0], "high": dr.payload_cog_offset_z[1], "unit": "m"},
        {"name": "Main Added Mass Surge", "dim_idx": 18,
         "low": main.added_mass[0] * dr.added_mass_scale[0],
         "high": main.added_mass[0] * dr.added_mass_scale[1], "unit": "kg"},
    ]


# ---------------------------------------------------------------------------
# TensorBoard utilities
# ---------------------------------------------------------------------------


def load_tb_scalars(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Load all scalar metrics from TensorBoard event files.

    Returns:
        Dict mapping tag -> list of (step, value) tuples.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def find_hero_agent_runs(logs_root: str = "/workspace/isaaclab/logs/rsl_rl") -> list:
    """Find Hero Agent training runs, sorted newest first.

    Returns:
        List of pathlib.Path objects.
    """
    from pathlib import Path

    root = Path(logs_root)
    if not root.exists():
        return []
    runs = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("hero_agent"):
            continue
        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and list(run_dir.glob("events.out.tfevents.*")):
                runs.append(run_dir)
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs
```

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/__init__.py scripts/analysis/common.py
git commit -m "feat(analysis): add common.py with hero_agent imports as SSOT"
```

---

## Chunk 2: Encoder Z Sweep Rewrite

### Task 2: Rewrite `encoder_z_sweep.py` to use `common.py`

**Files:**
- Modify: `scripts/analysis/encoder_z_sweep.py`

- [ ] **Step 1: Rewrite `encoder_z_sweep.py`**

Key changes:
- Remove hardcoded `NOMINAL_19D` (19 lines) -> `common.build_nominal_obs()`
- Remove hardcoded `PRIVILEGED_DIM`, `ENCODER_HIDDEN_DIMS`, `ENCODER_LATENT_DIM` -> `common.get_encoder_architecture()` or checkpoint inference
- Remove hardcoded `SWEEP_PARAMS` (12 lines) -> `common.build_sweep_params()`
- Remove `LATEST_CHECKPOINT` hardcoded path -> no default, require `--checkpoint`
- Keep `build_encoder_mlp()` and `load_encoder()` as they work with arbitrary checkpoints
- Use checkpoint-inferred dims as primary, `common.py` as fallback for sweep params
- Move `activate_z()` function: remove (it duplicates tanh already built into MLP)

The rewritten script should:
1. Accept `--checkpoint` (required)
2. Infer encoder architecture from checkpoint weights
3. Get nominal obs + sweep ranges from `common.py` (Isaac Lab) or fallback to checkpoint-only mode
4. Keep all plotting logic intact

- [ ] **Step 2: Test locally**

```bash
cd /workspace/isaaclab
python scripts/analysis/encoder_z_sweep.py --help
```

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/encoder_z_sweep.py
git commit -m "refactor(analysis): encoder_z_sweep uses common.py, no hardcoded constants"
```

---

## Chunk 3: DR Evaluation Scripts

### Task 3: Move `eval_dr_comparison.py` -> `scripts/analysis/eval_dr.py`

**Files:**
- Create: `scripts/analysis/eval_dr.py` (moved from `scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py`)
- Delete: `scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py`

- [ ] **Step 1: Move file and update imports**

Key changes:
- Add `sys.path` for `cli_args` access:
  ```python
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl"))
  ```
- Replace local `DR_LEVELS`, `DR_COLORS`, `DR_SCALE` with `from common import DR_LEVELS, DR_COLORS, DR_SCALE`
- Keep `build_dr_config()`, `build_step_trajectory()`, `run_evaluation()`, `compute_metrics()`, `generate_plots()` intact (these are Isaac Sim-specific)
- Remove `DR_ANGLES` dict (all levels use same 15 deg) -> single constant `MAX_ANGLE_DEG = 15.0`

- [ ] **Step 2: Delete old file**

```bash
rm scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/eval_dr.py
git add -u scripts/reinforcement_learning/rsl_rl/eval_dr_comparison.py
git commit -m "refactor(analysis): move eval_dr_comparison -> analysis/eval_dr.py"
```

### Task 4: Rename `compare_eval_dr.py` -> `compare_dr.py` + use `common.py`

**Files:**
- Rename: `scripts/analysis/compare_eval_dr.py` -> `scripts/analysis/compare_dr.py`

- [ ] **Step 1: Rename and update**

Key changes:
- Replace local `DR_LEVELS`, `DR_SCALE` with `from common import DR_LEVELS, DR_SCALE`
- Merge `compute_level_metrics()` to be consistent with `eval_dr.py`'s `compute_metrics()`
- Keep the rest intact

- [ ] **Step 2: Delete old file**

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/compare_dr.py
git add -u scripts/analysis/compare_eval_dr.py
git commit -m "refactor(analysis): rename compare_eval_dr -> compare_dr, use common.py"
```

---

## Chunk 4: Training Visualization + TB Comparison

### Task 5: Rename `plot_encoder_training.py` -> `plot_training.py` + use `common.py`

**Files:**
- Rename: `scripts/analysis/plot_encoder_training.py` -> `scripts/analysis/plot_training.py`

- [ ] **Step 1: Rename and update**

Key changes:
- Use `common.load_tb_scalars()` and `common.smooth()` instead of local definitions
- Add `--run-index` option to select run by index (0=latest) via `common.find_hero_agent_runs()`
- Keep 6-panel dashboard layout intact

- [ ] **Step 2: Commit**

```bash
git add scripts/analysis/plot_training.py
git add -u scripts/analysis/plot_encoder_training.py
git commit -m "refactor(analysis): rename plot_encoder_training -> plot_training, use common.py"
```

### Task 6: Rewrite `analyze_tb_runs.py` -> `scripts/analysis/compare_tb_runs.py`

**Files:**
- Create: `scripts/analysis/compare_tb_runs.py`
- Delete: `scripts/analyze_tb_runs.py`

- [ ] **Step 1: Rewrite as proper CLI tool**

Key changes:
- Remove 4 hardcoded run paths -> `--runs` CLI arg (accepts paths or indices)
- Remove hardcoded `TARGET_METRICS` -> `--metrics` CLI arg with sensible defaults
- Remove hardcoded `TARGET_ITERS` -> `--iters` CLI arg with defaults
- Use `common.load_tb_scalars()` and `common.find_hero_agent_runs()`
- Keep console table output format

Usage:
```bash
# Compare two runs by index
python scripts/analysis/compare_tb_runs.py --runs 0 1

# Compare by path
python scripts/analysis/compare_tb_runs.py \
    --runs logs/rsl_rl/hero_agent_.../run_a logs/rsl_rl/hero_agent_.../run_b

# Custom metrics
python scripts/analysis/compare_tb_runs.py --runs 0 1 \
    --metrics "Attitude_Error/roll_deg" "Train/mean_reward"
```

- [ ] **Step 2: Delete old file**

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/compare_tb_runs.py
git add -u scripts/analyze_tb_runs.py
git commit -m "refactor(analysis): rewrite analyze_tb_runs as CLI tool compare_tb_runs"
```

---

## Chunk 5: Debug + WandB

### Task 7: Rewrite `debug_encoder.py` -> `scripts/analysis/debug_checkpoint.py`

**Files:**
- Create: `scripts/analysis/debug_checkpoint.py`
- Delete: `scripts/debug_encoder.py`

- [ ] **Step 1: Rewrite as proper CLI tool**

Key changes:
- Remove 3 hardcoded checkpoint paths -> `--checkpoint` required CLI arg + optional `--baseline`
- Remove `softplus` activation (outdated) -> use `common.get_encoder_architecture_from_checkpoint()` to detect activation
- Restructure into functions: `analyze_weights()`, `test_forward_pass()`, `compare_checkpoints()`, `inspect_optimizer()`
- Keep console output format

Usage:
```bash
# Analyze single checkpoint
python scripts/analysis/debug_checkpoint.py --checkpoint logs/.../model_500.pt

# Compare baseline vs current
python scripts/analysis/debug_checkpoint.py \
    --checkpoint logs/.../model_500.pt \
    --baseline logs/.../model_0.pt
```

- [ ] **Step 2: Delete old file**

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/debug_checkpoint.py
git add -u scripts/debug_encoder.py
git commit -m "refactor(analysis): rewrite debug_encoder as CLI tool debug_checkpoint"
```

### Task 8: Move + update `create_wandb_dashboard.py`

**Files:**
- Move: `scripts/hero_agent/create_wandb_dashboard.py` -> `scripts/analysis/create_wandb_dashboard.py`

- [ ] **Step 1: Move and update panel definitions**

Key changes:
- Remove SAC-MPC references (panels 3, 4, 6 reference deleted architecture)
- Update panels for current PPO+Encoder+TDC architecture:
  - Panel 1: Core metrics (reward, attitude error, episode length)
  - Panel 2: Reward breakdown (tracking, settling, linear_error, action penalties)
  - Panel 3: Encoder health (z_mean, z_std, z_min, z_max, grad_norm)
  - Panel 4: Policy health (entropy, noise_std, lr, KL, line_search_success)
  - Panel 5: Constraints (cost_returns, margins, barrier_t) -- if constrained
  - Panel 6: Dynamics (joint_vel, effort_sat, angular_velocity)
  - Panel 7: DR parameters (buoyancy, inertia, payload, current)
  - Panel 8: Performance (fps, episode_length, termination rates)
- Remove `./isaaclab.sh -p` requirement (pure wandb API, no Isaac Sim needed)

- [ ] **Step 2: Delete old file**

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/create_wandb_dashboard.py
git add -u scripts/hero_agent/create_wandb_dashboard.py
git commit -m "refactor(analysis): move wandb dashboard, update for PPO+Encoder arch"
```

---

## Chunk 6: Skill Update + Cleanup

### Task 9: Update `hero-agent-analysis` skill

**Files:**
- Modify: `~/.claude/skills/hero-agent-analysis/SKILL.md`

- [ ] **Step 1: Update skill with new file paths and commands**

Update all script paths to reflect new `scripts/analysis/` locations.
Add version management note: "All constants imported from hero_agent -- no manual sync needed."

- [ ] **Step 2: Commit skill**

### Task 10: Update CLAUDE.md if needed

- [ ] **Step 1: Check if any analysis script paths are referenced in CLAUDE.md**
- [ ] **Step 2: Update paths if found**
- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "docs: update analysis tool paths in skill and CLAUDE.md"
```
