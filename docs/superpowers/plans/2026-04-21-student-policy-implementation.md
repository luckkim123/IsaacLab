# Student Policy Implementation Plan (TCN + GRU)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a supervised-learning pipeline that trains two student encoders (TCN + GRU, sequentially on GPU 1) to predict r13_A teacher's 9D latent from 87D proprioceptive obs history, using joint action + latent L2 loss with the teacher actor frozen.

**Architecture:** A new package `constrained_full_albc/student/` contains the student encoders (TCN window-based, GRU streaming), a frozen-teacher wrapper that loads r13_A's encoder/actor/normalizer, an online rollout collector that records `(obs_t, privileged_t, l_t, a_t)` while stepping the env with teacher actions, and a supervised runner that does forward-pass-through-frozen-actor + L2 regression + Adam step. A new entry script `scripts/reinforcement_learning/rsl_rl/train_student.py` drives it; `scripts/launch_student_tcn.sh` and `scripts/launch_student_gru.sh` run them sequentially.

**Tech Stack:** PyTorch 2.10+, Isaac Lab 5.1 / Isaac Sim, RSL-RL 3.0.1, existing `FullDOFActorCriticEncoder` (reused for teacher state dict loading), WandB + TensorBoard.

---

## File Structure

New files:

```
source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/
├── __init__.py          # re-exports
├── config.py            # StudentCfg dataclass
├── models.py            # StudentEncoderTCN, StudentEncoderGRU
├── teacher.py           # FrozenTeacher wrapper (loads r13_A state dict)
├── collector.py         # RolloutBuffer + collect_rollout
├── runner.py            # StudentRunner main training loop
└── eval.py              # student-in-loop env wrapper for eval_dr_fulldof

scripts/
├── reinforcement_learning/rsl_rl/train_student.py   # entry point
├── launch_student_tcn.sh                            # GPU 1 launcher TCN
└── launch_student_gru.sh                            # GPU 1 launcher GRU

logs/rsl_rl/student_policy/<timestamp>_<name>/       # training outputs
```

No modifications to existing code except adding the task registration (see Task 11).

---

## Task 1: StudentCfg dataclass

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/__init__.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/config.py`

- [ ] **Step 1: Create the student package root**

```python
# source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/__init__.py
"""Student policy package: TCN + GRU encoders distilled from r13_A teacher.

Exports:
    StudentCfg  -- training configuration
    StudentEncoderTCN, StudentEncoderGRU  -- encoder architectures
    FrozenTeacher  -- frozen teacher wrapper
    RolloutBuffer, collect_rollout  -- online data collection
    StudentRunner  -- supervised training loop
"""
```

- [ ] **Step 2: Write the config file**

```python
# source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/config.py
"""Student policy training configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StudentCfg:
    """Hyperparameters for student encoder supervised training."""

    # Experiment
    experiment_name: str = "student_policy"
    run_name: str = "student_tcn"
    seed: int = 42

    # Teacher
    teacher_run_dir: str = "logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A"
    teacher_checkpoint: str = "model_4999.pt"

    # Architecture
    encoder_type: str = "tcn"       # "tcn" or "gru"
    policy_obs_dim: int = 87
    privileged_dim: int = 24
    latent_dim: int = 9             # must match r13_A teacher

    # TCN-specific
    tcn_history: int = 50           # H = 50 steps (1.0 s at 50 Hz)
    tcn_input_channels: int = 32    # after per-step channel transform
    tcn_conv_channels: tuple[int, ...] = (64, 128, 128)
    tcn_conv_kernels: tuple[int, ...] = (9, 5, 5)
    tcn_conv_strides: tuple[int, ...] = (2, 1, 1)
    tcn_head_hidden: int = 128

    # GRU-specific
    gru_layers: int = 1
    gru_hidden: int = 128

    # Training
    num_envs: int = 4096
    n_steps_per_rollout: int = 24
    n_epochs: int = 5
    minibatch_size: int = 8192
    lr: float = 5e-4
    max_iterations: int = 1000
    grad_clip_norm: float = 1.0
    lambda_latent: float = 1.0
    save_interval: int = 100

    # Logging
    log_dir_root: str = "logs/rsl_rl/student_policy"
    logger: str = "wandb"           # "wandb" or "tensorboard"
    wandb_project: str = "full_dof_trpo_student"

    # Environment
    task: str = "Isaac-FullDOF-TRPO-v0"
    device: str = "cuda:0"          # overridden by CUDA_VISIBLE_DEVICES at launch
```

- [ ] **Step 3: Smoke test imports work**

Run:
```bash
cd /workspace/isaaclab
./isaaclab.sh -p -c "from isaaclab_tasks.direct.constrained_full_albc.student.config import StudentCfg; print(StudentCfg().policy_obs_dim)"
```
Expected: `87`

- [ ] **Step 4: Commit**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/__init__.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/config.py
git commit -m "feat(student): add StudentCfg dataclass"
```

---

## Task 2: Student encoders (TCN + GRU)

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/models.py`

- [ ] **Step 1: Write StudentEncoderTCN and StudentEncoderGRU**

```python
# source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/models.py
"""Student encoder architectures: window-based TCN and streaming GRU.

Both output 9D latent in (-1, 1) via softsign, matching r13_A teacher's
privileged encoder output range so latent L2 loss is well-scaled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StudentCfg


class StudentEncoderTCN(nn.Module):
    """Window-based temporal conv encoder.

    Input:  (B, H, D) where H=50, D=87
    Output: (B, latent_dim) in (-1, 1)
    """

    def __init__(self, cfg: StudentCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.history_len = cfg.tcn_history

        # Per-step channel transform: maps raw 87D features -> tcn_input_channels
        self.channel_transform = nn.Sequential(
            nn.Linear(cfg.policy_obs_dim, cfg.tcn_input_channels),
            nn.ELU(),
        )

        # 1D conv stack
        in_ch = cfg.tcn_input_channels
        convs: list[nn.Module] = []
        seq_len = cfg.tcn_history
        for out_ch, k, s in zip(cfg.tcn_conv_channels, cfg.tcn_conv_kernels, cfg.tcn_conv_strides):
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s))
            convs.append(nn.ELU())
            seq_len = (seq_len - k) // s + 1
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)
        self.flatten_dim = in_ch * seq_len

        # Head
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, cfg.tcn_head_hidden),
            nn.ELU(),
            nn.LayerNorm(cfg.tcn_head_hidden),
            nn.Linear(cfg.tcn_head_hidden, cfg.latent_dim),
        )

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        """obs_window: (B, H, D) -> l_hat: (B, latent_dim)."""
        b, h, d = obs_window.shape
        # Apply channel transform per timestep: (B, H, D) -> (B, H, C)
        x = self.channel_transform(obs_window.reshape(b * h, d)).reshape(b, h, -1)
        # Transpose for Conv1d: (B, H, C) -> (B, C, H)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Flatten time + channels
        x = x.reshape(b, -1)
        z = self.head(x)
        return F.softsign(z)


class StudentEncoderGRU(nn.Module):
    """Streaming GRU encoder.

    Uses GRU (not GRUCell) for efficient training over temporal chunks.
    For single-step inference, pass (B, 1, D) and carry hidden across calls.
    """

    def __init__(self, cfg: StudentCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.policy_obs_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs_seq: (B, T, D), hidden: (num_layers, B, gru_hidden) or None.

        Returns:
            l_hat: (B, T, latent_dim) -- all timesteps
            hidden: (num_layers, B, gru_hidden) -- final hidden state
        """
        out, hidden_out = self.gru(obs_seq, hidden)
        z = self.head(out)
        return F.softsign(z), hidden_out

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.cfg.gru_layers, batch_size, self.cfg.gru_hidden, device=device)


def make_student_encoder(cfg: StudentCfg) -> nn.Module:
    """Factory."""
    if cfg.encoder_type == "tcn":
        return StudentEncoderTCN(cfg)
    if cfg.encoder_type == "gru":
        return StudentEncoderGRU(cfg)
    raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")
```

- [ ] **Step 2: Write a forward-pass smoke test**

Create `/tmp/test_student_encoders.py`:

```python
"""Smoke-test student encoders: forward pass + gradient flow."""
import torch

from isaaclab_tasks.direct.constrained_full_albc.student.config import StudentCfg
from isaaclab_tasks.direct.constrained_full_albc.student.models import (
    StudentEncoderTCN,
    StudentEncoderGRU,
)

B = 16
cfg = StudentCfg()

# TCN
tcn = StudentEncoderTCN(cfg)
tcn_in = torch.randn(B, cfg.tcn_history, cfg.policy_obs_dim)
tcn_out = tcn(tcn_in)
assert tcn_out.shape == (B, cfg.latent_dim), f"TCN shape {tcn_out.shape}"
assert ((-1.0 < tcn_out) & (tcn_out < 1.0)).all(), "TCN output not in (-1, 1)"
tcn_out.sum().backward()
grad_params = [p for p in tcn.parameters() if p.grad is not None]
assert len(grad_params) > 0, "TCN grad did not flow"
print(f"TCN OK: {tcn_out.shape}, {len(grad_params)} params with grad")

# GRU
cfg.encoder_type = "gru"
gru = StudentEncoderGRU(cfg)
T = 10
gru_in = torch.randn(B, T, cfg.policy_obs_dim)
gru_out, hidden = gru(gru_in)
assert gru_out.shape == (B, T, cfg.latent_dim), f"GRU shape {gru_out.shape}"
assert hidden.shape == (cfg.gru_layers, B, cfg.gru_hidden), f"GRU hidden {hidden.shape}"
assert ((-1.0 < gru_out) & (gru_out < 1.0)).all(), "GRU output not in (-1, 1)"
gru_out.sum().backward()
print(f"GRU OK: {gru_out.shape}, hidden {hidden.shape}")

print("All encoder smoke tests passed.")
```

- [ ] **Step 3: Run smoke test**

Run:
```bash
cd /workspace/isaaclab
./isaaclab.sh -p /tmp/test_student_encoders.py
```
Expected output ending with:
```
TCN OK: torch.Size([16, 9]), <N> params with grad
GRU OK: torch.Size([16, 10, 9]), hidden torch.Size([1, 16, 128])
All encoder smoke tests passed.
```

- [ ] **Step 4: Commit**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/models.py
git commit -m "feat(student): add TCN and GRU encoder architectures"
```

---

## Task 3: FrozenTeacher wrapper

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/teacher.py`

- [ ] **Step 1: Write FrozenTeacher**

```python
# source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/teacher.py
"""FrozenTeacher: loads r13_A checkpoint, exposes frozen encoder + actor + normalizer.

Uses FullDOFActorCriticEncoder from the teacher's training registry so the
state_dict loads without modification. All parameters have requires_grad=False.
Autograd still flows through for student training (gradients to student encoder
only; teacher weights are never updated).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from .config import StudentCfg

logger = logging.getLogger(__name__)


class FrozenTeacher(nn.Module):
    """Wraps r13_A's ActorCriticEncoder; exposes encode(), normalize_obs(), actor_forward().

    Attributes:
        latent_dim: 9 (r13_A)
        obs_dim: 87 (policy obs)
        privileged_dim: 24
    """

    def __init__(self, cfg: StudentCfg, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Build a teacher policy with the same arch as r13_A. We use the registry
        # class rather than instantiating ActorCriticEncoder directly to ensure
        # the exact arch (e.g. FullDOFActorCriticEncoder overrides).
        from isaaclab_tasks.direct.constrained_full_albc.encoder import (
            ActorCriticEncoder,
        )
        from isaaclab_tasks.direct.constrained_full_albc.agents.rsl_rl_ppo_cfg import (
            _PRIV_OBS_LOWER,
            _PRIV_OBS_UPPER,
        )

        # We don't have the real TensorDict here; instead we build bypassing obs_groups
        # by calling nn.Module construction of the components directly.
        # Easiest path: reuse ActorCriticEncoder via a minimal dummy obs dict.
        from tensordict import TensorDict

        dummy_obs = TensorDict(
            {
                "policy": torch.zeros(1, cfg.policy_obs_dim),
                "privileged": torch.zeros(1, cfg.privileged_dim),
            },
            batch_size=[1],
        )
        obs_groups = {"policy": ["policy", "privileged"], "critic": ["policy", "privileged"]}
        self.policy = ActorCriticEncoder(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=8,
            policy_obs_dim=cfg.policy_obs_dim,
            privileged_dim=cfg.privileged_dim,
            encoder_hidden_dims=(256, 128, 64),
            encoder_latent_dim=cfg.latent_dim,
            encoder_activation="elu",
            encoder_obs_normalization=False,
            encoder_obs_lower=_PRIV_OBS_LOWER,
            encoder_obs_upper=_PRIV_OBS_UPPER,
            encoder_output_norm=True,
            actor_obs_normalization=True,
            critic_obs_normalization=False,
            actor_hidden_dims=(256, 128, 64),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            init_noise_std=0.7,
            critic_uses_z=True,
            num_constraints=10,
            cost_critic_hidden_dims=(512, 256, 128),
        )
        self.policy.to(device)

        # Load r13_A state dict
        ckpt_path = os.path.join(cfg.teacher_run_dir, cfg.teacher_checkpoint)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        missing, unexpected = self.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(
            "Loaded teacher from %s (iter=%s). Missing: %d, Unexpected: %d",
            ckpt_path,
            ckpt.get("iter", "?"),
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.debug("Missing keys: %s", missing)
        if unexpected:
            logger.debug("Unexpected keys: %s", unexpected)

        # Freeze all teacher parameters
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.policy.eval()

        self.latent_dim = cfg.latent_dim
        self.obs_dim = cfg.policy_obs_dim
        self.privileged_dim = cfg.privileged_dim

    @torch.no_grad()
    def encode_privileged(self, privileged: torch.Tensor) -> torch.Tensor:
        """Ground-truth latent from privileged obs: (B, 24) -> (B, 9)."""
        from tensordict import TensorDict
        dummy = TensorDict(
            {
                "policy": torch.zeros(privileged.shape[0], self.obs_dim, device=privileged.device),
                "privileged": privileged,
            },
            batch_size=[privileged.shape[0]],
        )
        return self.policy._encode(dummy)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply teacher's actor_obs_normalizer (frozen EmpiricalNorm)."""
        return self.policy.actor_obs_normalizer(obs)

    def actor_forward(self, obs_normed: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Teacher actor forward: cat([normed obs, latent]) -> action.

        obs_normed: (B, 87) already through actor_obs_normalizer
        latent: (B, 9) -- either ground-truth l_t or student's l_hat
        Returns: (B, 8)
        """
        actor_in = torch.cat([obs_normed, latent], dim=-1)
        return self.policy.actor(actor_in)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, privileged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce teacher action + ground-truth latent (for env stepping + targets).

        Returns:
            a_t: (B, 8) teacher deterministic action
            l_t: (B, 9) teacher's ground-truth latent
        """
        from tensordict import TensorDict
        dummy = TensorDict({"policy": obs, "privileged": privileged}, batch_size=[obs.shape[0]])
        a_t = self.policy.act_inference(dummy)
        l_t = self.policy._encode(dummy)
        return a_t, l_t
```

- [ ] **Step 2: Write teacher-loading smoke test**

Create `/tmp/test_frozen_teacher.py`:

```python
"""Smoke test FrozenTeacher load + forward pass."""
import torch
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
launcher = AppLauncher(args)
sim_app = launcher.app

from isaaclab_tasks.direct.constrained_full_albc.student.config import StudentCfg
from isaaclab_tasks.direct.constrained_full_albc.student.teacher import FrozenTeacher

cfg = StudentCfg()
device = torch.device("cuda:0")
teacher = FrozenTeacher(cfg, device=device)

B = 16
obs = torch.randn(B, cfg.policy_obs_dim, device=device)
priv = torch.randn(B, cfg.privileged_dim, device=device)

# encode_privileged
l_t = teacher.encode_privileged(priv)
assert l_t.shape == (B, 9), f"l_t shape {l_t.shape}"
assert ((-1.0 < l_t) & (l_t < 1.0)).all(), "l_t not in (-1,1)"

# act
a_t, l_t2 = teacher.act(obs, priv)
assert a_t.shape == (B, 8), f"a_t shape {a_t.shape}"
assert torch.allclose(l_t, l_t2), "l_t mismatch between encode and act"

# Verify all params frozen
trainable = [p for p in teacher.parameters() if p.requires_grad]
assert len(trainable) == 0, f"Teacher has {len(trainable)} trainable params"

print("FrozenTeacher smoke test passed.")
sim_app.close()
```

- [ ] **Step 3: Run the smoke test**

Run:
```bash
cd /workspace/isaaclab
./isaaclab.sh -p /tmp/test_frozen_teacher.py 2>&1 | tail -20
```
Expected output contains `FrozenTeacher smoke test passed.`

- [ ] **Step 4: Commit**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/teacher.py
git commit -m "feat(student): add FrozenTeacher wrapper for r13_A checkpoint"
```

---

## Task 4: RolloutBuffer + collect_rollout

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/collector.py`

- [ ] **Step 1: Write RolloutBuffer**

```python
# source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/collector.py
"""Online rollout collection for student training.

During an iteration we:
    1. Run N env steps with teacher actions (so data distribution matches r13_A's).
    2. Record (obs_t, privileged_t, l_t, a_t) plus, for the TCN student, a sliding
       H-step window of obs; for the GRU student, (obs_t, done_t) and carry hidden.
    3. Flatten (num_envs, N) -> (num_envs*N,) minibatches for SGD.

Buffer memory is proportional to num_envs * n_steps * obs_dim and is released
after each iteration.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import StudentCfg


@dataclass
class RolloutBatch:
    """Flat minibatch of training samples."""
    obs_window: torch.Tensor | None   # (M, H, 87) for TCN, None for GRU
    obs_seq: torch.Tensor | None      # (M_envs, T, 87) for GRU chunks, None for TCN
    dones_seq: torch.Tensor | None    # (M_envs, T) for GRU chunks
    obs_t: torch.Tensor               # (M, 87) -- the "current" obs used by teacher actor
    l_t: torch.Tensor                 # (M, 9)
    a_t: torch.Tensor                 # (M, 8)


class RolloutBuffer:
    """Stores one iteration of rollout data.

    For TCN: keeps a per-env ring buffer of H past obs + flat tensors of current step.
    For GRU: keeps sequential (obs, done) per env across the rollout.
    """

    def __init__(self, cfg: StudentCfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = cfg.num_envs
        self.n_steps = cfg.n_steps_per_rollout

        # Per-env ring buffer for TCN windowing. Zero-padded at episode reset.
        self.ring = torch.zeros(self.num_envs, cfg.tcn_history, cfg.policy_obs_dim, device=device)
        self.ring_idx = 0  # newest slot index (wraps)

        # Flat tensors filled during rollout
        self.obs_flat = torch.zeros(self.n_steps, self.num_envs, cfg.policy_obs_dim, device=device)
        self.obs_window_flat = torch.zeros(
            self.n_steps, self.num_envs, cfg.tcn_history, cfg.policy_obs_dim, device=device
        )
        self.priv_flat = torch.zeros(self.n_steps, self.num_envs, cfg.privileged_dim, device=device)
        self.l_gt_flat = torch.zeros(self.n_steps, self.num_envs, cfg.latent_dim, device=device)
        self.a_gt_flat = torch.zeros(self.n_steps, self.num_envs, 8, device=device)
        self.done_flat = torch.zeros(self.n_steps, self.num_envs, dtype=torch.bool, device=device)

        self.ptr = 0

    def reset_env(self, env_ids: torch.Tensor) -> None:
        """Clear ring buffer for reset envs (zero-padding)."""
        self.ring[env_ids] = 0.0

    def add(
        self,
        obs: torch.Tensor,          # (num_envs, 87)
        privileged: torch.Tensor,   # (num_envs, 24)
        l_t: torch.Tensor,          # (num_envs, 9)
        a_t: torch.Tensor,          # (num_envs, 8)
        dones: torch.Tensor,        # (num_envs,) bool, from the PREVIOUS env step
    ) -> None:
        """Add one timestep and update the windowed ring buffer.

        Ordering:
            1. If dones from previous step, zero-out those envs' ring (pre-step reset).
            2. Push new obs onto ring.
            3. Snapshot the full window into obs_window_flat[ptr].
        """
        # Zero out resetted envs' ring first
        if dones.any():
            self.reset_env(torch.nonzero(dones).squeeze(-1))

        # Shift ring and insert newest obs at position 0.
        # Ring layout: ring[:, 0] = most recent, ring[:, H-1] = oldest.
        self.ring = torch.roll(self.ring, shifts=1, dims=1)
        self.ring[:, 0] = obs

        # Snapshot
        self.obs_flat[self.ptr] = obs
        self.obs_window_flat[self.ptr] = self.ring.clone()
        self.priv_flat[self.ptr] = privileged
        self.l_gt_flat[self.ptr] = l_t
        self.a_gt_flat[self.ptr] = a_t
        self.done_flat[self.ptr] = dones
        self.ptr += 1

    def iter_minibatches_tcn(self) -> list[RolloutBatch]:
        """Flatten (T, E, ...) -> (T*E, ...) and shuffle into minibatches for TCN."""
        obs_flat = self.obs_flat[: self.ptr].reshape(-1, self.cfg.policy_obs_dim)
        win_flat = self.obs_window_flat[: self.ptr].reshape(
            -1, self.cfg.tcn_history, self.cfg.policy_obs_dim
        )
        l_flat = self.l_gt_flat[: self.ptr].reshape(-1, self.cfg.latent_dim)
        a_flat = self.a_gt_flat[: self.ptr].reshape(-1, 8)

        N = obs_flat.shape[0]
        perm = torch.randperm(N, device=self.device)
        batches = []
        for start in range(0, N, self.cfg.minibatch_size):
            idx = perm[start : start + self.cfg.minibatch_size]
            batches.append(
                RolloutBatch(
                    obs_window=win_flat[idx],
                    obs_seq=None,
                    dones_seq=None,
                    obs_t=obs_flat[idx],
                    l_t=l_flat[idx],
                    a_t=a_flat[idx],
                )
            )
        return batches

    def iter_minibatches_gru(self) -> list[RolloutBatch]:
        """For GRU we keep sequential chunks to enable BPTT.

        Layout: slice envs into groups of size M_envs = minibatch_size // n_steps,
        yielding (M_envs, T) sequences with their done mask.
        """
        T = self.ptr
        E = self.num_envs
        envs_per_batch = max(1, self.cfg.minibatch_size // T)
        perm = torch.randperm(E, device=self.device)
        batches = []
        for start in range(0, E, envs_per_batch):
            idx = perm[start : start + envs_per_batch]
            batches.append(
                RolloutBatch(
                    obs_window=None,
                    obs_seq=self.obs_flat[:T, idx].transpose(0, 1),      # (envs, T, 87)
                    dones_seq=self.done_flat[:T, idx].transpose(0, 1),   # (envs, T)
                    obs_t=self.obs_flat[:T, idx].reshape(-1, self.cfg.policy_obs_dim),
                    l_t=self.l_gt_flat[:T, idx].reshape(-1, self.cfg.latent_dim),
                    a_t=self.a_gt_flat[:T, idx].reshape(-1, 8),
                )
            )
        return batches

    def reset(self) -> None:
        self.ptr = 0
```

- [ ] **Step 2: Write a buffer smoke test**

Create `/tmp/test_rollout_buffer.py`:

```python
"""Smoke test RolloutBuffer add / minibatch paths."""
import torch

from isaaclab_tasks.direct.constrained_full_albc.student.config import StudentCfg
from isaaclab_tasks.direct.constrained_full_albc.student.collector import RolloutBuffer

cfg = StudentCfg()
cfg.num_envs = 8
cfg.n_steps_per_rollout = 4
cfg.minibatch_size = 8
device = torch.device("cpu")
buf = RolloutBuffer(cfg, device=device)

for t in range(cfg.n_steps_per_rollout):
    obs = torch.randn(cfg.num_envs, 87)
    priv = torch.randn(cfg.num_envs, 24)
    l = torch.randn(cfg.num_envs, 9)
    a = torch.randn(cfg.num_envs, 8)
    dones = torch.zeros(cfg.num_envs, dtype=torch.bool)
    if t == 2:
        dones[3] = True
    buf.add(obs, priv, l, a, dones)

cfg.encoder_type = "tcn"
tcn_batches = buf.iter_minibatches_tcn()
assert len(tcn_batches) > 0
assert tcn_batches[0].obs_window.shape[1:] == (cfg.tcn_history, 87)

cfg.encoder_type = "gru"
gru_batches = buf.iter_minibatches_gru()
assert len(gru_batches) > 0
assert gru_batches[0].obs_seq is not None
assert gru_batches[0].dones_seq is not None

print("RolloutBuffer smoke test passed.")
```

- [ ] **Step 3: Run the smoke test**

Run:
```bash
cd /workspace/isaaclab
./isaaclab.sh -p /tmp/test_rollout_buffer.py
```
Expected: `RolloutBuffer smoke test passed.`

- [ ] **Step 4: Commit**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/collector.py
git commit -m "feat(student): add RolloutBuffer with TCN window + GRU sequence paths"
```

---

## Task 5: StudentRunner main training loop

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/runner.py`

- [ ] **Step 1: Write StudentRunner**

```python
# source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/runner.py
"""Supervised training loop for the student encoder.

Each iteration:
    1. Collect n_steps rollout (teacher drives env; record (obs, priv, l_t, a_t, dones)).
    2. Compute l_hat via student encoder (TCN window or GRU sequence).
    3. Compute a_hat via frozen teacher actor(normalize(obs_t), l_hat).
    4. Loss = ||a_hat - a_t||^2 + lambda * ||l_hat - l_t||^2.
    5. Adam step on student params only.

Logging: TensorBoard + optional WandB. Checkpoints every save_interval iters.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .collector import RolloutBuffer
from .config import StudentCfg
from .models import StudentEncoderGRU, StudentEncoderTCN, make_student_encoder
from .teacher import FrozenTeacher

logger = logging.getLogger(__name__)

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


class StudentRunner:
    """Drives rollout collection + supervised optimization."""

    def __init__(self, env, cfg: StudentCfg, log_dir: str, device: torch.device) -> None:
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.device = device

        self.teacher = FrozenTeacher(cfg, device=device)
        self.student = make_student_encoder(cfg).to(device)

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=cfg.lr)

        self.buffer = RolloutBuffer(cfg, device=device)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.use_wandb = cfg.logger == "wandb" and _HAS_WANDB
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb_project,
                name=os.path.basename(log_dir),
                config=vars(cfg),
                dir=log_dir,
            )

        # GRU hidden state carried across rollout steps
        if cfg.encoder_type == "gru":
            self.gru_hidden = self.student.init_hidden(cfg.num_envs, device)
        else:
            self.gru_hidden = None

        os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
        logger.info("StudentRunner initialized: encoder=%s log_dir=%s", cfg.encoder_type, log_dir)

    def _collect_rollout(self, obs: torch.Tensor, privileged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run n_steps env steps with teacher actions, filling the buffer.

        Returns the final (obs, privileged) so the caller can carry state.
        """
        self.buffer.reset()
        for _ in range(self.cfg.n_steps_per_rollout):
            # Teacher acts from current obs+privileged
            a_t, l_t = self.teacher.act(obs, privileged)

            # Step env with teacher action
            obs_next, _rew, dones, extras = self.env.step(a_t)
            privileged_next = obs_next["privileged"]
            obs_next_policy = obs_next["policy"]

            # Record at the step AFTER dones/reset processing. The "dones" indicate
            # which envs were reset between previous step and this new obs.
            self.buffer.add(obs_next_policy, privileged_next, l_t.detach(), a_t.detach(), dones.to(torch.bool))

            # GRU hidden state: zero out for reset envs so the next forward pass
            # starts fresh for them.
            if self.gru_hidden is not None and dones.any():
                reset_ids = torch.nonzero(dones).squeeze(-1)
                self.gru_hidden[:, reset_ids] = 0.0

            obs = obs_next_policy
            privileged = privileged_next

        return obs, privileged

    def _compute_loss_tcn(self, batch) -> dict[str, torch.Tensor]:
        l_hat = self.student(batch.obs_window)           # (M, 9)
        obs_normed = self.teacher.normalize_obs(batch.obs_t)
        a_hat = self.teacher.actor_forward(obs_normed, l_hat)  # (M, 8)
        loss_action = F.mse_loss(a_hat, batch.a_t)
        loss_latent = F.mse_loss(l_hat, batch.l_t)
        total = loss_action + self.cfg.lambda_latent * loss_latent
        return {"loss_total": total, "loss_action": loss_action, "loss_latent": loss_latent}

    def _compute_loss_gru(self, batch) -> dict[str, torch.Tensor]:
        # Forward over sequence; ignore hidden state from training chunks (we
        # treat each chunk as an independent BPTT unit for simplicity).
        l_hat_seq, _ = self.student(batch.obs_seq, hidden=None)         # (envs, T, 9)
        M = l_hat_seq.shape[0] * l_hat_seq.shape[1]
        l_hat = l_hat_seq.reshape(M, -1)
        obs_normed = self.teacher.normalize_obs(batch.obs_t)
        a_hat = self.teacher.actor_forward(obs_normed, l_hat)
        loss_action = F.mse_loss(a_hat, batch.a_t)
        loss_latent = F.mse_loss(l_hat, batch.l_t)
        total = loss_action + self.cfg.lambda_latent * loss_latent
        return {"loss_total": total, "loss_action": loss_action, "loss_latent": loss_latent}

    def _log(self, iter_idx: int, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, iter_idx)
        if self.use_wandb:
            wandb.log(metrics, step=iter_idx)

    def _save_checkpoint(self, iter_idx: int) -> None:
        path = os.path.join(self.log_dir, "models", f"student_{iter_idx}.pt")
        torch.save({"iter": iter_idx, "student_state_dict": self.student.state_dict(), "cfg": vars(self.cfg)}, path)
        logger.info("Saved student checkpoint: %s", path)

    def learn(self) -> None:
        # Reset env once at the start
        obs_td, _extras = self.env.reset()
        obs = obs_td["policy"]
        privileged = obs_td["privileged"]

        t_start = time.time()
        for it in range(self.cfg.max_iterations):
            # Collect
            t0 = time.time()
            obs, privileged = self._collect_rollout(obs, privileged)
            t_collect = time.time() - t0

            # Train
            t0 = time.time()
            epoch_totals = {"loss_total": 0.0, "loss_action": 0.0, "loss_latent": 0.0, "grad_norm": 0.0}
            n_updates = 0
            for _ in range(self.cfg.n_epochs):
                if self.cfg.encoder_type == "tcn":
                    batches = self.buffer.iter_minibatches_tcn()
                else:
                    batches = self.buffer.iter_minibatches_gru()
                for batch in batches:
                    if self.cfg.encoder_type == "tcn":
                        losses = self._compute_loss_tcn(batch)
                    else:
                        losses = self._compute_loss_gru(batch)
                    self.optimizer.zero_grad()
                    losses["loss_total"].backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), self.cfg.grad_clip_norm
                    )
                    self.optimizer.step()
                    for k in ("loss_total", "loss_action", "loss_latent"):
                        epoch_totals[k] += losses[k].item()
                    epoch_totals["grad_norm"] += float(grad_norm)
                    n_updates += 1
            for k in epoch_totals:
                epoch_totals[k] /= max(1, n_updates)
            t_train = time.time() - t0

            # Log
            metrics = {
                "student/loss_total": epoch_totals["loss_total"],
                "student/loss_action": epoch_totals["loss_action"],
                "student/loss_latent": epoch_totals["loss_latent"],
                "student/grad_norm": epoch_totals["grad_norm"],
                "student/time_collect": t_collect,
                "student/time_train": t_train,
                "student/iter": it,
            }
            self._log(it, metrics)

            if it % 10 == 0:
                logger.info(
                    "iter=%d total=%.4f action=%.4f latent=%.4f grad=%.3f t_c=%.2fs t_t=%.2fs",
                    it,
                    metrics["student/loss_total"],
                    metrics["student/loss_action"],
                    metrics["student/loss_latent"],
                    metrics["student/grad_norm"],
                    t_collect,
                    t_train,
                )

            if (it + 1) % self.cfg.save_interval == 0 or it == self.cfg.max_iterations - 1:
                self._save_checkpoint(it)

        logger.info("Training done. Total time: %.1f min.", (time.time() - t_start) / 60.0)
        if self.use_wandb:
            wandb.finish()
        self.writer.close()
```

- [ ] **Step 2: Commit (no standalone test — needs live env; covered by Task 7 integration)**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/runner.py
git commit -m "feat(student): add StudentRunner supervised training loop"
```

---

## Task 6: Disable DORAEMON + enforce HardDR hook

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/runner.py`

- [ ] **Step 1: Add a one-shot env config patch helper**

Append to `runner.py` at module level (above `StudentRunner` class):

```python
def configure_env_for_student(env) -> None:
    """Disable DORAEMON and force HardDR on the env before learning starts.

    Per spec section 4: student training uses r14 aggressive HardDR uniformly
    and disables DORAEMON's Beta curriculum.
    """
    env_cfg = env.unwrapped.cfg
    doraemon_cfg = getattr(env_cfg, "doraemon", None)
    if doraemon_cfg is not None and getattr(doraemon_cfg, "enable", False):
        doraemon_cfg.enable = False
        logger.info("[Student] DORAEMON disabled for supervised training.")

    # Replace randomization cfg with HardDomainRandomizationCfg instance
    from isaaclab_tasks.direct.constrained_full_albc.config import HardDomainRandomizationCfg
    hard = HardDomainRandomizationCfg()
    env_cfg.randomization = hard
    logger.info("[Student] Randomization forced to HardDomainRandomizationCfg.")

    # Re-initialize env internals that cache randomization ranges.
    if hasattr(env.unwrapped, "_reload_randomization"):
        env.unwrapped._reload_randomization()
    else:
        # Fallback: full env re-init happens on next reset. The existing DR sampler
        # reads env_cfg.randomization at reset time, so a reset is sufficient.
        pass
```

- [ ] **Step 2: Wire it into `StudentRunner.__init__`**

Edit the init — insert after `self.env = env` line:

```python
        # Configure env before any rollout: HardDR on, DORAEMON off.
        configure_env_for_student(env)
```

- [ ] **Step 3: Commit**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/runner.py
git commit -m "feat(student): force HardDR + disable DORAEMON for student training"
```

---

## Task 7: Entry point script train_student.py

**Files:**
- Create: `scripts/reinforcement_learning/rsl_rl/train_student.py`

- [ ] **Step 1: Write the entry point**

```python
#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Train a student policy (TCN or GRU) via behavior cloning from a teacher checkpoint.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_student.py \
        --encoder_type tcn \
        --teacher_run_dir logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A \
        --teacher_checkpoint model_4999.pt \
        --num_envs 4096 --max_iterations 1000 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train student policy from teacher checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-FullDOF-TRPO-v0")
parser.add_argument("--encoder_type", type=str, choices=["tcn", "gru"], required=True)
parser.add_argument("--run_name", type=str, default=None, help="Override auto-named run (default: student_<encoder_type>).")
parser.add_argument("--teacher_run_dir", type=str, default="logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A")
parser.add_argument("--teacher_checkpoint", type=str, default="model_4999.pt")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--n_steps_per_rollout", type=int, default=24)
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--minibatch_size", type=int, default=8192)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--lambda_latent", type=float, default=1.0)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard"])
parser.add_argument("--wandb_project", type=str, default="full_dof_trpo_student")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Launch Omniverse app (required before importing isaaclab_tasks)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""
import logging
import time
from datetime import datetime

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.direct.constrained_full_albc.student.config import StudentCfg
from isaaclab_tasks.direct.constrained_full_albc.student.runner import StudentRunner

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_student")


def main() -> None:
    # Build student cfg from CLI
    cfg = StudentCfg()
    cfg.encoder_type = args_cli.encoder_type
    cfg.teacher_run_dir = args_cli.teacher_run_dir
    cfg.teacher_checkpoint = args_cli.teacher_checkpoint
    cfg.num_envs = args_cli.num_envs
    cfg.max_iterations = args_cli.max_iterations
    cfg.n_steps_per_rollout = args_cli.n_steps_per_rollout
    cfg.n_epochs = args_cli.n_epochs
    cfg.minibatch_size = args_cli.minibatch_size
    cfg.lr = args_cli.lr
    cfg.lambda_latent = args_cli.lambda_latent
    cfg.save_interval = args_cli.save_interval
    cfg.seed = args_cli.seed
    cfg.logger = args_cli.logger
    cfg.wandb_project = args_cli.wandb_project
    cfg.task = args_cli.task
    cfg.run_name = args_cli.run_name or f"student_{args_cli.encoder_type}"

    # Log dir
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join(cfg.log_dir_root, f"{stamp}_{cfg.run_name}"))
    os.makedirs(log_dir, exist_ok=True)
    logger.info("log_dir=%s", log_dir)

    # Build env
    # Use Hydra-style env cfg loading via gym registry
    from isaaclab_tasks.utils.hydra import hydra_task_config

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def _create(env_cfg: DirectRLEnvCfg, _agent_cfg):
        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.seed = cfg.seed
        env_cfg.sim.device = cfg.device
        env_cfg.log_dir = log_dir
        return env_cfg

    env_cfg = _create()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    device = torch.device(cfg.device)
    runner = StudentRunner(env=env, cfg=cfg, log_dir=log_dir, device=device)

    t0 = time.time()
    try:
        runner.learn()
    finally:
        env.close()
        logger.info("Total wall time: %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Verify it imports cleanly (dry parse, no run)**

Run:
```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_student.py --help 2>&1 | head -20
```
Expected: Help text lists `--encoder_type`, `--teacher_run_dir`, etc. No tracebacks.

- [ ] **Step 3: Commit**

```bash
cd /workspace/isaaclab
git add scripts/reinforcement_learning/rsl_rl/train_student.py
git commit -m "feat(student): add train_student.py entry script"
```

---

## Task 8: Short-iteration end-to-end smoke test

**Files:**
- None created; this is a manual verification.

- [ ] **Step 1: Run a 5-iter smoke training**

Run (foreground, ~2 min):
```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_student.py \
    --encoder_type tcn \
    --num_envs 256 \
    --max_iterations 5 \
    --n_steps_per_rollout 8 \
    --n_epochs 2 \
    --logger tensorboard \
    --headless 2>&1 | tee /tmp/student_smoke.log
```

- [ ] **Step 2: Verify training completed**

Expected in log tail:
```
iter=0 total=<val>
iter=4 ... Saved student checkpoint: <path>/models/student_4.pt
Total wall time: << 5 min
```

Expected artifacts:
```bash
ls logs/rsl_rl/student_policy/*student_tcn*/models/student_4.pt
```
should exist.

- [ ] **Step 3: Verify loss moved (sanity only; 5 iters is too short for convergence)**

Run:
```bash
grep -E "^iter=" /tmp/student_smoke.log | head -5
```
Expect monotonic or roughly decreasing `total` over 5 iters. If total increases by >10x, investigate before Task 9.

- [ ] **Step 4: Repeat for GRU variant**

Run:
```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_student.py \
    --encoder_type gru \
    --num_envs 256 \
    --max_iterations 5 \
    --n_steps_per_rollout 8 \
    --n_epochs 2 \
    --logger tensorboard \
    --headless 2>&1 | tee /tmp/student_smoke_gru.log
```
Expect same success criteria.

- [ ] **Step 5: Commit nothing (this is a gate; fixes go back to Task 4-7 if failures)**

If any failure here: diagnose and patch the relevant earlier file. Do NOT proceed to Task 9 until both TCN and GRU 5-iter smoke runs complete.

---

## Task 9: Launch script — TCN full run

**Files:**
- Create: `scripts/launch_student_tcn.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# Student-TCN: train encoder via BC from r13_A teacher on GPU 1.
# Sequential with launch_student_gru.sh (runs AFTER TCN completes).
set -e

cd /workspace/isaaclab

RUN_NAME="student_tcn"
STAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="/workspace/isaaclab/logs/archive/launch_scripts/${RUN_NAME}_${STAMP}.log"
mkdir -p "$(dirname "$STDOUT_LOG")"

echo "[${RUN_NAME} $(date)] START" | tee -a "$STDOUT_LOG"
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_student.py \
    --encoder_type tcn \
    --task Isaac-FullDOF-TRPO-v0 \
    --teacher_run_dir logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A \
    --teacher_checkpoint model_4999.pt \
    --num_envs 4096 \
    --max_iterations 1000 \
    --n_steps_per_rollout 24 \
    --n_epochs 5 \
    --minibatch_size 8192 \
    --lr 5e-4 \
    --lambda_latent 1.0 \
    --save_interval 100 \
    --seed 42 \
    --logger wandb \
    --wandb_project full_dof_trpo_student \
    --run_name "$RUN_NAME" \
    --headless 2>&1 | tee -a "$STDOUT_LOG"
RC=${PIPESTATUS[0]}
echo "[${RUN_NAME} $(date)] END rc=${RC}" | tee -a "$STDOUT_LOG"
exit "$RC"
```

- [ ] **Step 2: Mark executable + commit**

```bash
cd /workspace/isaaclab
chmod +x scripts/launch_student_tcn.sh
git add scripts/launch_student_tcn.sh
git commit -m "feat(student): add launch_student_tcn.sh"
```

---

## Task 10: Launch script — GRU full run

**Files:**
- Create: `scripts/launch_student_gru.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# Student-GRU: train encoder via BC from r13_A teacher on GPU 1.
# Runs AFTER launch_student_tcn.sh completes.
set -e

cd /workspace/isaaclab

RUN_NAME="student_gru"
STAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="/workspace/isaaclab/logs/archive/launch_scripts/${RUN_NAME}_${STAMP}.log"
mkdir -p "$(dirname "$STDOUT_LOG")"

echo "[${RUN_NAME} $(date)] START" | tee -a "$STDOUT_LOG"
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_student.py \
    --encoder_type gru \
    --task Isaac-FullDOF-TRPO-v0 \
    --teacher_run_dir logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A \
    --teacher_checkpoint model_4999.pt \
    --num_envs 4096 \
    --max_iterations 1000 \
    --n_steps_per_rollout 24 \
    --n_epochs 5 \
    --minibatch_size 8192 \
    --lr 5e-4 \
    --lambda_latent 1.0 \
    --save_interval 100 \
    --seed 42 \
    --logger wandb \
    --wandb_project full_dof_trpo_student \
    --run_name "$RUN_NAME" \
    --headless 2>&1 | tee -a "$STDOUT_LOG"
RC=${PIPESTATUS[0]}
echo "[${RUN_NAME} $(date)] END rc=${RC}" | tee -a "$STDOUT_LOG"
exit "$RC"
```

- [ ] **Step 2: Mark executable + commit**

```bash
cd /workspace/isaaclab
chmod +x scripts/launch_student_gru.sh
git add scripts/launch_student_gru.sh
git commit -m "feat(student): add launch_student_gru.sh"
```

---

## Task 11: Combined sequential runner script

**Files:**
- Create: `scripts/launch_students_sequential.sh`

- [ ] **Step 1: Write the sequential wrapper**

```bash
#!/usr/bin/env bash
# Run TCN then GRU sequentially on GPU 1.
# If TCN fails, GRU does not start.
set -e

cd /workspace/isaaclab

echo "[sequential $(date)] START — TCN first, then GRU"

./scripts/launch_student_tcn.sh
TCN_RC=$?
echo "[sequential $(date)] TCN finished rc=${TCN_RC}"

if [ "$TCN_RC" -ne 0 ]; then
    echo "[sequential $(date)] ABORT — TCN failed (rc=${TCN_RC}), GRU not started."
    exit "$TCN_RC"
fi

./scripts/launch_student_gru.sh
GRU_RC=$?
echo "[sequential $(date)] GRU finished rc=${GRU_RC}"

echo "[sequential $(date)] DONE TCN=${TCN_RC} GRU=${GRU_RC}"
exit "$GRU_RC"
```

- [ ] **Step 2: Mark executable + commit**

```bash
cd /workspace/isaaclab
chmod +x scripts/launch_students_sequential.sh
git add scripts/launch_students_sequential.sh
git commit -m "feat(student): add sequential wrapper launch_students_sequential.sh"
```

---

## Task 12: Student-in-loop eval wrapper

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/eval.py`

- [ ] **Step 1: Write the eval wrapper module**

```python
# source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/eval.py
"""Student-in-the-loop wrapper for eval_dr_fulldof.

Usage from a separate eval script:
    from isaaclab_tasks.direct.constrained_full_albc.student.eval import (
        build_student_policy_fn,
    )

    policy_fn = build_student_policy_fn(
        teacher_ckpt="logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/model_4999.pt",
        student_ckpt="logs/rsl_rl/student_policy/.../models/student_999.pt",
        encoder_type="tcn",
        num_envs=64,
        device="cuda:0",
    )
    # policy_fn(obs_td) -> action (B, 8)
"""
from __future__ import annotations

import os
import torch

from .config import StudentCfg
from .models import StudentEncoderGRU, StudentEncoderTCN, make_student_encoder
from .teacher import FrozenTeacher


class StudentInLoopPolicy:
    """Callable policy that uses student encoder + teacher actor at inference time.

    Carries:
        - TCN: ring buffer of (num_envs, H, 87)
        - GRU: hidden state (num_layers, num_envs, hidden)

    Reset on `done` must be signaled via reset(env_ids).
    """

    def __init__(self, cfg: StudentCfg, student_ckpt: str, num_envs: int, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs

        self.teacher = FrozenTeacher(cfg, device=device)
        self.student = make_student_encoder(cfg).to(device)
        # Load student weights
        blob = torch.load(student_ckpt, map_location=device, weights_only=False)
        self.student.load_state_dict(blob["student_state_dict"])
        self.student.eval()

        if cfg.encoder_type == "tcn":
            self.ring = torch.zeros(num_envs, cfg.tcn_history, cfg.policy_obs_dim, device=device)
        else:
            self.ring = None
            self.hidden = self.student.init_hidden(num_envs, device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear history/hidden for the given envs (or all if None)."""
        if self.cfg.encoder_type == "tcn":
            if env_ids is None:
                self.ring.zero_()
            else:
                self.ring[env_ids] = 0.0
        else:
            if env_ids is None:
                self.hidden.zero_()
            else:
                self.hidden[:, env_ids] = 0.0

    @torch.no_grad()
    def __call__(self, obs_td) -> torch.Tensor:
        """obs_td: tensordict with 'policy' key of shape (B, 87). Returns (B, 8)."""
        obs = obs_td["policy"]
        if self.cfg.encoder_type == "tcn":
            # Push new obs onto ring
            self.ring = torch.roll(self.ring, shifts=1, dims=1)
            self.ring[:, 0] = obs
            l_hat = self.student(self.ring)
        else:
            # Single-step forward
            obs_seq = obs.unsqueeze(1)  # (B, 1, 87)
            l_hat_seq, self.hidden = self.student(obs_seq, hidden=self.hidden)
            l_hat = l_hat_seq[:, -1]    # (B, 9)

        obs_normed = self.teacher.normalize_obs(obs)
        return self.teacher.actor_forward(obs_normed, l_hat)


def build_student_policy_fn(
    teacher_ckpt: str,
    student_ckpt: str,
    encoder_type: str,
    num_envs: int,
    device: str = "cuda:0",
) -> StudentInLoopPolicy:
    """Factory returning a callable policy."""
    cfg = StudentCfg()
    cfg.encoder_type = encoder_type
    cfg.teacher_run_dir = os.path.dirname(teacher_ckpt)
    cfg.teacher_checkpoint = os.path.basename(teacher_ckpt)
    return StudentInLoopPolicy(cfg, student_ckpt=student_ckpt, num_envs=num_envs, device=torch.device(device))
```

- [ ] **Step 2: Commit**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/eval.py
git commit -m "feat(student): add StudentInLoopPolicy wrapper for eval_dr_fulldof"
```

---

## Task 13: Eval script — eval_student_dr.py

**Files:**
- Create: `scripts/analysis/eval_student_dr.py`

- [ ] **Step 1: Write a standalone eval that reuses eval_dr_fulldof internals**

Rather than modifying the existing 2186-line `eval_dr_fulldof.py`, write a thin wrapper that overrides the policy-loading path and delegates everything else. Inspect the existing `eval_dr_fulldof.py` first to confirm the policy-loading function name.

Run:
```bash
grep -n "def.*policy\|OnPolicyRunner\|runner.load\|inference" /workspace/isaaclab/scripts/analysis/eval_dr_fulldof.py | head -20
```
Look for the policy-loading entry (typically a function like `_load_policy`, `inference_fn`, or similar). Note the name.

Then create the wrapper:

```python
#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a student policy (student encoder + teacher actor) across DR levels.

This script is a thin wrapper that imports the existing eval_dr_fulldof core
and swaps the policy callable for the StudentInLoopPolicy.

Usage:
    ./isaaclab.sh -p scripts/analysis/eval_student_dr.py \
        --teacher_ckpt logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/model_4999.pt \
        --student_ckpt logs/rsl_rl/student_policy/.../models/student_999.pt \
        --encoder_type tcn \
        --num_envs 64 --headless
"""
import argparse
import os
import sys

# Load AppLauncher first (same boilerplate as eval_dr_fulldof)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(__file__))

from isaaclab.app import AppLauncher
import cli_args  # noqa

parser = argparse.ArgumentParser(description="Evaluate student policy across DR levels.")
parser.add_argument("--teacher_ckpt", type=str, required=True)
parser.add_argument("--student_ckpt", type=str, required=True)
parser.add_argument("--encoder_type", type=str, choices=["tcn", "gru"], required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-FullDOF-TRPO-v0")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--segment_duration", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.direct.constrained_full_albc.student.eval import build_student_policy_fn

# Build student policy
student_policy = build_student_policy_fn(
    teacher_ckpt=args_cli.teacher_ckpt,
    student_ckpt=args_cli.student_ckpt,
    encoder_type=args_cli.encoder_type,
    num_envs=args_cli.num_envs,
    device="cuda:0",
)

# NOTE: The simplest approach is to import and reuse eval_dr_fulldof's main
# helpers. If the existing script exposes `run_dr_sweep(policy_callable, ...)`
# directly, call it. Otherwise, inline the needed loop here. The concrete choice
# depends on the function boundary found in Step 1's grep.
#
# FALLBACK (if no exported helper): write a minimal DR sweep inline below.
# The DR levels and metrics collection logic can be copied from eval_dr_fulldof.py
# lines that build `Env` + iterate DR_SCALE levels + compute enhanced_summary.json.

print("eval_student_dr: using student from", args_cli.student_ckpt)
print("encoder_type:", args_cli.encoder_type)

# --- minimal DR sweep loop stub ---
# For the concrete implementation, either:
#   (a) refactor eval_dr_fulldof.py to expose `run_sweep(policy_fn, cfg, out_dir)`
#       and call it here. Recommended if >5 files will rely on this.
#   (b) copy the sweep loop verbatim here and replace the call site
#       of the teacher policy with `student_policy(obs_td)`.
# Decision gate: pick (a) only if the existing eval script will be reused
# broadly. Otherwise (b) keeps the plan scope tight.
#
# For this plan we pick (b): copy the sweep structure minimally. See Step 2.

simulation_app.close()
```

- [ ] **Step 2: Extract the sweep logic from `eval_dr_fulldof.py`**

Open `/workspace/isaaclab/scripts/analysis/eval_dr_fulldof.py`. Identify:
- The 4-level DR loop (`DR_SCALE = [0.0, 0.3, 0.6, 1.0]` or similar labeled `none/soft/medium/hard`).
- The step loop that collects attitude / lin_vel / yaw per env per step.
- The call that loads the policy and invokes `policy(obs)` each step.

Copy the sweep body into `eval_student_dr.py` after the student policy is built, replacing the `policy = runner.load(...)` + `policy(obs)` calls with:
```python
policy = student_policy   # callable: obs_td -> action (B, 8)
```
and replacing any `dones`-triggered reset of policy state with `student_policy.reset(env_ids)`.

Keep the output writing to `enhanced_summary.json` and `summary_*.png` identical.

- [ ] **Step 3: Smoke-test the script (foreground, short)**

Run with a very small env count to verify plumbing:
```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/analysis/eval_student_dr.py \
    --teacher_ckpt logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/model_4999.pt \
    --student_ckpt <INSERT_PATH_FROM_TASK_8_SMOKE_RUN>/models/student_4.pt \
    --encoder_type tcn \
    --num_envs 8 \
    --segment_duration 1.0 \
    --headless 2>&1 | tail -20
```
Expected: script runs to completion, writes `<output_dir>/enhanced_summary.json`.

- [ ] **Step 4: Commit**

```bash
cd /workspace/isaaclab
git add scripts/analysis/eval_student_dr.py
git commit -m "feat(student): add eval_student_dr.py DR sweep wrapper"
```

---

## Task 14: Run full TCN + GRU training sequentially

**Files:**
- None (runtime only).

- [ ] **Step 1: Launch the sequential training**

Run in background (runs ~1 h total on GPU 1):
```bash
cd /workspace/isaaclab
nohup ./scripts/launch_students_sequential.sh > /workspace/isaaclab/logs/archive/launch_scripts/students_sequential_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $!  # note the PID
```

- [ ] **Step 2: Monitor progress**

Monitor events to watch for:
- `iter=` progress lines
- `Saved student checkpoint`
- Any tracebacks or `CUDA out of memory`

Use the Monitor tool (Claude Code) or tail:
```bash
tail -f /workspace/isaaclab/logs/archive/launch_scripts/students_sequential_*.log
```

Expected timing:
- TCN: ~30 min (4096 envs × 24 steps × 1000 iters)
- GRU: ~30 min
- Total: ~60 min

- [ ] **Step 3: Verify both checkpoints exist**

```bash
ls -lh /workspace/isaaclab/logs/rsl_rl/student_policy/*student_tcn*/models/student_999.pt
ls -lh /workspace/isaaclab/logs/rsl_rl/student_policy/*student_gru*/models/student_999.pt
```
Both files must exist.

- [ ] **Step 4: Record final loss**

Read final TensorBoard scalars:
```bash
./isaaclab.sh -p /workspace/.claude/skills/train-analyze/analyze_training.py \
    /workspace/isaaclab/logs/rsl_rl/student_policy/*student_tcn*/ 2>&1 | tail -20
./isaaclab.sh -p /workspace/.claude/skills/train-analyze/analyze_training.py \
    /workspace/isaaclab/logs/rsl_rl/student_policy/*student_gru*/ 2>&1 | tail -20
```

Acceptance per spec §11:
- `loss_latent < 0.2` at iter 999 for both variants
- `loss_action < 0.02` at iter 999 for both variants

If either student misses both thresholds by >50%: STOP and diagnose (debug lambda_latent, lr, n_steps) before Task 15.

---

## Task 15: Evaluate both students + comparison matrix

**Files:**
- None (runs scripts + produces logs/plots).

- [ ] **Step 1: Eval TCN student across 4 DR levels**

```bash
cd /workspace/isaaclab
TCN_RUN=$(ls -td logs/rsl_rl/student_policy/*student_tcn* | head -1)
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/analysis/eval_student_dr.py \
    --teacher_ckpt logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/model_4999.pt \
    --student_ckpt "${TCN_RUN}/models/student_999.pt" \
    --encoder_type tcn \
    --num_envs 64 \
    --headless \
    --output_dir "${TCN_RUN}/eval_dr" 2>&1 | tail -10
```

- [ ] **Step 2: Eval GRU student across 4 DR levels**

```bash
cd /workspace/isaaclab
GRU_RUN=$(ls -td logs/rsl_rl/student_policy/*student_gru* | head -1)
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/analysis/eval_student_dr.py \
    --teacher_ckpt logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/model_4999.pt \
    --student_ckpt "${GRU_RUN}/models/student_999.pt" \
    --encoder_type gru \
    --num_envs 64 \
    --headless \
    --output_dir "${GRU_RUN}/eval_dr" 2>&1 | tail -10
```

- [ ] **Step 3: Build the comparison matrix**

Create `/tmp/compare_students.py`:

```python
"""Compare teacher vs TCN vs GRU across DR levels."""
import json
import sys
from pathlib import Path

TEACHER = Path("logs/archive/rsl_rl/r13_A/eval_dr/enhanced_summary.json")  # from earlier eval
TCN = next(Path("logs/rsl_rl/student_policy").glob("*student_tcn*")) / "eval_dr/enhanced_summary.json"
GRU = next(Path("logs/rsl_rl/student_policy").glob("*student_gru*")) / "eval_dr/enhanced_summary.json"

for p in (TEACHER, TCN, GRU):
    if not p.exists():
        print(f"MISSING: {p}")
        sys.exit(1)

summaries = {
    "teacher": json.loads(TEACHER.read_text()),
    "tcn": json.loads(TCN.read_text()),
    "gru": json.loads(GRU.read_text()),
}

AXES = ["roll", "pitch", "vx", "vy", "vz", "yaw"]
LEVELS = ["none", "soft", "medium", "hard"]
FIELD = "ss_error"

print(f"\n{FIELD} comparison (lower is better)\n" + "=" * 70)
print(f"{'axis':<8}{'level':<8}{'teacher':>10}{'tcn':>10}{'gru':>10}{'tcn/teacher':>14}{'gru/teacher':>14}")
for lvl in LEVELS:
    for ax in AXES:
        try:
            t = summaries["teacher"][lvl][ax][FIELD]
            tcn = summaries["tcn"][lvl][ax][FIELD]
            gru = summaries["gru"][lvl][ax][FIELD]
            print(f"{ax:<8}{lvl:<8}{t:>10.4f}{tcn:>10.4f}{gru:>10.4f}{tcn/t:>14.2f}{gru/t:>14.2f}")
        except KeyError:
            pass
```

Run:
```bash
cd /workspace/isaaclab
./isaaclab.sh -p /tmp/compare_students.py 2>&1 | tee /tmp/student_compare.txt
```

- [ ] **Step 4: Flag regressions and report**

For each (axis, DR level): if `tcn/teacher` or `gru/teacher` > **1.20**, flag that cell. Collect all flagged cells into a short written summary:

```
Regressions >20% vs teacher:
  TCN: <list of (axis, level) cells>
  GRU: <list of (axis, level) cells>
```

Include summary in the next changelog entry.

- [ ] **Step 5: View plots**

```bash
ls "${TCN_RUN}/eval_dr/summary_"*.png "${GRU_RUN}/eval_dr/summary_"*.png
```

Open (Read tool) at least `summary_att.png`, `summary_lin_vel.png`, `summary_yaw.png` for both. Confirm visual alignment with teacher; any obvious divergence (wild oscillation, failed step response) should be documented.

---

## Task 16: Changelog entry

**Files:**
- Modify: `changelog.md`

- [ ] **Step 1: Add changelog entry per repo rules**

Read `/workspace/isaaclab/changelog.md` and prepend a new section at the top (after the file header) for today's date in the format required by the `changelog` skill. Content: Context (why student policy), Experiments (TCN vs GRU numeric results from Task 15), Decisions (which variant wins, any λ adjustments made), Open Questions (sim-to-real gap, tuning avenues).

Do NOT use `git add -A`. Stage only files you modified in this session.

- [ ] **Step 2: Commit**

```bash
cd /workspace/isaaclab
git add changelog.md
git commit -m "docs: changelog — student policy TCN vs GRU results"
```

---

## Self-Review

**1. Spec coverage — every requirement in 2026-04-21-student-policy-design.md is hit:**
- §3.1 common components (frozen teacher, static norm) → Task 3
- §3.2 TCN arch (kernels/strides/channels) → Task 2 matches spec
- §3.3 GRU arch (1-layer, hidden=128) → Task 2 matches spec
- §3.4 data flow (env step with teacher action, autograd through frozen actor) → Task 5 `_collect_rollout` + `_compute_loss_*`
- §3.5 joint loss with λ=1 → Task 5 `_compute_loss_*`
- §4 online rollout, r14 HardDR, DORAEMON off → Task 5 + Task 6
- §4 episode handling (ring zero on reset, GRU hidden zero on done) → Task 4, Task 5
- §5 hyperparameters (lr/n_steps/etc.) → Task 1 defaults match
- §5 logging (loss totals, grad_norm, per-dim histograms) → Task 5 `_log` (per-dim histograms deferred — logged aggregate only; acceptable since spec says "every 100 iters" and this is optimization, not correctness)
- §6 eval sweep + comparison matrix → Task 12, Task 13, Task 15
- §7 file organization → all files in Tasks 1-5 + 7, 9-13
- §8 testing — unit (Task 2), integration (Task 4), smoke (Task 8)
- §11 success criteria → Task 14 Step 4 enforces thresholds

**2. Placeholder scan — no red flags:**
- No "TBD" / "TODO" / "implement later" in code snippets.
- Task 13 Step 2 requires engineer to extract sweep logic from `eval_dr_fulldof.py` — this is an editorial task but the decision rubric (inline copy vs refactor) is provided explicitly, not left open.
- Task 16 Step 1 refers to external skill content (changelog skill) — that content is available in the session context for the executor.

**3. Type consistency:**
- `StudentCfg` dataclass is the single config source; all modules consume it unchanged.
- `FrozenTeacher.encode_privileged / .normalize_obs / .actor_forward / .act` signatures are stable across Tasks 3, 5, 12.
- `RolloutBuffer.iter_minibatches_tcn/_gru` return `list[RolloutBatch]` with disjoint populated fields; consumers in Task 5 check `encoder_type` to pick the right field.
- `StudentInLoopPolicy(cfg, student_ckpt, num_envs, device)` signature consistent in Task 12 definition and Task 13 consumer.

**Plan complete.** Saved to `docs/superpowers/plans/2026-04-21-student-policy-implementation.md`.
