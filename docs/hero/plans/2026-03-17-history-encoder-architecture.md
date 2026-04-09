# History-Augmented Encoder Architecture Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared proprioception history (TCN) to the encoder pipeline so that all modules (encoder, actor, critics) benefit from dynamic response context, replacing the current privileged-only encoder that produces near-constant z.

**Architecture:**
```
Proprio History (30, 8D) --> HistoryTCN --> h_embed (32D)   [shared module]

Encoder:       [policy_obs(13D), h_embed(32D), privileged(19D)] --> z(13D)
Actor:         [policy_obs(13D), h_embed(32D), z(13D)]          --> actions(2D)
Reward Critic: [policy_obs(13D), h_embed(32D), privileged(19D)] --> value(1D)
Cost Critic:   [policy_obs(13D), h_embed(32D), privileged(19D)] --> cost_values(K)
```

Actor alone cannot see privileged info -- it accesses DR parameters only through z.
Encoder, reward critic, and cost critic all see the full input including privileged.
HistoryTCN is a shared module trained end-to-end with all losses.

**Tech Stack:** PyTorch, RSL-RL, Isaac Lab DirectRLEnv, TensorDict

**Motivation:** Encoder z sweep analysis (2026-03-17) showed 10/13 z dimensions near-constant, cosine similarity 0.9482 across DR conditions, max |r|=0.239 with physics params. The encoder fails to encode DR parameters from privileged info alone because static hydrodynamic parameters don't reveal dynamic response characteristics. Adding proprioception history provides the command-response relationship (same action + different physics -> different angular velocity) that the encoder needs.

**Phase 2 impact:** The shared TCN trained in Phase 1 carries over to Phase 2. The adaptation module becomes a simple `adapt_head: h_embed(32D) --> z_hat(13D)` instead of a full TCN from scratch. The deployment path (sim-to-real) requires only shared_tcn + adapt_head + actor -- none of which need privileged info.

---

## File Structure

### New files
| File | Responsibility |
|------|---------------|
| `encoder/history_tcn.py` | `HistoryTCN` module: proprio_hist (N,30,8) --> h_embed (N,32). Extracted and generalized from ProprioAdaptTConv. |

### Modified files
| File | Changes |
|------|---------|
| `encoder/actor_critic_encoder.py` | Optional HistoryTCN when obs has "proprio_hist" key. Expanded encoder/actor/critic input dims. New obs_groups parsing (2 or 3 keys). |
| `encoder/actor_critic_encoder_constrained.py` | cost_critic input dim auto-adjusts via inherited `num_critic_obs`. |
| `encoder/adaptation.py` | Phase 2 reuses parent's shared TCN. `adapt_head: h_embed --> z_hat` replaces full ProprioAdaptTConv. |
| `encoder/__init__.py` | Export `HistoryTCN`. |
| `base_env.py` | Add proprio history ring buffer. Add "proprio_hist" to observations when `proprio_history_len > 0`. |
| `adapt_base_env.py` | Remove history buffer (now in base_env). Simplify to config-only subclass. |
| `config.py` | Add `proprio_history_len/proprio_feature_dim` to `HeroAgentEncoderTrainEnvCfg`. New constrained config with history. |
| `agents/rsl_rl_ppo_cfg.py` | Add `h_embed_dim` to policy configs. New obs_groups with 3 keys. |

---

## Chunk 1: HistoryTCN Module + Base Env History Buffer

### Task 1: Create HistoryTCN module

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/history_tcn.py`

- [ ] **Step 1: Create HistoryTCN class**

Extract and generalize from `ProprioAdaptTConv` (adaptation.py:54-115). Same TCN architecture, parameterized output_dim for h_embed (default 32 instead of 13).

```python
# encoder/history_tcn.py

"""Shared temporal convolution for proprioception history embedding.

Processes a sliding window of proprioceptive features (roll, pitch, p, q,
joint_pos_norm, prev_actions) through channel transform + temporal Conv1d
into a fixed-size embedding used by encoder, actor, and critics.

Architecture:
    Input: (N, H=30, D=8)
    -> channel_transform: Linear(8->32) -> ReLU -> Linear(32->32) -> ReLU
    -> temporal_conv: Conv1d x3 (kernels=[9,5,5], strides=[2,1,1])
    -> embed_proj: Linear(32*3 -> h_embed_dim)
    -> h_embed (N, h_embed_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _compute_conv_output_len(length: int, kernels: list[int], strides: list[int]) -> int:
    """Compute output temporal length after a sequence of Conv1d layers."""
    for k, s in zip(kernels, strides):
        length = (length - k) // s + 1
    return length


class HistoryTCN(nn.Module):
    """Shared temporal convolution: proprio_hist -> h_embed.

    Same architecture as ProprioAdaptTConv but with configurable output_dim
    for producing a general-purpose history embedding (not z directly).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 32,
        history_len: int = 30,
    ):
        super().__init__()

        kernels = [9, 5, 5]
        strides = [2, 1, 1]

        final_time_steps = _compute_conv_output_len(history_len, kernels, strides)
        if final_time_steps < 1:
            raise ValueError(
                f"Conv kernels {kernels} with strides {strides} produce 0 output "
                f"for history_len={history_len}. Use smaller kernels."
            )

        self.channel_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        for k, s in zip(kernels, strides):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, k, stride=s))
            layers.append(nn.ReLU(inplace=True))
        self.temporal_aggregation = nn.Sequential(*layers)

        self.embed_proj = nn.Linear(hidden_dim * final_time_steps, output_dim)

        # Small init: h_embed starts near zero, minimal disruption at init.
        nn.init.constant_(self.embed_proj.bias, 0.0)
        nn.init.normal_(self.embed_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (N, H, input_dim) -> (N, output_dim).

        Args:
            x: Proprioception history. Shape: (N, H, input_dim).

        Returns:
            History embedding. Shape: (N, output_dim).
        """
        x = self.channel_transform(x)  # (N, H, hidden_dim)
        x = x.permute(0, 2, 1)  # (N, hidden_dim, H) for Conv1d
        x = self.temporal_aggregation(x)  # (N, hidden_dim, T_final)
        return self.embed_proj(x.flatten(1))  # (N, output_dim)
```

- [ ] **Step 2: Verify file created correctly**

Run: `ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/history_tcn.py`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/history_tcn.py
git commit -m "feat(encoder): add HistoryTCN shared module for proprio history embedding"
```

---

### Task 2: Add proprio history buffer to base_env.py

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/base_env.py`

The history buffer currently lives only in `adapt_base_env.py`. Move it to `base_env.py` so all encoder configs can use it. Gate on `cfg.proprio_history_len > 0`.

- [ ] **Step 1: Add config fields to HeroAgentEnvCfg**

In `config.py`, add fields to the base config class `HeroAgentEnvCfg` (around line 220):

```python
# Proprioception history (for encoder with history augmentation)
proprio_history_len: int = 0  # 0 = disabled, 30 = default for history encoder
proprio_feature_dim: int = 8  # [roll, pitch, p, q, joint_pos_norm(2), prev_actions(2)]
```

Default `0` means history is disabled (backward compatible). Encoder configs override to `30`.

- [ ] **Step 2: Add history buffer initialization in base_env._init_state_buffers()**

After action latency buffer init (line ~422), add:

```python
# Proprioception history buffer (ring buffer for temporal conv encoder)
self._proprio_history_len = self.cfg.proprio_history_len
if self._proprio_history_len > 0:
    self._proprio_hist = torch.zeros(
        self.num_envs, self._proprio_history_len, self.cfg.proprio_feature_dim,
        device=self.device,
    )
else:
    self._proprio_hist = None
```

- [ ] **Step 3: Add _update_proprio_hist() to base_env**

Add method after `_get_proprio_features()` (line ~465):

```python
def _update_proprio_hist(self) -> None:
    """Shift ring buffer left and append current proprioception features.

    Only active when proprio_history_len > 0. Called from _pre_physics_step()
    before the control pipeline runs.
    """
    if self._proprio_hist is None:
        return
    roll, pitch, p, q, joint_pos_norm = self._get_proprio_features()
    new_entry = torch.cat(
        [roll.unsqueeze(-1), pitch.unsqueeze(-1), p, q, joint_pos_norm, self._prev_actions_obs],
        dim=-1,
    )
    self._proprio_hist[:, :-1] = self._proprio_hist[:, 1:].clone()
    self._proprio_hist[:, -1] = new_entry
```

- [ ] **Step 4: Call _update_proprio_hist() in _pre_physics_step()**

At the beginning of `_pre_physics_step()` (before action buffer updates), add:

```python
self._update_proprio_hist()
```

- [ ] **Step 5: Add "proprio_hist" to _get_observations()**

In `_get_observations()` (line ~825), after privileged obs:

```python
if self._proprio_hist is not None:
    observations["proprio_hist"] = self._proprio_hist.clone()
```

- [ ] **Step 6: Reset history buffer in _reset_idx()**

In `_reset_idx()`, after existing buffer resets:

```python
if self._proprio_hist is not None:
    self._proprio_hist[env_ids_] = 0.0
```

- [ ] **Step 7: Verify lint**

Run: `ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/base_env.py`

- [ ] **Step 8: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/base_env.py
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/config.py
git commit -m "feat(base_env): add proprio history ring buffer gated by config"
```

---

### Task 3: Simplify adapt_base_env.py

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/adapt_base_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/config.py`

History buffer now lives in base_env. `HeroAgentAdaptBaseEnv` only needs to exist if it has additional logic beyond the buffer. Since it doesn't, it can be simplified.

- [ ] **Step 1: Update HeroAgentAdaptBaseEnvCfg to set proprio_history_len=30**

In `config.py`, `HeroAgentAdaptBaseEnvCfg` (line ~555) already has `proprio_history_len: int = 30`. This will now activate the base_env buffer. Remove `proprio_feature_dim` from here if it's already in the base class.

- [ ] **Step 2: Simplify adapt_base_env.py**

Remove the buffer init, `_get_observations()`, `_pre_physics_step()`, `_update_proprio_hist()`, and `_reset_idx()` overrides since they're now handled by base_env:

```python
"""Hero Agent Phase 2 Adaptation Environment (base RL pipeline).

Extends the base RL env with proprioception history for training
the adaptation module (student). History buffer managed by base_env
when proprio_history_len > 0.
"""

from __future__ import annotations

from .base_env import HeroAgentEnv
from .config import HeroAgentAdaptBaseEnvCfg


class HeroAgentAdaptBaseEnv(HeroAgentEnv):
    """Phase 2 adaptation environment.

    With proprio_history_len=30 set in HeroAgentAdaptBaseEnvCfg,
    the base class automatically manages the history buffer and
    includes "proprio_hist" in observations.
    """

    cfg: HeroAgentAdaptBaseEnvCfg
```

- [ ] **Step 3: Verify lint**

Run: `ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/adapt_base_env.py`

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/adapt_base_env.py
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/config.py
git commit -m "refactor(adapt_base_env): simplify, history buffer now in base_env"
```

---

## Chunk 2: ActorCriticEncoder History Support

### Task 4: Modify ActorCriticEncoder for optional history input

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/actor_critic_encoder.py`

This is the core architectural change. When obs_groups has 3 keys (policy, privileged, proprio_hist), the encoder creates a shared HistoryTCN and expands all input dimensions.

- [ ] **Step 1: Add HistoryTCN import and new __init__ parameters**

Add import at top:
```python
from .history_tcn import HistoryTCN
```

Add new parameters to `__init__()` after `asymmetric_critic`:
```python
# History encoder parameters (activated when obs has "proprio_hist" key)
h_embed_dim: int = 0,
proprio_history_len: int = 30,
proprio_feature_dim: int = 8,
```

`h_embed_dim=0` means no history (backward compatible). When > 0, creates HistoryTCN.

- [ ] **Step 2: Update obs_groups parsing (lines 122-130)**

Replace the strict 2-key assertion with flexible parsing:

```python
# Extract obs key names from obs_groups for direct TensorDict access
policy_groups = obs_groups["policy"]
if len(policy_groups) < 2:
    raise ValueError(
        f"ActorCriticEncoder requires at least 2 obs groups in 'policy', "
        f"got {len(policy_groups)}: {policy_groups}"
    )
self._policy_obs_key = policy_groups[0]
self._privileged_key = policy_groups[1]

# Optional history key (3rd element in obs_groups)
self._has_history = len(policy_groups) > 2 and h_embed_dim > 0
if self._has_history:
    self._proprio_hist_key = policy_groups[2]
else:
    self._proprio_hist_key = None
```

- [ ] **Step 3: Add HistoryTCN creation and normalizer**

After obs dimension verification (line ~141), add:

```python
# Shared history TCN (optional, activated by h_embed_dim > 0)
self.h_embed_dim = h_embed_dim
if self._has_history:
    proprio_hist_shape = obs[self._proprio_hist_key].shape
    if len(proprio_hist_shape) != 3:
        raise ValueError(
            f"proprio_hist must be 3D (batch, history, features), got shape {proprio_hist_shape}"
        )
    self.shared_tcn = HistoryTCN(
        input_dim=proprio_feature_dim,
        hidden_dim=32,
        output_dim=h_embed_dim,
        history_len=proprio_history_len,
    )
    self.hist_normalizer = EmpiricalNormalization(proprio_feature_dim)
    logger.info("HistoryTCN: %dD input x %d steps -> %dD h_embed", proprio_feature_dim, proprio_history_len, h_embed_dim)
else:
    self.shared_tcn = None
    self.hist_normalizer = None
```

- [ ] **Step 4: Update encoder input dimension**

Change encoder MLP creation (line ~144):

```python
# Encoder input: privileged (+ optional policy_obs + h_embed)
if self._has_history:
    encoder_input_dim = policy_obs_dim + h_embed_dim + privileged_dim
else:
    encoder_input_dim = privileged_dim
```

Replace `privileged_dim` with `encoder_input_dim` in MLP construction:
```python
if encoder_output_activation == "tanh":
    self.encoder = MLP(
        encoder_input_dim,  # was: privileged_dim
        encoder_latent_dim,
        list(encoder_hidden_dims),
        encoder_activation,
        last_activation="tanh",
    )
else:
    self.encoder = MLP(encoder_input_dim, encoder_latent_dim, list(encoder_hidden_dims), encoder_activation)
```

- [ ] **Step 5: Update actor input dimension**

Change actor input (line ~170):

```python
# Actor input: policy_obs + z (+ optional h_embed)
if self._has_history:
    num_actor_obs = policy_obs_dim + h_embed_dim + encoder_latent_dim
else:
    num_actor_obs = policy_obs_dim + encoder_latent_dim
```

- [ ] **Step 6: Update critic input dimension**

Change critic input (line ~181):

```python
# Critic input: policy_obs + privileged/z (+ optional h_embed)
# With history: critic always sees privileged (inherently asymmetric)
if self._has_history:
    num_critic_obs = policy_obs_dim + h_embed_dim + privileged_dim
elif asymmetric_critic:
    num_critic_obs = policy_obs_dim + privileged_dim
else:
    num_critic_obs = policy_obs_dim + encoder_latent_dim
```

Note: With history, critic is inherently asymmetric (sees privileged directly, not z). The `asymmetric_critic` flag becomes redundant when history is enabled.

- [ ] **Step 7: Add _compute_h_embed() method**

After `_activate_z()` method:

```python
def _compute_h_embed(self, obs: TensorDict) -> torch.Tensor:
    """Compute history embedding from proprioception history.

    Normalizes per-feature, then passes through shared TCN.

    Args:
        obs: TensorDict containing proprio_hist key.

    Returns:
        h_embed: History embedding. Shape: (N, h_embed_dim).
    """
    proprio_hist = obs[self._proprio_hist_key]
    N, H, D = proprio_hist.shape
    flat = proprio_hist.reshape(N * H, D)
    flat_norm = self.hist_normalizer(flat)
    proprio_hist_norm = flat_norm.reshape(N, H, D)
    return self.shared_tcn(proprio_hist_norm)
```

- [ ] **Step 8: Update _encode() for expanded input**

Modify `_encode()` (line ~242) to accept policy_obs and h_embed when history is active:

```python
def _encode(self, obs: TensorDict, *, store_z: bool = False) -> torch.Tensor:
    """Encode into latent z.

    Without history: encoder(privileged) -> z
    With history: encoder(cat([policy_obs, h_embed, privileged])) -> z
    """
    privileged = obs[self._privileged_key]
    if self._has_history:
        policy_obs = obs[self._policy_obs_key]
        h_embed = self._compute_h_embed(obs)
        encoder_input = torch.cat([policy_obs, h_embed, privileged], dim=-1)
    else:
        encoder_input = privileged

    normalized = self.encoder_obs_normalizer(encoder_input)
    if self.encoder_output_activation == "tanh":
        z = self.encoder(normalized)
    else:
        z = self._activate_z(self.encoder(normalized))
    if store_z:
        self._last_z = z
    return z
```

- [ ] **Step 9: Update _get_combined_obs() for actor input**

Modify `_get_combined_obs()` (line ~280):

```python
def _get_combined_obs(self, obs: TensorDict, *, store_z: bool = False) -> torch.Tensor:
    """Combined observation for actor: cat([policy_obs, (h_embed,) z])."""
    policy_obs = obs[self._policy_obs_key]
    z = self._encode(obs, store_z=store_z)
    if self._has_history:
        h_embed = self._compute_h_embed(obs)
        return torch.cat([policy_obs, h_embed, z], dim=-1)
    return torch.cat([policy_obs, z], dim=-1)
```

**Important:** `_compute_h_embed()` is called twice (in `_encode` and here). This is acceptable because the TCN is lightweight and consistent -- both calls use the same obs. To optimize, we could cache h_embed per forward pass, but this adds complexity for minimal gain. Keep it simple for now.

**Optimization note:** If profiling shows the double TCN forward is a bottleneck, add `self._cached_h_embed` that is computed once per `act()`/`evaluate()` call. This is a future optimization, not needed now.

- [ ] **Step 10: Update _get_critic_obs() for critic input**

Modify `_get_critic_obs()` (line ~292):

```python
def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
    """Critic observation.

    With history: cat([policy_obs, h_embed, privileged]) -- always asymmetric.
    Without history + asymmetric: cat([policy_obs, privileged]).
    Without history + symmetric: cat([policy_obs, z]).
    """
    if self._has_history:
        policy_obs = obs[self._policy_obs_key]
        h_embed = self._compute_h_embed(obs)
        privileged_raw = obs[self._privileged_key]
        return torch.cat([policy_obs, h_embed, privileged_raw], dim=-1)
    if not self.asymmetric_critic:
        return self._get_combined_obs(obs)
    policy_obs = obs[self._policy_obs_key]
    privileged_raw = obs[self._privileged_key]
    return torch.cat([policy_obs, privileged_raw], dim=-1)
```

- [ ] **Step 11: Update act_with_z_hat() for Phase 2 compatibility**

Modify `act_with_z_hat()` (line ~329):

```python
@torch.no_grad()
def act_with_z_hat(self, obs: TensorDict, z_hat: torch.Tensor) -> torch.Tensor:
    """Get deterministic action using a pre-computed z_hat."""
    policy_obs = obs[self._policy_obs_key]
    if self._has_history:
        h_embed = self._compute_h_embed(obs)
        combined_obs = torch.cat([policy_obs, h_embed, z_hat.detach()], dim=-1)
    else:
        combined_obs = torch.cat([policy_obs, z_hat.detach()], dim=-1)
    actor_obs = self.actor_obs_normalizer(combined_obs)
    return self.actor(actor_obs).clamp(-1.0, 1.0)
```

- [ ] **Step 12: Update update_normalization() for history normalizer**

Add history normalizer update to `update_normalization()`:

```python
def update_normalization(self, obs: TensorDict) -> None:
    """Update observation normalization statistics."""
    # History normalizer (shared TCN input)
    if self._has_history and hasattr(self.hist_normalizer, "update"):
        proprio_hist = obs[self._proprio_hist_key]
        N, H, D = proprio_hist.shape
        self.hist_normalizer.update(proprio_hist.reshape(N * H, D))

    # Encoder input normalizer
    if self.encoder_obs_normalization and hasattr(self.encoder_obs_normalizer, "update"):
        # Build the same encoder input that _encode() uses
        if self._has_history:
            policy_obs = obs[self._policy_obs_key]
            h_embed = self._compute_h_embed(obs)
            privileged = obs[self._privileged_key]
            encoder_input = torch.cat([policy_obs, h_embed, privileged], dim=-1)
        else:
            encoder_input = obs[self._privileged_key]
        self.encoder_obs_normalizer.update(encoder_input)

    combined = self._get_combined_obs(obs)
    if self.actor_obs_normalization and hasattr(self.actor_obs_normalizer, "update"):
        self.actor_obs_normalizer.update(combined)
    if self.critic_obs_normalization and hasattr(self.critic_obs_normalizer, "update"):
        critic_input = self._get_critic_obs(obs)
        self.critic_obs_normalizer.update(critic_input)
```

- [ ] **Step 13: Update encoder_obs_normalizer dimension**

In `__init__()`, the encoder_obs_normalizer dimension must match the encoder input:

```python
self.encoder_obs_normalizer = (
    EmpiricalNormalization(encoder_input_dim) if encoder_obs_normalization else nn.Identity()
)
```

(Change from `EmpiricalNormalization(privileged_dim)` to `EmpiricalNormalization(encoder_input_dim)`)

- [ ] **Step 14: Verify lint**

Run: `ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/actor_critic_encoder.py`

- [ ] **Step 15: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/actor_critic_encoder.py
git commit -m "feat(encoder): add optional HistoryTCN with expanded input dims"
```

---

### Task 5: Update ActorCriticEncoderConstrained

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/actor_critic_encoder_constrained.py`

The cost_critic uses `self.num_critic_obs` from the parent, which now auto-adjusts when history is enabled. Only the `evaluate_costs()` method needs to be verified -- it already calls `self._get_critic_obs(obs)` which handles history.

- [ ] **Step 1: Verify no code changes needed**

`ActorCriticEncoderConstrained.__init__()` does:
```python
num_critic_obs = self.num_critic_obs  # Set by parent
self.cost_critic = MLP(num_critic_obs, num_constraints, ...)
```

The parent's `num_critic_obs` already includes `h_embed_dim` when history is enabled. `evaluate_costs()` calls `self._get_critic_obs(obs)` which includes h_embed. **No changes needed.**

- [ ] **Step 2: Commit (if any formatting changes)**

```bash
# Only if there were changes
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/actor_critic_encoder_constrained.py
git commit -m "chore(encoder): verify constrained critic compatible with history"
```

---

### Task 6: Update ActorCriticEncoderAdapt for new Phase 2

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/adaptation.py`

Phase 2 now reuses the parent's shared TCN. Instead of a full ProprioAdaptTConv, use a simple `adapt_head: h_embed -> z_hat`.

- [ ] **Step 1: Simplify ActorCriticEncoderAdapt**

Replace the full ProprioAdaptTConv with a lightweight adapt_head:

```python
class ActorCriticEncoderAdapt(ActorCriticEncoder):
    """Phase 2 adaptation: adapt_head replaces encoder for z estimation.

    Uses the parent's shared_tcn (HistoryTCN) to produce h_embed,
    then a simple linear adapt_head maps h_embed -> z_hat.

    Phase 1 shared_tcn is frozen; only adapt_head trains.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not self._has_history:
            raise ValueError(
                "ActorCriticEncoderAdapt requires history support. "
                "Set h_embed_dim > 0 and provide proprio_hist in obs_groups."
            )

        # Simple adapt head: h_embed -> z_hat (replaces full TCN)
        self.adapt_head = nn.Sequential(
            nn.Linear(self.h_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.encoder_latent_dim),
        )
        # Small init: z_hat starts near zero -> _activate_z(0) = tanh(0) = 0
        nn.init.constant_(self.adapt_head[-1].bias, 0.0)
        nn.init.normal_(self.adapt_head[-1].weight, std=0.01)

    def compute_z_hat(self, obs: TensorDict) -> torch.Tensor:
        """Compute z_hat: shared_tcn(hist) -> h_embed -> adapt_head -> z_hat."""
        h_embed = self._compute_h_embed(obs)
        z_hat_raw = self.adapt_head(h_embed)
        return self._activate_z(z_hat_raw)

    def _get_combined_obs(self, obs: TensorDict, *, store_z: bool = False) -> torch.Tensor:
        """Actor obs: use z_hat from adapt_head (detached)."""
        policy_obs = obs[self._policy_obs_key]
        h_embed = self._compute_h_embed(obs)
        z_hat = self.compute_z_hat(obs)
        return torch.cat([policy_obs, h_embed, z_hat.detach()], dim=-1)

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Critic uses z_hat, matching actor's state representation."""
        policy_obs = obs[self._policy_obs_key]
        h_embed = self._compute_h_embed(obs)
        z_hat = self.compute_z_hat(obs)
        critic_obs = torch.cat([policy_obs, h_embed, z_hat.detach()], dim=-1)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def compute_z_gt(self, obs: TensorDict) -> torch.Tensor:
        """Compute ground truth z from frozen Phase 1 encoder."""
        with torch.no_grad():
            return self._encode(obs)

    def get_adapt_parameters(self):
        """Return only adapt_head parameters (for optimizer)."""
        return self.adapt_head.parameters()

    def freeze_base(self):
        """Freeze all weights except adapt_head."""
        for name, param in self.named_parameters():
            if "adapt_head" not in name:
                param.requires_grad = False
```

**Key changes from current:**
- `ProprioAdaptTConv` replaced by `adapt_head` (2-layer MLP, h_embed -> z_hat)
- Reuses `self._compute_h_embed()` from parent (shared TCN)
- `compute_z_gt()` uses `self._encode(obs)` which now accepts full obs dict
- `hist_normalizer` removed (handled by parent's `_compute_h_embed`)

**Note:** Keep `ProprioAdaptTConv` class in adaptation.py for now (it's still referenced by legacy checkpoints / old Phase 2 configs). It can be deprecated later.

- [ ] **Step 2: Verify lint**

Run: `ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/adaptation.py`

- [ ] **Step 3: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/adaptation.py
git commit -m "feat(adaptation): reuse shared TCN, adapt_head replaces full ProprioAdaptTConv"
```

---

### Task 7: Update encoder __init__.py exports

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/__init__.py`

- [ ] **Step 1: Add HistoryTCN export**

```python
from .actor_critic_encoder import ActorCriticEncoder
from .actor_critic_encoder_constrained import ActorCriticEncoderConstrained
from .adaptation import ActorCriticEncoderAdapt, ProprioAdaptTConv
from .history_tcn import HistoryTCN

__all__ = [
    "ActorCriticEncoder",
    "ActorCriticEncoderConstrained",
    "ActorCriticEncoderAdapt",
    "HistoryTCN",
    "ProprioAdaptTConv",
]
```

- [ ] **Step 2: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/__init__.py
git commit -m "chore(encoder): export HistoryTCN"
```

---

## Chunk 3: Configuration and Registration

### Task 8: Update config.py

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/config.py`

- [ ] **Step 1: Add proprio_history fields to HeroAgentEnvCfg base class**

Find `HeroAgentEnvCfg` class definition and add:

```python
# Proprioception history (for encoder with history augmentation)
# 0 = disabled (default), 30 = standard for history-augmented encoder
proprio_history_len: int = 0
proprio_feature_dim: int = 8
```

- [ ] **Step 2: Enable history in HeroAgentEncoderTrainEnvCfg**

In `HeroAgentEncoderTrainEnvCfg` (line ~392), add:

```python
proprio_history_len: int = 30
```

This activates the history buffer for ALL encoder configs (Encoder-Base, Encoder-Base-Debug, Constrained, Adapt-Base) since they all inherit from this class.

- [ ] **Step 3: Remove duplicate fields from HeroAgentAdaptBaseEnvCfg**

`HeroAgentAdaptBaseEnvCfg` already has `proprio_history_len=30` and `proprio_feature_dim=8`. Since these are now inherited from `HeroAgentEncoderTrainEnvCfg`, remove the duplicates from `HeroAgentAdaptBaseEnvCfg` to avoid confusion:

```python
@configclass
class HeroAgentAdaptBaseEnvCfg(HeroAgentEncoderTrainEnvCfg):
    """Phase 2 adaptation training config.

    Inherits proprio_history_len=30 from HeroAgentEncoderTrainEnvCfg.
    History buffer and "proprio_hist" observation managed by base_env.
    """
    pass  # All fields inherited
```

- [ ] **Step 4: Verify lint**

Run: `ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/config.py`

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/config.py
git commit -m "feat(config): enable proprio history for all encoder env configs"
```

---

### Task 9: Update rsl_rl_ppo_cfg.py

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/agents/rsl_rl_ppo_cfg.py`

- [ ] **Step 1: Define new obs_groups with history key**

After `_PRIVILEGED_OBS_GROUPS` (line ~114), add:

```python
# Observation groups for configs that use proprioception history.
_HISTORY_PRIVILEGED_OBS_GROUPS: dict[str, list[str]] = {
    "policy": ["policy", "privileged", "proprio_hist"],
    "critic": ["policy", "privileged", "proprio_hist"],
}
```

- [ ] **Step 2: Add h_embed_dim to encoder policy configs**

In `_RslRlPpoEncoderBaseCfg` (line ~63), add:

```python
h_embed_dim: int = 32
proprio_history_len: int = 30
proprio_feature_dim: int = 8
```

- [ ] **Step 3: Update HeroAgentEncoderPPORunnerCfg**

Change `obs_groups` to use history version (line ~208):

```python
obs_groups = _HISTORY_PRIVILEGED_OBS_GROUPS
```

- [ ] **Step 4: Update HeroAgentConstrainedEncoderRunnerCfg**

Change `obs_groups` (line ~355):

```python
obs_groups = _HISTORY_PRIVILEGED_OBS_GROUPS
```

- [ ] **Step 5: Update HeroAgentAdaptBaseRunnerCfg**

Change `obs_groups` (line ~253):

```python
obs_groups = _HISTORY_PRIVILEGED_OBS_GROUPS
```

- [ ] **Step 6: Update RslRlPpoActorCriticEncoderAdaptCfg**

Remove `proprio_history_len` and `proprio_feature_dim` from here if they're now in the base config `_RslRlPpoEncoderBaseCfg`.

- [ ] **Step 7: Verify lint**

Run: `ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/agents/rsl_rl_ppo_cfg.py`

- [ ] **Step 8: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/agents/rsl_rl_ppo_cfg.py
git commit -m "feat(cfg): add h_embed_dim, history obs_groups for encoder configs"
```

---

### Task 10: Update runner module namespace registration

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/agents/rsl_rl_ppo_cfg.py`

The RSL-RL runner resolves policy `class_name` dynamically from the runner module namespace. Custom classes are injected at module top.

- [ ] **Step 1: Verify HistoryTCN is importable via encoder module**

HistoryTCN doesn't need runner namespace injection because it's instantiated internally by ActorCriticEncoder, not by the runner. The runner only instantiates the top-level policy class (e.g., `ActorCriticEncoder`).

No changes needed for HistoryTCN. The existing `_runner_module.ActorCriticEncoder = ActorCriticEncoder` injection handles it.

- [ ] **Step 2: No commit needed (no changes)**

---

## Chunk 4: Verification

### Task 11: Full lint check

- [ ] **Step 1: Run ruff check on all modified files**

```bash
cd /workspace/isaaclab
ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/encoder/
ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/base_env.py
ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/adapt_base_env.py
ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/config.py
ruff check source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/agents/rsl_rl_ppo_cfg.py
```

- [ ] **Step 2: Run ruff format**

```bash
ruff format source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/
```

### Task 12: Smoke test -- import and instantiation

- [ ] **Step 1: Verify imports work**

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -c "
from isaaclab_tasks.direct.hero_agent.encoder import HistoryTCN, ActorCriticEncoder
from isaaclab_tasks.direct.hero_agent.encoder import ActorCriticEncoderConstrained, ActorCriticEncoderAdapt
print('All encoder imports OK')
import torch
tcn = HistoryTCN(input_dim=8, hidden_dim=32, output_dim=32, history_len=30)
x = torch.randn(4, 30, 8)
h = tcn(x)
print(f'HistoryTCN: input {x.shape} -> output {h.shape}')
assert h.shape == (4, 32), f'Expected (4, 32), got {h.shape}'
print('HistoryTCN shape test PASSED')
"
```

- [ ] **Step 2: Verify env observations include proprio_hist**

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -c "
import gymnasium as gym
import isaaclab_tasks.direct.hero_agent  # register tasks
env = gym.make('Isaac-HeroAgent-Encoder-Base-v0', num_envs=4, headless=True)
obs, _ = env.reset()
print('Obs keys:', list(obs.keys()) if isinstance(obs, dict) else type(obs))
if 'proprio_hist' in obs:
    print(f'proprio_hist shape: {obs[\"proprio_hist\"].shape}')
else:
    print('WARNING: proprio_hist not in observations')
env.close()
"
```

### Task 13: Short training run

- [ ] **Step 1: Run 50 iterations of constrained encoder training**

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-HeroAgent-Constrained-Encoder-Base-v0 \
    --num_envs 1024 --max_iterations 50 --headless \
    --logger wandb --log_project_name hero_agent_hist_test
```

Expected: No crashes, loss values logged, 6 constraint lambdas reported.

- [ ] **Step 2: Analyze training log**

```bash
python3 ~/.claude/skills/analyzing-training-logs/analyze_training.py --tier 3
```

Verify:
- Reward logged (not NaN)
- All 6 constraint costs logged
- z_range and z_std metrics present
- No encoder gradient anomalies

- [ ] **Step 3: Final commit with all fixes**

```bash
git add -A  # Only if no sensitive files
git commit -m "feat: history-augmented encoder architecture (Phase 1 + Phase 2)"
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| h_embed_dim=32 | Matches TCN hidden_dim. Compact enough to not dominate inputs (13D policy + 13D z), expressive enough for dynamics context. |
| Shared TCN, not per-module TCN | Single TCN trained with all losses produces richer embedding. Reduces parameter count. |
| Critic inherently asymmetric with history | Critic sees [policy, h_embed, privileged] directly. No z in critic path -- encoder gradient comes only from actor loss. Matches NORBC design. |
| History in base_env, gated by config | Simplifies adapt_base_env. All encoder configs benefit. Zero overhead when disabled (proprio_history_len=0). |
| adapt_head (2-layer MLP) instead of full TCN | Phase 2 reuses frozen shared_tcn. adapt_head is 64-param vs ~5K-param ProprioAdaptTConv. Faster convergence. |
| ProprioAdaptTConv kept (not deleted) | Legacy checkpoint compatibility. Can be deprecated later. |
| _encode() accepts full obs dict | Cleaner API: encoder can access all needed keys internally. No separate policy_obs/h_embed/privileged arguments. |

## Risk and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Double _compute_h_embed() per step (encode + actor) | ~2x TCN compute | Acceptable for 32D output. Cache if profiling shows bottleneck. |
| h_embed dim too small/large | Under/over-fitting | 32D is configurable via h_embed_dim. Try 16, 32, 64. |
| Encoder input dim jump (19D -> 64D) | Slower convergence | Same hidden dims [256,128,64] can handle 64D input. Encoder MLP capacity sufficient. |
| Phase 2 adapt_head too simple | Poor z_hat quality | 2-layer MLP with h_embed=32D input. If insufficient, increase to 3-layer or larger hidden. |
| Old checkpoint incompatibility | Load failures | load_state_dict() already handles missing/extra keys. New modules init randomly. |
