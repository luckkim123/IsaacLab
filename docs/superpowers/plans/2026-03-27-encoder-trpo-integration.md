# Encoder TRPO Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move encoder parameters into the TRPO trust region so that CG + line search jointly optimizes actor and encoder, and the KL constraint covers both.

**Architecture:** Remove the separate Adam-based encoder update (`_update_encoder`). Include encoder params in `_policy_params` so TRPO's natural gradient, line search, and KL constraint apply to the combined actor+encoder parameter set. Line search now verifies barrier feasibility for joint actor+encoder updates.

**Tech Stack:** PyTorch, RSL-RL, TensorDict

---

### Task 1: Integrate encoder params into TRPO policy params

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/algorithms/constraint_trpo.py:129-164`

- [ ] **Step 1: Modify parameter grouping in `__init__`**

Replace the 3-way split (actor/encoder/value) with 2-way split (policy/value). Encoder params join `_policy_params` for TRPO. Track encoder param offset for gradient logging.

```python
# --- Parameter groups ---
# Policy (actor + encoder + log_std): TRPO natural gradient (no optimizer)
# Value (shared backbone + reward/cost heads): Adam
value_prefixes = ("critic", "cost_critic", "value_backbone", "reward_head", "cost_head")
value_params = []
self._policy_params = []
self._encoder_param_offset = 0  # flat index where encoder params start
self._encoder_param_count = 0

actor_numel = 0
for name, param in self.policy.named_parameters():
    if any(name.startswith(p) for p in value_prefixes):
        value_params.append(param)
    elif name.startswith("encoder"):
        # Encoder goes into TRPO policy params
        self._policy_params.append(param)
        self._encoder_param_count += param.numel()
    else:
        # actor + log_std
        self._policy_params.append(param)
        if not name.startswith("encoder"):
            actor_numel += param.numel()

# Encoder offset = total - encoder count (encoder appended after actor)
# Actually compute properly by iterating
offset = 0
enc_start = None
for name, param in self.policy.named_parameters():
    if any(name.startswith(p) for p in value_prefixes):
        continue
    if name.startswith("encoder") and enc_start is None:
        enc_start = offset
    offset += param.numel()
self._encoder_param_offset = enc_start if enc_start is not None else offset

self._value_params = value_params
self.value_optimizer = optim.Adam(value_params, lr=value_lr)

logger.info(
    "ConstraintTRPO: %d policy params (TRPO, incl encoder), %d value params (Adam), encoder offset=%d count=%d",
    sum(p.numel() for p in self._policy_params),
    sum(p.numel() for p in value_params),
    self._encoder_param_offset,
    self._encoder_param_count,
)
```

- [ ] **Step 2: Remove encoder optimizer and related init fields**

Remove `encoder_optimizer`, `_encoder_params`, `_has_encoder_params`, `num_encoder_epochs` storage.
Remove `encoder_lr` and `num_encoder_epochs` from `__init__` signature (accept via `**_kwargs` for backward compat).
Remove `_last_pre_encoder_kl` monitoring field.
Add `_last_encoder_grad_norm` monitoring field.

- [ ] **Step 3: Verify parameter ordering**

Run: `python3 -c "...print named_parameters order..."` to confirm encoder params appear contiguously in the iteration and the offset is correct.

---

### Task 2: Remove separate encoder update from `update()`

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/algorithms/constraint_trpo.py:366-445`

- [ ] **Step 1: Remove encoder update section (lines 420-429)**

Delete the entire encoder update block:
```python
# DELETE: lines 420-429
# --- 3. Encoder update (always, decoupled from line search) ---
# with torch.no_grad():
#     pre_encoder_kl = ...
# self._last_pre_encoder_kl = pre_encoder_kl
# if self.encoder_optimizer is not None:
#     ...
#     self._update_encoder(...)
```

The KL measurement at line 431-432 stays (now it reflects TRPO-only KL, which should be within budget).

- [ ] **Step 2: Delete `_update_encoder()` method entirely (lines 493-514)**

---

### Task 3: Store encoder gradient norm from TRPO step

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/algorithms/constraint_trpo.py:451-491`

- [ ] **Step 1: Extract encoder grad norm from flat gradient in `_trpo_step`**

After computing `g` at line 461, extract the encoder portion:

```python
g = self._flat_grad(loss, self._policy_params)

# Store encoder gradient norm for logging
if self._encoder_param_count > 0:
    enc_g = g[self._encoder_param_offset : self._encoder_param_offset + self._encoder_param_count]
    self._last_encoder_grad_norm = enc_g.norm().item()
```

This must be done BEFORE the gradient clipping at lines 463-466.

---

### Task 4: Update logging

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/utils/logging.py:166-170`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/runners/constraint_encoder_runner.py:211`

- [ ] **Step 1: Update `log_encoder_metrics` to accept algorithm reference**

Change the grad_norm section to read from algorithm's stored value instead of `.grad` attributes:

```python
# Gradient norm: read from algorithm's stored TRPO gradient (no .grad available after autograd.grad)
# Fallback: if alg not provided, skip grad_norm
```

Add `alg` parameter to `log_encoder_metrics()`. If `alg` has `_last_encoder_grad_norm`, use it.

- [ ] **Step 2: Remove `pre_encoder_kl` logging from runner**

In `_log_constraint_metrics`: remove line `metrics["Policy/pre_encoder_kl"] = alg._last_pre_encoder_kl`.

- [ ] **Step 3: Pass algorithm to `log_encoder_metrics` in runner**

Update the call in `log()` to pass `alg=self.alg`.

---

### Task 5: Update runner checkpoint save/load

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/runners/constraint_encoder_runner.py:146-178`

- [ ] **Step 1: Remove encoder optimizer save/load**

In `save()`: remove the `encoder_optimizer.pt` save block.
In `load()`: remove the `encoder_optimizer.pt` load block.

---

### Task 6: Update config

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/agents/rsl_rl_ppo_cfg.py:117-119`

- [ ] **Step 1: Remove encoder-specific config fields**

Remove `num_encoder_epochs` and `encoder_lr` from `RslRlConstraintTrpoCfg`.

---

### Task 7: Smoke test

- [ ] **Step 1: Verify import and instantiation**

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -c "
from isaaclab_tasks.direct.constrained_albc.algorithms.constraint_trpo import ConstraintTRPO
print('Import OK')
"
```

- [ ] **Step 2: Run short training (10 iterations)**

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Constrained-ALBC-Encoder-v0 \
    --num_envs 64 --max_iterations 10 --headless
```

Verify: no crashes, KL within budget, encoder grad_norm logged.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: integrate encoder into TRPO trust region (joint natural gradient + line search)"
```
