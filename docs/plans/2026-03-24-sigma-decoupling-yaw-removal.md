# Sigma-KL Decoupling + Yaw Quad Damp Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple log_std from TRPO natural gradient (separate Adam optimizer) and remove non-actionable yaw_quad_damp from privileged observations to fix encoder capacity waste.

**Architecture:** Two independent changes applied together. (1a) log_std is moved from `_policy_params` to a dedicated `_std_params` group with its own Adam optimizer, so TRPO's KL budget is used exclusively for mean updates while sigma follows the natural score-function equilibrium. (1b) yaw_quad_damp (privileged obs index 26) is removed, reducing privileged dim from 28D to 27D and freeing encoder capacity for actionable parameters.

**Tech Stack:** PyTorch, RSL-RL, Isaac Lab configclass

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `algorithms/constraint_trpo.py` | Modify | Add `std_lr` param, split log_std into `_std_params`, create `std_optimizer`, add sigma Adam step after TRPO |
| `agents/rsl_rl_ppo_cfg.py` | Modify | Add `std_lr` config field |
| `runners/constraint_encoder_runner.py` | Modify | Save/load `std_optimizer.pt` |
| `mdp/observations.py` | Modify | Remove yaw_quad_damp from privileged obs |
| `encoder/actor_critic_encoder.py` | Modify | Update fixed normalizer from 28D to 27D |
| `config.py` | Modify | `state_space` 28 -> 27 |

All paths relative to: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/`

---

## Task 1: Add `std_lr` config field

**Files:**
- Modify: `agents/rsl_rl_ppo_cfg.py:138-148`

- [ ] **Step 1: Add config field**

In `RslRlConstraintTRPOAlgorithmCfg`, add `std_lr` after `min_std`:

```python
    min_std: float = 0.2
    """Minimum action standard deviation. Clamped after TRPO step (outside
    trust region optimization). Prevents exploration collapse without consuming
    KL budget."""

    std_lr: float = 1e-4
    """Learning rate for the separate log_std Adam optimizer. log_std is decoupled
    from TRPO natural gradient so that sigma follows the score-function equilibrium
    (dlogpi/dsigma = ((a-mu)^2 - sigma^2) / sigma^3) without consuming KL budget.
    Conservative value (1/3 of encoder_lr) to prevent sigma oscillation."""
```

- [ ] **Step 2: Verify config loads**

```bash
cd /workspace/isaaclab
python -c "from isaaclab_tasks.direct.constrained_albc.agents.rsl_rl_ppo_cfg import RslRlConstraintTRPOAlgorithmCfg; c = RslRlConstraintTRPOAlgorithmCfg(); print(f'std_lr={c.std_lr}')"
```

Expected: `std_lr=0.0001`

- [ ] **Step 3: Commit**

---

## Task 2: Decouple log_std from TRPO in constraint_trpo.py

**Files:**
- Modify: `algorithms/constraint_trpo.py:78-88` (init signature)
- Modify: `algorithms/constraint_trpo.py:156-190` (param grouping)
- Modify: `algorithms/constraint_trpo.py:557-562` (sigma update section)

- [ ] **Step 1: Add `std_lr` to __init__ signature**

At line 80 (after `min_std`), add:

```python
        min_std: float = 0.2,
        std_lr: float = 1e-4,
```

Store it at line 119 (after `self.min_std = min_std`):

```python
        self.min_std = min_std
        self.std_lr = std_lr
```

- [ ] **Step 2: Split log_std out of _policy_params**

Replace the param grouping block (lines 156-190) with:

```python
        # Separate parameter groups:
        # - Actor params: TRPO natural gradient (no optimizer)
        # - Std params: separate Adam (sigma follows score-function equilibrium)
        # - Encoder params: separate Adam (indirect distribution influence)
        # - Value params (critic + cost_critic): Adam optimizer
        value_params = []
        encoder_params = []
        std_params = []
        self._policy_params = []  # Actor MLP weights only (TRPO)

        encoder_prefixes = ("encoder",)
        for name, param in self.policy.named_parameters():
            is_value = name.startswith("critic") or name.startswith("cost_critic")
            is_encoder = any(name.startswith(p) for p in encoder_prefixes)
            is_std = name == "log_std"
            if is_value:
                value_params.append(param)
            elif is_encoder:
                encoder_params.append(param)
            elif is_std:
                std_params.append(param)
            else:
                self._policy_params.append(param)

        self._value_params = value_params
        self.value_optimizer = optim.Adam(value_params, lr=value_lr)
        self._has_encoder_params = len(encoder_params) > 0
        self.encoder_lr = encoder_lr
        if self._has_encoder_params:
            self._encoder_params = encoder_params
            self.encoder_optimizer = optim.Adam(encoder_params, lr=encoder_lr, weight_decay=1e-5)
        else:
            self._encoder_params = []
            self.encoder_optimizer = None

        # Separate optimizer for log_std: decoupled from TRPO KL budget.
        # Sigma update follows score-function gradient dlogpi/dsigma without
        # competing with mean for KL trust region capacity.
        self._std_params = std_params
        if std_params:
            self.std_optimizer = optim.Adam(std_params, lr=std_lr)
        else:
            self.std_optimizer = None

        logger.info(
            "ConstraintTRPO: %d actor params (TRPO), %d std params (Adam lr=%.0e), "
            "%d encoder params (Adam), %d value params (Adam)",
            len(self._policy_params),
            len(std_params),
            std_lr,
            len(encoder_params),
            len(value_params),
        )
```

- [ ] **Step 3: Add sigma Adam step after TRPO**

Replace the noise floor section (lines 557-562):

**Before:**
```python
        ls_success = self._trpo_step(obs_flat, old_mu_flat, old_sigma_flat, surrogate)

        # Noise floor: applied after TRPO step (outside trust region optimization).
        min_log_std = math.log(self.min_std)
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)
```

**After:**
```python
        ls_success = self._trpo_step(obs_flat, old_mu_flat, old_sigma_flat, surrogate)

        # ------------------------------------------------------------------
        # 2b. Sigma update (separate Adam, decoupled from TRPO KL budget)
        # ------------------------------------------------------------------
        # Sigma follows the score-function equilibrium: the gradient
        # dlogpi/dsigma = ((a-mu)^2 - sigma^2) / sigma^3 is self-correcting.
        # When advantage-weighted action spread > sigma^2, gradient pushes sigma up;
        # when < sigma^2, pushes down. Decoupling from TRPO lets this mechanism
        # find the natural equilibrium without KL budget competition.
        min_log_std = math.log(self.min_std)
        if self.std_optimizer is not None:
            # Compute gradient of same surrogate w.r.t. log_std only
            std_loss = surrogate()
            self.std_optimizer.zero_grad()
            std_loss.backward()
            self.std_optimizer.step()

        # Noise floor: hard clamp applied after both TRPO and Adam steps.
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)
```

- [ ] **Step 4: Verify import / instantiation**

```bash
cd /workspace/isaaclab
python -c "
from isaaclab_tasks.direct.constrained_albc.algorithms.constraint_trpo import ConstraintTRPO
print('ConstraintTRPO imported successfully')
"
```

Expected: No errors.

- [ ] **Step 5: Commit**

---

## Task 3: Save/load std_optimizer state in runner

**Files:**
- Modify: `runners/constraint_encoder_runner.py:146-178`

- [ ] **Step 1: Add save logic**

In `save()`, after the encoder_optimizer save block (line 152), add:

```python
        # Save std optimizer state for seamless resume
        if getattr(self.alg, "std_optimizer", None) is not None:
            self._save_aux_state(path, "std_optimizer.pt", self.alg.std_optimizer.state_dict())
```

- [ ] **Step 2: Add load logic**

In `load()`, after the encoder_optimizer restore block (line 168), add:

```python
        # Restore std optimizer state for seamless resume
        if load_optimizer and getattr(self.alg, "std_optimizer", None) is not None:
            std_opt_state = self._load_aux_state(path, "std_optimizer.pt", self.device)
            if std_opt_state is not None:
                self.alg.std_optimizer.load_state_dict(std_opt_state)
                logger.info("Restored std optimizer state from checkpoint")
```

- [ ] **Step 3: Commit**

---

## Task 4: Remove yaw_quad_damp from privileged observations

**Files:**
- Modify: `mdp/observations.py:170-173`

- [ ] **Step 1: Remove yaw_quad_damp lines**

Delete these lines from `compute_privileged_obs()`:

```python
    # Yaw quadratic damping (1D): independently DR'd from roll/pitch.
    # Relevant for yaw_velocity constraint (ALBC cannot generate yaw torque).
    priv_obs.append(env._hydro.quadratic_damping[:, 5:6])  # 1D: yaw axis
```

The function docstring's total should also be updated from "Total: 28D" to "Total: 27D", and remove the `- Yaw quadratic damping (1D)` line from the docstring.

- [ ] **Step 2: Commit**

---

## Task 5: Update encoder fixed normalizer from 28D to 27D

**Files:**
- Modify: `encoder/actor_critic_encoder.py:186-289`

- [ ] **Step 1: Update `_build_fixed_encoder_normalizer`**

In the `mean` tensor, remove index [26] (yaw quad_damp):

```python
            # REMOVED: 1.0,  # [26] yaw quad_damp: 1.0 * mean(U(0.5,1.5)) = 1.0
```

In the `std` tensor, remove index [26]:

```python
            # REMOVED: 1.0 * 1.0 / s12,  # [26] yaw quad_damp (scale range 1.0)
```

Update the comment indices for the remaining `[27] water density` → `[26] water density`.

Update the dim guard:

```python
        if dim != 27:
            logger.warning(
                "Fixed encoder normalizer expects 27D privileged obs, got %d. Falling back to EmpiricalNormalization.",
                dim,
            )
            return EmpiricalNormalization(dim)

        normalizer = _FixedNormalization(mean, std)
        logger.info("Encoder using fixed normalization (27D, analytical DR stats)")
```

Update the method docstring from "28D" to "27D".

- [ ] **Step 2: Commit**

---

## Task 6: Update config dimensions (28 -> 27)

**Files:**
- Modify: `config.py:~163` (`state_space`)
- Modify: `agents/rsl_rl_ppo_cfg.py:63` (`privileged_dim`)

- [ ] **Step 1: Update state_space**

In `ALBCEnvCfg` (config.py):

```python
    state_space: int = 27
```

- [ ] **Step 2: Update privileged_dim**

In `_EncoderPolicyCfg` (agents/rsl_rl_ppo_cfg.py):

```python
    privileged_dim: int = 27
```

- [ ] **Step 3: Commit**

---

## Task 7: Smoke test full training pipeline

- [ ] **Step 1: Dry run (10 iter, 64 envs)**

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Constrained-ALBC-Encoder-v0 \
    --num_envs 64 --max_iterations 10 --headless \
    --logger wandb --log_project_name hero_agent
```

Verify:
- No crashes
- Log shows "ConstraintTRPO: N actor params (TRPO), 2 std params (Adam lr=1e-04), ..."
- Log shows "Encoder using fixed normalization (27D, ...)"
- `Policy/mean_noise_std` metric appears in TB/WandB
- noise_std does NOT immediately collapse (may stay near init=1.0 for 10 iter)

- [ ] **Step 2: Check param counts**

After run, verify log output:
- `std params` count = 2 (one per action dimension)
- `actor params (TRPO)` count decreased by 2 compared to previous runs

- [ ] **Step 3: Commit all changes with summary**

---

## Task 8: Ruff check + format

- [ ] **Step 1: Run linting**

```bash
cd /workspace/isaaclab
ruff check source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/ --fix
ruff format source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/
```

- [ ] **Step 2: Fix any issues and commit**

---

## Notes

- **Checkpoint compatibility**: Both changes break checkpoint compatibility. Fresh training start required. `_handle_dim_mismatch` auto-reinitializes encoder on 27D load.
- **yaw_vel constraint**: Unaffected. Cost function reads angular velocity directly from simulation state, independent of privileged obs.
- **DORAEMON**: No dimension indexing in doraemon.py. Operates at physics level only. Safe.
- **max_std**: Intentionally omitted. Current problem is floor-stuck sigma, not runaway. Add later if needed.
- **Monitoring**: Parent runner already logs `Policy/mean_noise_std`. Watch for sigma recovery from 0.2 floor in first 100 iterations as the key success metric.
