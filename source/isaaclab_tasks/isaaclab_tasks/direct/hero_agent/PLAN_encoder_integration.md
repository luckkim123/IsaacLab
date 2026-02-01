# Phase 1 Encoder Integration Plan: Extrinsics Encoder for Hero Agent ALBC

## Overview

HORA/RMA-style Phase 1 teacher policy: Add an **Extrinsics Encoder** that compresses 64D raw privileged hydrodynamic parameters into 2D latent `z` via scaled sigmoid, concatenated with the 13D policy observation (total 15D actor input).

## Architecture Summary

```
Environment                              Network (ActorCriticEncoder)

obs["policy"]  (N, 13)  ────────────>  ┌──────────────────────────┐
                                        │  get_actor_obs():        │
obs["privileged"] (N, 64) ───────────> │    priv -> Encoder -> z  │
                                        │    cat([policy, z]) = 15D│
                                        │    -> Actor MLP -> 2D   │
                                        │                          │
                                        │  get_critic_obs():       │
                                        │    cat([policy, z, priv])│
                                        │    = 79D -> Critic MLP   │
                                        └──────────────────────────┘
```

## Files to Modify

| # | File | Change |
|---|------|--------|
| 1 | `isaaclab_tasks/models/hydrodynamics.py` | Add `get_privileged_info()` method |
| 2 | `isaaclab_tasks/direct/hero_agent/mdp/events.py` | Add `randomize_buoy_hydrodynamics()` function |
| 3 | `isaaclab_tasks/direct/hero_agent/hero_agent_env.py` | Return `{"policy": 13D, "privileged": 64D}`, call buoy DR |
| 4 | `isaaclab_tasks/direct/hero_agent/hero_agent_env_cfg.py` | Add `HeroAgentEncoderTrainEnvCfg` |
| 5 | `isaaclab_tasks/direct/hero_agent/agents/rsl_rl_ppo_cfg.py` | Add encoder PPO config with `obs_groups` |
| 6 | `isaaclab_tasks/models/actor_critic_encoder.py` | **NEW**: Custom `ActorCriticEncoder(nn.Module)` |
| 7 | `isaaclab_tasks/direct/hero_agent/__init__.py` | Register new env in Gymnasium |

## Detailed Implementation

### Step 1: `hydrodynamics.py` - Add `get_privileged_info()`

Add method to `HydrodynamicsModel` that extracts all per-env tensors as a flat vector:

- `_added_mass_matrix` diagonal: `(N,6,6)` -> `diagonal()` -> `(N,6)`
- `_linear_damping_diag`: `(N,6)`
- `_quadratic_damping_diag`: `(N,6)`
- `_volume`: `(N,)` -> unsqueeze -> `(N,1)`
- `_vehicle_mass`: `(N,)` -> unsqueeze -> `(N,1)`
- `_r_cb`: `(N,3)`
- `_r_cg`: `(N,3)`
- `_rigid_body_inertia`: `(N,3)`
- `_current_velocity`: `(N,6)` (main body only)

**Main body**: 6+6+6+1+1+3+3+3+6 = **35D**
**Buoy** (no current): 6+6+6+1+1+3+3+3 = **29D**
**Combined**: 35 + 29 = **64D**

Method signature:
```python
def get_privileged_info(self, env_ids=None, include_current=True) -> torch.Tensor:
```

### Step 2: `events.py` - Add buoy randomization

New function `randomize_buoy_hydrodynamics()` that calls `env._buoy_hydro.randomize_parameters()` with same scale ranges as main body (from `DomainRandomizationCfg`).

### Step 3: `hero_agent_env.py` - Two changes

**`_get_observations()`**: Conditionally return privileged info:
```python
observations = {"policy": obs}
if self.cfg.state_space > 0:
    main_priv = self._hydro.get_privileged_info()
    buoy_priv = self._buoy_hydro.get_privileged_info(include_current=False)
    observations["privileged"] = torch.cat([main_priv, buoy_priv], dim=-1)
return observations
```

**`_reset_idx()`**: Add buoy randomization after main body randomization (after line 475).

**Import**: Add `randomize_buoy_hydrodynamics` to the import from `events.py`.

### Step 4: `hero_agent_env_cfg.py` - Encoder training config

```python
@configclass
class HeroAgentEncoderTrainEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent encoder training with privileged info."""
    state_space: int = 64  # signals privileged info mode
```

### Step 5: `rsl_rl_ppo_cfg.py` - Encoder PPO config

New config classes:
- `RslRlPpoActorCriticEncoderCfg`: extends `RslRlPpoActorCriticCfg` with encoder params (`encoder_hidden_dims`, `encoder_latent_dim`, `z_min`, `z_max`, `privileged_dim`, `policy_obs_dim`)
- `HeroAgentEncoderTrainPPORunnerCfg`: sets `obs_groups = {"policy": ["policy", "privileged"], "critic": ["policy", "privileged"]}`

**Class name registration**: RSL-RL resolves `class_name` string via Python's built-in name resolution in `on_policy_runner.py`'s module scope. Our custom class must be injected into that scope. We do this by importing and monkey-patching the runner module:

```python
# At top of rsl_rl_ppo_cfg.py
from isaaclab_tasks.models.actor_critic_encoder import ActorCriticEncoder
import rsl_rl.runners.on_policy_runner as _runner_module
_runner_module.ActorCriticEncoder = ActorCriticEncoder
```

### Step 6: `actor_critic_encoder.py` - Custom Network (NEW FILE)

Key design:

```python
class ActorCriticEncoder(nn.Module):
    """ActorCritic with extrinsics encoder for HORA Phase 1."""
    is_recurrent: bool = False

    def __init__(self, obs, obs_groups, num_actions,
                 policy_obs_dim, privileged_dim,
                 encoder_hidden_dims, encoder_latent_dim, encoder_activation,
                 z_min, z_max,
                 actor_hidden_dims, critic_hidden_dims, activation,
                 actor_obs_normalization, critic_obs_normalization,
                 init_noise_std, noise_std_type="scalar", **kwargs):
```

**Internal flow**:
- `_get_combined_obs(obs)`: concatenates obs_groups["policy"] tensors
- `_split_obs(combined)`: splits into `policy_obs[:, :13]` and `privileged[:, 13:77]`
- `_encode(privileged)`: MLP + scaled sigmoid -> `z` (2D)
- `get_actor_obs(obs)`: returns `cat([policy_obs, z])` (15D)
- `get_critic_obs(obs)`: returns `cat([policy_obs, z, privileged])` (79D)
- `act()`, `act_inference()`, `_update_distribution()`, `evaluate()`: same pattern as base `ActorCritic`
- Implements all required interfaces: `reset()`, `action_mean`, `action_std`, `entropy`, etc.

**Scaled sigmoid**: `z = z_min + (z_max - z_min) * sigmoid(raw_encoder_output)`
- Guarantees `z in [0.1, 5.0]`, bounded, differentiable
- Compatible with TDC's `M_hat = diag(z)` requirement

**Gradient flow**: Encoder is part of `nn.Module`. During PPO update, stored obs replayed through the full network -> encoder gradients flow via actor/critic loss backpropagation.

### Step 7: Gymnasium Registration

In `__init__.py`, register:
```python
gym.register(
    id="Isaac-HeroAgent-ALBC-Encoder-Train-v0",
    entry_point="...:HeroAgentEnv",
    kwargs={
        "env_cfg_entry_point": "...:HeroAgentEncoderTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "...:HeroAgentEncoderTrainPPORunnerCfg",
    },
)
```

## Key Design Decisions

1. **Encoder inside network, not env**: PPO rollout uses inference mode (no gradients). Encoder must be inside the network class so that during the PPO update phase (gradients enabled), encoder parameters get optimized through actor/critic losses.

2. **obs_groups routing**: Both actor and critic receive `["policy", "privileged"]`. The `ActorCriticEncoder` internally splits by known dimensions and routes differently:
   - Actor: `cat([policy, z])` = 15D (compressed)
   - Critic: `cat([policy, z, privileged])` = 79D (full information, asymmetric)

3. **Scaled sigmoid**: `z = 0.1 + 4.9 * sigmoid(x)` guarantees z > 0, bounded, differentiable.

4. **Monkey-patch for class resolution**: Register `ActorCriticEncoder` into the RSL-RL runner module namespace to work with the runner's string-based class resolution pattern.

5. **64D privileged info**: Main body 35D (with current) + Buoy 29D (no current) based on actual tensor shapes in HydrodynamicsModel.

## Verification Plan

1. **Dimension check**: Verify `get_privileged_info()` returns correct tensor shapes (35D main, 29D buoy)
2. **Encoder forward pass**: Test `ActorCriticEncoder` with dummy TensorDict matching expected shapes
3. **Training smoke test**: Run ~10 iterations with `Isaac-HeroAgent-ALBC-Encoder-Train-v0` to confirm no crashes
4. **Gradient flow**: Confirm `encoder.parameters()` have non-zero gradients after PPO update
5. **Buoy DR**: Verify buoy parameters change between resets with randomization enabled
