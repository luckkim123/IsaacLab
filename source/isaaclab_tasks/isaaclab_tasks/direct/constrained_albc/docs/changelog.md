# Constrained ALBC Changelog

## 2026-03-26: Paper-Aligned Architecture Overhaul (uncommitted)

Major restructuring to align with NORBC paper's Teacher-Student framework.
**670 insertions, 2,902 deletions across 16 files.**

### Network Architecture (3-Layer MLP)
- Encoder hidden: [128, 64] -> [256, 128, 64] (2 -> 3 hidden layers)
- Actor hidden: [128, 64] -> [256, 128, 64]
- Critic hidden: [256, 128] -> [512, 256, 128]
- Value backbone: [256] -> [512, 256, 128] -> 64D features (shared multi-head)
- Files: `encoder/actor_critic_encoder.py`, `encoder/actor_critic_encoder_constrained.py`, `agents/rsl_rl_ppo_cfg.py`

### Observation Redesign (14D + 23D)
- Policy obs o_t (14D): euler(3) + ang_vel(3) + att_err_rp(2) + joint_pos(2) + joint_vel(2) + prev_actions(2)
- Privileged p_t (23D): hydro(6) + inertia(4) + damping(4) + mass(2) + payload(4) + joint(2) + density(1)
- Encoder input: p_t only (was 280D with history + policy_obs + privileged)
- File: `mdp/observations.py`

### Action Space: EE Delta -> Joint PD Targets
- New: `q_des = q_nominal + action_scale * a_t` (5 lines, 1 method)
- Removed: FK + analytical IK + workspace clamp + rate limiting (65 lines, 3 methods)
- `nominal_joint_pos = (0.0, pi)` (EE at body center), `action_scale = pi`
- Removed: `_compute_ee_position()`, `_apply_ee_delta_action()`, `compute_equilibrium_joint_positions()`
- Files: `albc_env.py`, `config.py`, `mdp/events.py`

### Control Frequency: 1:40 Ratio
- Physics PD: 2000Hz (dt=0.0005s)
- Policy: 50Hz (decimation=40, control_decimation=1)
- Was: 200Hz PD (dt=0.005, decimation=1, control_decimation=4)

### Shared Backbone Multi-Head Value Function
- Shared backbone: cat([o_t, p_t]) -> MLP[512,256,128] -> 64D features
- Reward head: Linear(64 -> 1), Cost head: Linear(64 -> K=4)
- Replaces separate critic MLP; parameter grouping updated in `constraint_trpo.py`

### Code Simplification
- Deleted `utils/debug_vis.py` (333 lines)
- Removed proprio history (`_get_proprio_features`, `_update_proprio_hist`, EMA buffers)
- Removed equilibrium joint init mode
- Naming: C-TRPO -> TRPO + IPO (all files)

---

## Legacy History (2026-03-20 to 2026-03-26, 85 commits)

### Phase 1: Extraction and Refactoring (2026-03-20)
Initial extraction from hero_agent as standalone package, followed by aggressive refactoring.

| Commit | Description |
|--------|-------------|
| f24f409 | Extract constrained_albc as standalone package from hero_agent |
| 86f8274 | Flatten agents/ config hierarchy from 3-level to 2-level |
| 912f25d | Clean up utils/ dead code, docstrings, magic numbers |
| 9dc2e04 | Simplify runners/ with property extraction and pattern dedup |
| cd63b8a | Remove 5 unused constraint cost functions from mdp/ |
| bf3ff43 | Remove hardcoded num_constraints/budgets duplicates |
| 1834347 | Simplify constraint_trpo.py internal duplication |
| 40592e3 | Structural cleanup of encoder/ activation, helpers, OCP |
| 4b990ed | Extract helpers from monolithic methods, remove dead code |
| cea93c5 | Flatten 3-level runner hierarchy to single class |
| 9978524 | Simplify constraint_trpo.py surrogates, logging, cost GAE |
| 1dfb2c1 | Remove dead branches from encoder (465->348, 131->110 LOC) |
| 7b65d6e | Simplify constrained ALBC MDP module (rewards, events, constraints, observations) |
| 8615e27 | Merge 4-class config hierarchy into single ConstrainedALBCEnvCfg |
| e2afd27 | Fix config naming, delete doraemon.py (old), remove DR infeasibility logging |
| 72be9ac | Remove enable_payload conditional and state_space guards |
| 762d7c9 | Remove backward compat from encoder load_state_dict |
| cbd2dd2 | Remove kwargs, rnd, and hasattr guards from ConstraintTRPO |
| efb4e3e | Final DORAEMON cleanup + DRSampler.get() signature simplification |

### Phase 2: Bug Fixes and Code Review (2026-03-21 -- 2026-03-22)
Post-refactoring stabilization through code reviews and bug fixes.

| Commit | Description |
|--------|-------------|
| d36dbc4 | Remove unused policy config fields |
| 2b610fa | Format actor_critic_encoder_constrained.py |
| 53b4f70 | Initialize _last_* monitoring attrs + fix docstring |
| f6089fb | Code review fixes + runtime integration bugs |
| 3d70b5d | _prev_joint_pos reset timing + control_dt latent bug |
| 3f48317 | Encoder code review: DRY, no_grad perf, backward compat |
| 7c206f7 | Save encoder optimizer on checkpoint + NaN guard |
| 59e37eb | MDP code review: 3 critical bugs + 5 theoretical fixes |
| 8026720 | Remove redundant PBRS progress reward |
| 8350642 | C-TRPO mode oscillation fix: EMA smoothing + critic LR gating |
| 56b38c5 | 5-7 deg plateau fix: Laplacian reward + noise floor + overshoot relaxation |

### Phase 3: Reward Tuning and Constraint Improvements (2026-03-22 -- 2026-03-24)
Iterative reward engineering and TRPO + IPO algorithm tuning.

| Commit | Description |
|--------|-------------|
| 0ee185d | Add TRPO step quality diagnostic logging |
| ec47ede | Deduplicate WandB metrics + update train-analyze skill |
| 0783337 | Add exponential command kernel + joint torque penalty |
| 706accd | Per-axis sigma for roll/pitch asymmetry in command reward |
| 7028fe4 | Sigma -> direct coefficient form + k_c=100 scaling |
| 4a00c06 | Tune exponential coefficients c=5/7.5, revert k_c=5 |
| 6abe899 | entropy_coef=0, barrier_alpha=0.02 (NORBC paper) |
| d0dd0e1 | Scale max_kl=0.002 for 2D action space (per-dim KL normalization) |
| 4de20a2 | Decouple sigma from TRPO + remove yaw_quad_damp from privileged obs |
| 6935ac7 | Increase std_lr 1e-4 -> 3e-3 for faster sigma equilibrium |
| b260ba5 | encoder_lr 3e-4->1e-3, encoder_epochs 3->5, DORAEMON threshold 10->15 |
| ad5c945 | Add reconstruction auxiliary loss for encoder gradient survival |
| 4e8d218 | Revert reconstruction auxiliary loss (failed experiment) |
| 8db7660 | Encoder dynamic input (280D) matching NORBC architecture |
| f80f293 | Increase TRPO max_kl from 0.002 to 0.005 |
| a328c62 | Encoder KL gating freeze + max_encoder_kl proportional scaling |

### Phase 4: Action Mode Evolution (2026-03-24 -- 2026-03-26)
Progression through action representations, culminating in delta EE before current session's Joint PD switch.

| Commit | Description |
|--------|-------------|
| 961241c | Add EE position action mode with analytical IK |
| 344ed08 | Tune torque_weight from -0.001 to -0.01 |
| 58bf466 | Escalate torque_weight from -0.01 to -0.05 |
| 7239ff6 | Quadratic reward + violation-proportional barrier weights |
| 96540850 | Reduce command_weight from -5.0 to -1.0 |
| 5c2d203 | Revert: restore exponential reward for fix1/fix2 isolation test |
| a1bcb86 | Revert: remove violation-proportional barrier weights |
| c909d21 | EE position rate limiting + reset init + quadratic reward |
| d834779 | Add tanh squashing to actor output + expand workspace radius |
| 9abb758 | Store raw actions to eliminate lossy atanh + barrier_t=100 |
| 7da1146 | Switch to delta EE action mode to fix arm freeze root cause |
| 641db83 | Fix 3 gradient explosion bugs in ConstraintTRPO |
| bcb6e2f | Add gradient decomposition diagnostics to ConstraintTRPO |

### Key Lessons from Legacy Development
1. **Encoder auxiliary loss failed**: Reconstruction loss (ad5c945) did not help encoder learning; reverted (4e8d218)
2. **Barrier parameter sensitivity**: barrier_t=10 caused 80% line search failure; fixed to barrier_t=100 (9abb758)
3. **Action space evolution**: absolute EE -> EE position -> delta EE -> Joint PD targets. Each transition addressed specific failure modes
4. **Reward engineering**: Multiple iterations from simple to exponential to Laplacian kernels. Current: tracking + linear_error + settling + action penalties
5. **TRPO scaling**: max_kl needs per-dimension normalization for low-DOF systems (d0dd0e1)
