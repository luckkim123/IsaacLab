# Student Policy Design (r13_A → Student-TCN, Student-GRU)

**Date:** 2026-04-21
**Teacher baseline:** r13_A (`logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/model_4999.pt`)
**Purpose:** Replace teacher's privileged-information encoder with a history-based encoder so the policy can run on real hardware without access to privileged state (mass, damping, ocean current, payload, etc.).

---

## 1. Background

### Teacher architecture (r13_A)

- **Privileged → latent encoder**: MLP [256, 128, 64] + LayerNorm + softsign
  - Input: 24D privileged observation (hydro 7D + dynamics 5D + payload 4D + actuator 4D + environment 4D)
  - Output: 9D latent `z` ∈ (-1, 1)^9 (softsign bounded)
- **Actor**: MLP [256, 128, 64] + ELU
  - Input: 87D proprioceptive obs (normalized) + 9D raw latent = 96D
  - Output: 8D action (2D arm delta + 6D thruster)
- **Actor input normalizer**: EmpiricalNorm (87D only; latent is not normalized)

### Teacher proprioceptive observation (87D)

- Current state (26D): command(6) + body(9) + arm(5) + thruster(6)
- Temporal history (55D): joint_tracking(12) + body_tracking(27) + action_hist(16)
  - `hist_len=3`, `hist_stride=3`, `step_dt=0.02s` → **span = 180ms**
- Integral error (6D): leaky integral of [roll, pitch, vx, vy, vz, yaw_rate]

### Problem

The teacher's privileged encoder cannot run on a real robot (no onboard way to measure body_mass, added_mass, water_density, ocean_current_velocity, etc.). We need a **student encoder** that infers the same 9D latent from proprioceptive history alone.

---

## 2. Goals

1. **Train two student encoders** (TCN and GRU) that predict the teacher's 9D latent `z` from proprioceptive obs history.
2. **Reuse teacher's frozen actor** — no re-training of the action head; only the encoder is replaced.
3. **Compare TCN vs GRU** on reconstruction loss and on closed-loop control performance (eval_dr_fulldof).
4. **Keep the 87D per-step input identical to teacher** — student operates on exactly the observation the teacher actor consumes.

### Non-goals

- Re-deriving a new actor architecture (teacher actor stays frozen).
- Full sim-to-real deployment in this spec (addressed in a follow-up).
- Constraint satisfaction re-training (constraints were satisfied at teacher training; student inherits via frozen actor).

---

## 3. Architecture

### 3.1 Common components

- **Per-step input**: 87D teacher observation (`compute_policy_obs` output, the same tensor the teacher actor consumes).
- **Latent output dim**: 9D (matches teacher for latent L2 loss).
- **Teacher actor**: Frozen MLP [256, 128, 64] ELU. Loaded from r13_A final checkpoint.
- **Teacher privileged encoder**: Frozen. Used at rollout time to produce ground-truth `l_t = softsign(LayerNorm(encoder_mlp(privileged_24D)))`.
- **Teacher actor obs normalizer**: Frozen. Applied to 87D obs before both teacher and student actor forward passes so the action distribution is comparable.

### 3.2 Student-TCN encoder

History-window convolutional encoder (ProprioAdaptTConv-inspired, re-implemented for our 87D input).

```
input: obs_window ∈ R^(B × H=50 × 87)          # B envs, H=50 steps back (1.0s outer span)
  │
  ├─ per-step channel transform: Linear(87 → 32) + ELU     # mixes 87 raw features → 32 channels
  │
  ├─ transpose → (B × 32 × 50)
  │
  ├─ Conv1d(32,  64, kernel=9, stride=2) + ELU             # 50 → 21
  ├─ Conv1d(64, 128, kernel=5, stride=1) + ELU             # 21 → 17
  ├─ Conv1d(128,128, kernel=5, stride=1) + ELU             # 17 → 13
  │
  ├─ flatten → (B × 128·13 = 1664)
  │
  ├─ Linear(1664 → 128) + ELU
  ├─ LayerNorm(128)                                         # match teacher's pre-softsign norm
  ├─ Linear(128 → 9)
  └─ softsign → l_hat ∈ (-1, 1)^9
```

- Outer span: 50 control steps × 20 ms = **1.0 s** (captures slow hydro dynamics).
- Combined with the 180 ms inner history embedded in each 87D sample, effective observability ~1.18 s.
- Buffer: per-env ring buffer of shape `(num_envs, 50, 87)`, zero-padded on episode reset.

### 3.3 Student-GRU encoder

Recurrent encoder (NORBC-style).

```
input per step: obs_t ∈ R^(B × 87)
hidden state: h_{t-1} ∈ R^(B × 128)        # initialized to 0 at episode reset
  │
  ├─ GRUCell(input=87, hidden=128)   →   h_t
  │                                         (1-layer, hidden=128 — paper analysis recommendation)
  │
  ├─ Linear(128 → 9)
  ├─ LayerNorm(9)
  └─ softsign → l_hat ∈ (-1, 1)^9
```

- Hidden state persists across steps within an episode and resets (to zero) on `dones`.
- No explicit history window — GRU accumulates past via hidden.

### 3.4 Data flow at training time (both variants)

```
env step (4096 envs) ──►  obs_t (87D)            privileged_t (24D)
                            │                       │
                            │                       ▼
                            │                  Teacher encoder (frozen)
                            │                       │
                            │                       ▼
                            │                    l_t (9D)  ◄─── ground truth latent
                            │
                            ├──►  Student encoder (trainable)
                            │          │
                            │          ▼
                            │       l_hat (9D)
                            │
                            ▼                   ┌── teacher_actor(norm(obs_t), l_t)   = a_t     (action target)
                normalize(obs_t) (teacher      ──┤
                  actor_obs_normalizer)          └── teacher_actor(norm(obs_t), l_hat) = a_hat   (student action)
                                                                   ▲
                                                            (frozen weights,
                                                             autograd still flows
                                                             through to encoder)
                                                             
Loss = ‖a_hat - a_t‖² + λ · ‖l_hat - l_t‖²   (λ=1 default)
```

**Key point**: `a_hat` is computed with the frozen teacher actor, using `l_hat` as the latent slot. Actor weights are `requires_grad=False`, but autograd still propagates gradient back into the student encoder. This gives the encoder a *functional* gradient signal ("produce latents that make the actor output the correct action") on top of the *literal* latent-matching signal.

**Environment interaction**: during rollouts the env is stepped using the **teacher's action** `a_t` (not `a_hat`). This keeps the data distribution matched to what r13_A saw. The student is purely evaluated/trained offline-on-the-fly; it does not control the simulator during training.

### 3.5 Loss

$$L_{\text{student}}(\theta_{\text{enc}}) = \|\bar{a}_t - a_t\|_2^2 + \lambda \|\bar{l}_t - l_t\|_2^2$$

- Only `theta_enc` (student encoder) is trainable.
- `λ = 1.0` at start; monitor per-term magnitudes during the first 50 iters and rescale if one term dominates (>10× the other).
- Reduction: `mean` over batch and feature dims.

---

## 4. Data collection

### Approach: online rollout (no offline replay buffer)

- The teacher (encoder + actor) runs in inference mode in the simulator with 4096 parallel envs.
- At each env step, we record `(obs_t, privileged_t, l_t, a_t)` in a rollout buffer.
- After `n_steps` env steps (default 24), we run `n_epochs` (default 5) SGD passes over the buffer computing the loss in minibatches.
- Env steps use the teacher's sampled-mean action (log_std ignored — deterministic mean).

### Domain randomization

- Use **r14's aggressive HardDR** (wider ranges than r13_A trained on: `thrust_coef (0.3,1.5)`, `body_mass (0.5,1.5)`, `added_mass (0.3,1.8)`, `linear_damping (0.2,2.2)`, `quadratic_damping (0.2,2.2)`, `ocean_current (0.0,2.0)`, `action_latency (0,6)`, etc.).
- Rationale: student should be robust to a broader distribution than the teacher explicitly saw. The teacher's latent `l_t` is still meaningful under wider DR because the encoder applies a static normalization (midpoint/range computed at config time).
- DORAEMON is **disabled** during student training (the Beta curriculum is teacher-specific; student sees samples uniformly across the aggressive HardDR range).

### Episode handling

- Episode length 3000 steps unchanged.
- On reset: TCN buffer zero-padded; GRU hidden zeroed.
- Within-episode: buffer/hidden updated every step.

---

## 5. Training hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | Adam | |
| Learning rate | 5e-4 | standard for supervised BC on frozen backbone |
| LR schedule | constant | 1000 iters is short enough |
| Num envs | 4096 | same as r14 |
| Rollout `n_steps` | 24 | ~2400 samples per iter after stride |
| SGD `n_epochs` | 5 | per rollout buffer |
| Minibatch size | 8192 | splits 4096*24 / 5 ≈ 19600 into ~3 minibatches |
| Max iterations | 1000 | ~30 min total on GPU 1 |
| Loss weight λ | 1.0 | monitor and adjust |
| Gradient clip | 1.0 (global norm) | standard |
| Save interval | 100 | final + intermediate for diagnostic |
| Seed | 42 | reproducibility across TCN/GRU comparison |

### Logging (WandB + TB)

- `student/loss_total`, `student/loss_action`, `student/loss_latent`
- `student/l_hat_std`, `student/l_hat_range`, `student/a_hat_mae`
- `student/grad_norm`
- Every 100 iters: histogram of `l_hat - l_t` per dim, action residual per dim.

---

## 6. Evaluation

After each student run finishes:

1. **Standalone eval_dr_fulldof** with student-in-the-loop:
   - Replace env step action from `teacher_actor(o, l_teacher)` to `teacher_actor(o, student_encoder(o_history))`.
   - Run the 4-level DR sweep (none / soft / medium / hard).
   - Output: `enhanced_summary.json` + `summary_*.png` plots.

2. **Latent-match plots**:
   - Scatter `l_t` vs `l_hat` per dimension across 10 evaluation episodes.
   - Time-series overlay of `l_t(t)` and `l_hat(t)` across a representative episode.

3. **Action-match plots**:
   - Time-series overlay of `a_t(t)` and `a_hat(t)` on 2-3 sample envs.

4. **Comparison matrix** (TCN vs GRU vs teacher):
   - Per DR level × per axis (roll/pitch/vx/vy/vz/yaw): `ss_error`, `ss_jitter`, `os_env_mean`, survival_pct.
   - Report regressions >20% vs teacher.

---

## 7. File organization

Create these under `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/student/`:

```
student/
├── __init__.py
├── config.py          # StudentCfg (encoder type, H, hidden, lr, iters, etc.)
├── models.py          # StudentEncoderTCN, StudentEncoderGRU
├── collector.py       # TeacherRolloutCollector
├── runner.py          # StudentRunner (train loop, logging, checkpointing)
└── eval.py            # student-in-the-loop wrapper used by eval_dr_fulldof
```

Training entry point: `scripts/reinforcement_learning/rsl_rl/train_student.py` (new).

Launch script: `scripts/launch_student_tcn.sh`, `scripts/launch_student_gru.sh`.

---

## 8. Testing

- **Unit**: TCN/GRU forward pass with dummy 4096×50×87 / 4096×87 input produces 4096×9 output in (-1, 1); autograd gradient flows to encoder weights.
- **Integration**: One-step rollout + one gradient step reduces loss on a synthetic batch.
- **Regression**: Load r13_A checkpoint, run 1-iter student-in-loop eval, confirm action trajectory matches teacher within 5% MAE (establishes the pipeline works; student starts from untrained encoder so this first iter is just a plumbing test on a frozen loaded student checkpoint from a later iter).

---

## 9. Decisions log (per user discussion)

1. **Two encoder variants (TCN + GRU) run sequentially on GPU 1** — supervised learning is fast (~30 min each), total ~1 h.
2. **Loss: NORBC joint loss** (action + latent L2) with **λ=1.0**.
3. **Actor frozen**: use teacher's verified actor; encoder-only training. Action-loss gradient flows through the frozen actor into the encoder, giving dual supervision.
4. **Per-step input = teacher's 87D obs** (not a stripped-down subset). Real-robot deployment: all 87D components are onboard-computable (body pose from INS, joint state from encoders, thruster feedback, integral/history computed onboard).
5. **Outer history horizon**: TCN H=50 (1 s); GRU streaming. Inner 180 ms history from teacher's obs is preserved in each per-step input.
6. **Hero_agent code is NOT ported**. Fresh implementation under `constrained_full_albc/student/`.
7. **DR during student training = r14 aggressive HardDR** (broader than r13_A saw). DORAEMON disabled — uniform sampling.
8. **Student never controls the sim during training**; teacher drives the env. Student is evaluated closed-loop only after training via `eval_dr_fulldof`.
9. **GRU arch = 1-layer, hidden=128** (paper analysis recommendation; start small, expand if needed).
10. **TCN arch**: 3-layer Conv1d (kernel 9/5/5, stride 2/1/1, channels 32→64→128→128). Matches the HORA family but re-implemented from scratch.

---

## 10. Open risks

- **Loss scale imbalance**: action L2 is much smaller (~0.07) than latent L2 (~1.0). Monitor first 50 iters; if latent dominates, set λ=0.1.
- **GRU warm-up**: hidden=0 at reset means the first ~20 steps of each 3000-step episode have poor `l_hat`. Expected — reflected in the loss average. If problematic, can extend burn-in by discarding the first 50 steps per episode from loss.
- **Privileged encoder normalization**: the teacher encoder uses a static min-max normalization defined at config time (`_enc_obs_midpoint`, `_enc_obs_range`). When DR ranges widen for student training, some privileged inputs will be clipped past the teacher's training range. This is expected and matches r14's setup; `l_t` stays bounded by the softsign.
- **TCN causal vs non-causal**: we use non-causal convolutions since the window is past-only (no look-ahead). No issue.
- **Episode boundaries in the buffer**: the rollout buffer mixes samples from different episodes. Each sample carries its own history window (TCN) or hidden (GRU), so the boundary handling is per-env and doesn't leak across envs within a minibatch.

---

## 11. Success criteria

- Both students converge: `loss_latent < 0.2` and `loss_action < 0.02` by iter 500.
- Closed-loop `eval_dr_fulldof`: student performance within **+20% regression** vs teacher on all hard-DR axes.
- TCN vs GRU: at least one variant matches teacher within +10% on ss_error across DR levels. The other variant is the comparison baseline.

Follow-up (out of scope): larger DR, sim-to-real testing, ablation on H for TCN.
