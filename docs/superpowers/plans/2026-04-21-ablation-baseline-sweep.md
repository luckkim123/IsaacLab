# Ablation & Baseline Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate 5 policy variants (main, noenc, nocstr, ppoenc, pureppo) under a controlled env config so claims about encoder and IPO can be supported by evidence.

**Architecture:** Sequential GPU 0 training (single seed, 5000 iter each). Variable control enforced by locking the canonical `ALBCEnvCfg` to the best hist-series baseline's env config before Phase 1. Two new task variants (`TRPO-NoIPO`, `PPO-Enc`) are added on top of the two already registered (`NoEncoder`, `PPO`). Each variant gets a 500-iter sanity run (Phase 0.5) before the full 5000-iter run. Each training ends with `eval_dr_fulldof.py` + `eval_dr_switching.py`. Cross-run comparison uses `analyze_eval_dr.py` at the end.

**Tech Stack:** Isaac Lab 5.1 + rsl_rl, ConstraintTRPO + IPO + Asymmetric Encoder, DORAEMON DR curriculum, python `./isaaclab.sh -p` invocation.

**Operational constraints (critical):**
- No editable install swaps (`./isaaclab.sh --install`) while the user's student-policy training or any other training is running. All ablation work runs from `/workspace/isaaclab` main repo only.
- GPU 0 only. GPU 1 is reserved for the user's parallel experiments. User will notify when GPU 1 is freed.
- Baseline (#1): r13_A confirmed as initial pick from user's 4-way eval_dr + switching analysis (2026-04-21). Subject to replacement by Phase 0.6 challenger (hist5_act3 + latent=16) if challenger wins.

---

## File Structure

**Files to create:**
- `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config_noconstraint.py` — `ALBCNoConstraintEnvCfg` subclass with empty constraint list for variant #3.
- `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/ablation_cfgs.py` — New runner/policy cfgs: `FullDOFTRPONoIPORunnerCfg` (#3), `FullDOFPPOEncRunnerCfg` (#4). Keeps existing `rsl_rl_ppo_cfg.py` untouched.

**Files to modify:**
- `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/__init__.py` — Register `Isaac-FullDOF-TRPO-NoIPO-v0`, `Isaac-FullDOF-PPO-Enc-v0`.
- `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py` — If selected baseline has different `hist_len`/`hist_stride`/`hist_action_len`, update `ALBCEnvCfg` to match.
- `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/__init__.py` — Export new runner cfg classes.
- `/workspace/isaaclab/changelog.md` — Record ablation sweep design, baseline selection result, per-phase outcomes.

**Files to evaluate (no modification):**
- `/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py` — Launch training per phase.
- `/workspace/isaaclab/scripts/analysis/eval_dr_fulldof.py` — Steady-state eval.
- `/workspace/isaaclab/scripts/analysis/eval_dr_switching.py` — Mid-episode DR-switch eval.
- `/workspace/isaaclab/scripts/analysis/analyze_eval_dr.py` — Multi-run heavy-tail / sample-divergence / axis-decorrelation report.

No pytest suite is added. Verification is via smoke-training runs (5 iter) and eval scripts.

---

## Phase 0: Baseline (#1) Initial Selection (COMPLETED)

### Operational prohibition

**Never run `./isaaclab.sh --install` while the user's student-policy training (or any other training) is active.** Worktree install swaps corrupt imports for running processes. All ablation work uses the main `/workspace/isaaclab` repo's current install.

### Task 0.1: Record baseline initial selection

**Files:**
- Create: `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/ablation_sweep/baseline_selection.md`

Baseline = **r13_A** selected from user's 4-way eval_dr + eval_dr_switching analysis (2026-04-21).

- [ ] **Step 1: Create `baseline_selection.md` with the user's analysis**

```markdown
# Baseline (#1) Initial Selection

Date: 2026-04-21

## Candidates (4-way)

| Name | obs_dim | hist_len | hist_action_len | encoder_latent | Source |
|---|---|---|---|---|---|
| r13_A | 87 | 3 | 2 | 9 | /workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A |
| hist5 | 113 | 5 | 2 | 9 | /workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-21_04-24-32_r13a_hist5 |
| hist10 | 178 | 10 | 2 | 9 | /workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-21_07-53-11_r13a_hist10 |
| hist5_act3 | 121 | 5 | 3 | 9 | /workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-21_15-13-15_r13a_hist5_act3 |

(layernorm excluded: reward -33%, pos_drift +3278%)

## Hard-DR eval_dr summary (user analysis)

| Run | roll | pitch | vx | vy | vz | yaw |
|---|---|---|---|---|---|---|
| r13_A | 1.08° | 0.28° ✓ | 0.004 ✓ | 0.006 | 0.018 | 0.002 ✓ |
| hist5 | 1.25° | 0.31° | 0.008 | 0.008 | 0.015 | 0.003 |
| hist10 | 1.04° ✓ | 0.35° | 0.008 | 0.009 | 0.007 ✓✓ | 0.004 |
| hist5_act3 | 1.09° | 0.38° | 0.007 | 0.006 ✓ | 0.015 | 0.003 |

## Switching seg1-9 (hard-DR settled response)

| Run | peak_roll | ss_roll | pos_drift |
|---|---|---|---|
| r13_A | 7.63° ✓ | 0.67° | 0.085 |
| hist5 | 8.70° | 1.28° ✗ | 0.121 |
| hist10 | 8.60° | 0.68° | 0.070 ✓ |
| hist5_act3 | 7.72° | 0.59° ✓ | 0.087 |

## Decision

Selected initial baseline: **r13_A**

Rationale:
- Balanced across axes: pitch/vx/yaw/switching peak all 1st or tied-1st.
- Hist-length expansion did not yield decisive improvements — hist10 gains on vz only (CV 231%→69%); other axes tied or slightly worse.
- hist5_act3 best on settling ss_roll (0.59°) but regressed on vx CV (292%).
- All 4 hist variants used encoder_latent_dim=9. User's hypothesis: latent=9 bottleneck prevents longer history from propagating useful info; asymmetric critic leakage further suppresses hist marginal value.

## Subject to Phase 0.6 challenger

A `hist5_act3 + encoder_latent_dim=16` run will test the bottleneck hypothesis.
If challenger wins, baseline switches and canonical env cfg updates accordingly.
```

- [ ] **Step 2: Commit baseline selection record**

```bash
cd /workspace/isaaclab
git add logs/rsl_rl/fulldof_albc/ablation_sweep/baseline_selection.md
git commit -m "ablation: record r13_A as initial baseline from 4-way analysis"
```

---

## Phase 0.6: Baseline Challenger — hist5_act3 + `encoder_latent_dim=16`

### Rationale

User's analysis identified `encoder_latent_dim=9` as a candidate bottleneck. Testing the hypothesis requires one run with the best hist configuration (hist5_act3: best settling ss_roll, best vy nominal) but with 78% more encoder latent capacity (`16` vs `9`).

Three interpretations, each falsifiable:
- Bottleneck real: challenger wins clearly → latent=16 becomes canonical, baseline switches.
- Bottleneck exists but critic leakage dominates: challenger ties or loses marginally → keep r13_A but flag for future symmetric-critic ablation.
- Markovian dominance: challenger loses clearly → keep r13_A; hist extensions don't help under this architecture.

### Task 0.6.1: Pre-flight check — no other training active

**Why**: Editing main's `ALBCEnvCfg` while another training constructs env instances can cause import inconsistencies.

- [ ] **Step 1: Check GPU activity**

```bash
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
ps aux | grep "train.py" | grep -v grep
```

- [ ] **Step 2: Ask user to confirm no active training on main repo**

Ask: "hist5_act3+latent=16 challenger는 main config.py를 수정해야 합니다. 현재 student policy나 다른 main-repo 기반 훈련이 active 상태인지 확인해주세요."

Wait for user confirmation. If active, defer Phase 0.6 until safe window.

### Task 0.6.2: Edit main ALBCEnvCfg to hist5_act3 + latent=16

**Files:**
- Modify: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py`
- Modify: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py`

- [ ] **Step 1: Read current values**

```bash
grep -n "hist_len\|hist_action_len\|hist_stride\|observation_space\|policy_obs_dim\|encoder_latent_dim" \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py \
  | grep -v "^\s*#"
```

- [ ] **Step 2: Read hist5_act3 env.yaml for exact values**

```bash
grep -E "^(hist_len|hist_stride|hist_action_len|observation_space):" \
  /workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-21_15-13-15_r13a_hist5_act3/params/env.yaml
```

Expected values: `hist_len=5, hist_stride=3, hist_action_len=3, observation_space=121`.

- [ ] **Step 3: Edit config.py to hist5_act3 values**

Use Edit tool to change in `config.py`:
- `observation_space` → `121`
- `hist_len` → `5`
- `hist_action_len` → `3`
- (hist_stride should already be 3; verify)

- [ ] **Step 4: Edit rsl_rl_ppo_cfg.py**

Use Edit tool:
- `_EncoderPolicyCfg.policy_obs_dim: int = 87` → `121`
- `_EncoderPolicyCfg.encoder_latent_dim: int = 16` (confirm already 16; if not, change)
- `_FullDOFNoEncoderPolicyCfg.policy_obs_dim: int = 87` → `121`

- [ ] **Step 5: Do NOT run `./isaaclab.sh --install`**

The editable install maps `isaaclab_tasks` to main's source tree. In-place edits take effect on next Python import in new processes.

- [ ] **Step 6: Smoke-test 5-iter**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-TRPO-v0 \
  --num_envs 64 --max_iterations 5 --headless
```

Expected: 5 iters pass without shape mismatch. If fail, inspect error and fix config values.

- [ ] **Step 7: Commit the challenger env config (do not push)**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py
git commit -m "ablation: reconfigure env to hist5_act3 + encoder_latent_dim=16 for Phase 0.6 challenger"
```

### Task 0.6.3: Launch challenger training (5000 iter)

**Files:**
- Log: `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/<ts>_challenger_hist5_act3_enc16/`

- [ ] **Step 1: Start training, GPU 0, single seed 30**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-TRPO-v0 \
  --num_envs 2048 --max_iterations 5000 --headless \
  --logger wandb --log_project_name fulldof_albc --run_name challenger_hist5_act3_enc16 \
  2>&1 | tee /workspace/challenger_hist5_act3_enc16.log
```

Expected runtime: ~5 hr on RTX 4070.

- [ ] **Step 2: Periodic monitor (every 500 iter) with train-analyze skill**

Check at iter 500, 1500, 3000, 4500. Halt only if reward diverges (negative sustained >500 iter) or line_search_success < 0.3 sustained.

- [ ] **Step 3: Verify `model_4999.pt` exists**

```bash
LATEST=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_challenger_hist5_act3_enc16 | head -1)
ls "$LATEST/model_4999.pt"
```

### Task 0.6.4: Run eval_dr + eval_dr_switching for challenger

- [ ] **Step 1: Run eval_dr_fulldof**

```bash
cd /workspace/isaaclab
LATEST=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_challenger_hist5_act3_enc16 | head -1)
CKPT="$LATEST/model_4999.pt"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_fulldof.py \
  --task Isaac-FullDOF-TRPO-v0 --num_envs 64 --headless --checkpoint "$CKPT"
```

Expected: `<run>/eval_dr/enhanced_summary.json` + 3 PNG plots + 4 npz files.

- [ ] **Step 2: Run eval_dr_switching**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_switching.py \
  --task Isaac-FullDOF-TRPO-v0 --num_envs 64 --headless \
  --checkpoint "$CKPT" --seed 42 --segment_duration 5.0 --num_segments 10
```

- [ ] **Step 3: Inspect plots via Read tool**

Read `summary_att.png, summary_lin_vel.png, summary_yaw.png` from `<run>/eval_dr/`.

---

## Phase 0.7: Final Baseline Decision + Canonical Env Cfg Lock

### Task 0.7.1: Build head-to-head comparison table

**Files:**
- Append to: `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/ablation_sweep/baseline_selection.md`

- [ ] **Step 1: Run 2-way analyze_eval_dr**

```bash
R13A=/workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/eval_dr
CHAL=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_challenger_hist5_act3_enc16 | head -1)/eval_dr
python3 /workspace/isaaclab/scripts/analysis/analyze_eval_dr.py \
  "$R13A" "$CHAL" \
  --labels r13_A challenger_enc16 \
  --levels none soft medium hard
```

- [ ] **Step 2: Print summary table**

```bash
python3 -c "
import json
paths = {
    'r13_A': '/workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/eval_dr/enhanced_summary.json',
    'challenger_enc16': '$CHAL/enhanced_summary.json',
}
axes = ['roll', 'pitch', 'vx', 'vy', 'vz', 'yaw']
levels = ['none', 'soft', 'medium', 'hard']
import sys
# Total aggregate ss_error
for name, p in paths.items():
    with open(p) as f: s = json.load(f)
    total_ss = sum(s[lv][a]['ss_error'] for lv in levels for a in axes)
    total_std = sum(s[lv][a]['ss_error_std'] for lv in levels for a in axes)
    total_gt20 = sum(s[lv][a].get('n_gt20', 0) for lv in levels for a in axes if a in ('roll', 'pitch'))
    print(f'{name}: sum_ss={total_ss:.3f} sum_std={total_std:.3f} sum_gt20={total_gt20}')
    for lv in levels:
        row = [f'{a}={s[lv][a][\"ss_error\"]:.3f}' for a in axes]
        print(f'  [{lv}] ' + '  '.join(row))
"
```

- [ ] **Step 3: Apply decision rule**

Decision rule (per spec):
1. Primary: sum of `ss_error` across all `{none, soft, medium, hard} × {roll, pitch, vx, vy, vz, yaw}`; lowest wins.
2. Tie-break (gap within 5%): sum of `ss_error_std`.
3. Tertiary: sum of `n_gt20` + seg1-9 hard `peak_roll` mean from switching.

Decision:
- **Challenger wins** (lower aggregate): baseline = challenger, canonical cfg remains hist5_act3+latent=16 (already in main).
- **r13_A wins**: baseline = r13_A, canonical cfg reverts to hist_len=3+hist_action_len=2+obs=87+latent=9.

- [ ] **Step 4: Append decision to baseline_selection.md**

```markdown
## Phase 0.7 Final Decision (2026-04-??)

### Head-to-head aggregate

| Run | sum_ss | sum_std | sum_gt20 | seg1-9 peak_roll (mean) |
|---|---|---|---|---|
| r13_A | ... | ... | ... | 7.63° |
| challenger_enc16 | ... | ... | ... | ... |

### Per-axis per-level delta (challenger − r13_A, negative = challenger wins)

(paste per-axis table)

### Decision

Final baseline: <r13_A | challenger_enc16>
Canonical env cfg: <hist_len=3,obs=87,latent=9 | hist_len=5,obs=121,latent=16>
Rationale: <paragraph>
```

### Task 0.7.2: Lock canonical env cfg

**If challenger wins**: main is already at hist5_act3+latent=16 from Task 0.6.2. No further edits. Skip to Task 0.7.3.

**If r13_A wins**: revert `config.py` and `rsl_rl_ppo_cfg.py` to r13_A values.

- [ ] **Step 1: If r13_A wins, confirm no active training**

Same check as Task 0.6.1 Step 2. Ask user.

- [ ] **Step 2: If r13_A wins, revert config**

Use Edit tool:
- `config.py`: `observation_space=87, hist_len=3, hist_action_len=2`
- `rsl_rl_ppo_cfg.py`:
  - `_EncoderPolicyCfg.policy_obs_dim=87, encoder_latent_dim=9`
  - `_FullDOFNoEncoderPolicyCfg.policy_obs_dim=87`

- [ ] **Step 3: If r13_A wins, smoke-test reverted cfg**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-TRPO-v0 \
  --num_envs 64 --max_iterations 5 --headless
```

- [ ] **Step 4: If r13_A wins, commit revert**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py
git commit -m "ablation: revert to r13_A env cfg after Phase 0.7 challenger comparison"
```

### Task 0.7.3: Commit the decision record

- [ ] **Step 1: Commit baseline_selection.md**

```bash
cd /workspace/isaaclab
git add logs/rsl_rl/fulldof_albc/ablation_sweep/baseline_selection.md
git commit -m "ablation: record final baseline decision from Phase 0.7"
```

- [ ] **Step 2: Update changelog**

```bash
cat >> /workspace/isaaclab/changelog.md <<'EOF'

2026-04-?? Ablation Phase 0.6/0.7: baseline challenger + final decision
- Challenger: hist5_act3 + encoder_latent_dim=16
- Winner: <r13_A | challenger>
- Canonical env cfg: <brief summary>
EOF
```

---

## Phase 0.5: Baseline code review + 500-iter sanity runs

> **Dependency**: Phase 0.5 runs under the canonical env cfg LOCKED at Phase 0.7. Do not start until Phase 0.7 Task 0.7.3 is committed.

### Rationale

`ActorCriticAsymConstrained` (class body untouched since 2026-04-08) and standard rsl-rl `ActorCritic` have never been run for 5000 iter under the post-r14 env (action latency, k_bias reward, OU drift, angular ocean current, r14 Hard DR). Additionally, if Phase 0.7 selected the challenger cfg (hist_len=5, hist_action_len=3, obs=121, latent=16), neither class has ever been run on that obs dim. A 500-iter sanity run catches interaction bugs early.

### Task 0.5.1: Read baseline policy class code

**Files:**
- Read: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/encoder/actor_critic_asym_constrained.py` (entire file, ~150 lines)
- Read: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/encoder/_policy_base.py`

- [ ] **Step 1: Check that the class handles current obs_groups correctly**

Look for:
- `_get_actor_obs(obs)` returns only `obs[policy_key]` (87D or canonical dim), NOT cat with privileged
- `_get_critic_obs(obs)` returns `cat([o_t, p_t])` = canonical_dim + 24D
- `update_normalization(obs)` properly updates actor_obs_normalizer with o_t dim

- [ ] **Step 2: Confirm no hardcoded dim assumptions**

Grep for any `81`, `87`, `121` literals in the class body. If found, these are stale — replace with `policy_obs_dim` attribute references.

```bash
grep -n "81\|87\|121\|178\|113" /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/encoder/actor_critic_asym_constrained.py
```

Expected: no hits (except in comments or docstrings describing dims).

- [ ] **Step 3: Confirm it correctly handles `num_constraints` auto-sync**

`load_state_dict` should inject defaults for cost_critic when loading a checkpoint lacking it (already present in the file). Verify `self.cost_critic is not None` guards are present for `num_constraints > 0` paths.

### Task 0.5.2: 500-iter sanity run for `#2 NoEncoder`

**Files:**
- Log: `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/<ts>_sanity_noenc/`

- [ ] **Step 1: Launch 500-iter run, GPU 0**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-NoEncoder-v0 \
  --num_envs 2048 --max_iterations 500 --headless \
  --run_name sanity_noenc \
  2>&1 | tee /workspace/sanity_noenc.log
```

Expected runtime: ~35-40 min.

- [ ] **Step 2: Acceptance check**

Run `train-analyze` skill on the sanity run.

Acceptance criteria:
- Reward: monotone rising or at least positive by iter 400
- Line search success rate > 0.5
- No crashes / NaN
- Constraint cost values finite (not exploding)
- Encoder-related TB metrics absent (expected — no encoder)

If any criterion fails, diagnose with train-analyze output + any stderr traceback. Do NOT proceed to Phase 1 until root cause found and fixed.

### Task 0.5.3: 500-iter sanity run for `#5 PurePPO`

Same as Task 0.5.2 but `--task Isaac-FullDOF-PPO-v0 --run_name sanity_pureppo`.

Acceptance criteria:
- Reward: positive and rising
- PPO surrogate loss decreasing
- No crashes / NaN
- Cost extras silently ignored (PPO doesn't read them)

### Task 0.5.4: 5-iter smoke for `#3` and `#4` (after Phase 2 / Phase 3 implementation)

These are new task registrations (see Phase 2 / Phase 3). The 5-iter smoke at the end of implementation (Task 2.4, Task 3.1) acts as the Phase 0.5 sanity check for these variants. No additional work here.

---

## Phase 1: Variant #2 (`NoEncoder`)

### Task 1.1: Verify NoEncoder still compatible with current env

**Files:**
- Read: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py:249-299`

- [ ] **Step 1: Check `policy_obs_dim` matches current env `observation_space`**

```bash
grep "policy_obs_dim\|observation_space" \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py \
  | grep -v "^#"
```

Expected: `observation_space` in config matches `policy_obs_dim` in `_FullDOFNoEncoderPolicyCfg`. If mismatch, edit `rsl_rl_ppo_cfg.py:271` and `rsl_rl_ppo_cfg.py:129` to match.

- [ ] **Step 2: Smoke-test the NoEncoder task with 5 iters**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-NoEncoder-v0 \
  --num_envs 64 --max_iterations 5 --headless
```

Expected: 5 iters pass, reward log appears, cost critic initialized for 10 constraints (from env auto-sync). Any ValueError on obs/action shape is a blocker; debug and fix before proceeding.

### Task 1.2: Launch NoEncoder training (5000 iter)

**Files:**
- Log: `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/<ts>_noenc/`

- [ ] **Step 1: Start training, GPU 0, single seed 30**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-NoEncoder-v0 \
  --num_envs 2048 --max_iterations 5000 --headless \
  --logger wandb --log_project_name fulldof_albc --run_name ablation_noenc \
  2>&1 | tee /workspace/ablation_noenc.log
```

Expected runtime: ~5 hr on RTX 4070.

- [ ] **Step 2: Periodic monitor (every 500 iter) with train-analyze skill**

While training runs, invoke the train-analyze skill to inspect progress at iteration milestones (500, 1500, 3000, 4500). Halt and investigate only if reward diverges (negative for >500 iters) or line_search_success < 0.3 sustained.

- [ ] **Step 3: Verify training completion**

```bash
LATEST=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_ablation_noenc | head -1)
ls "$LATEST/model_4999.pt"
```

Expected: file exists.

### Task 1.3: Run eval_dr + eval_dr_switching for NoEncoder

**Files:**
- Write: `<run>/eval_dr/`, `<run>/eval_dr_switching/`

- [ ] **Step 1: Run eval_dr_fulldof**

```bash
cd /workspace/isaaclab
LATEST=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_ablation_noenc | head -1)
CKPT="$LATEST/model_4999.pt"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_fulldof.py \
  --task Isaac-FullDOF-NoEncoder-v0 --num_envs 64 --headless \
  --checkpoint "$CKPT"
```

Expected: 4 DR levels each produce `eval_<level>.npz`, summary plots, `enhanced_summary.json`.

- [ ] **Step 2: Run eval_dr_switching**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_switching.py \
  --task Isaac-FullDOF-NoEncoder-v0 --num_envs 64 --headless \
  --checkpoint "$CKPT" --seed 42 --segment_duration 5.0 --num_segments 10
```

Expected: `eval_dr_switching/` directory with segment-level metrics.

- [ ] **Step 3: Inspect plots with Read tool**

Use Read on `summary_att.png`, `summary_lin_vel.png`, `summary_yaw.png` from `<run>/eval_dr/`. If failures visible (large divergence, clipping), note in changelog.

- [ ] **Step 4: Update changelog**

```bash
cat >> /workspace/isaaclab/changelog.md <<'EOF'

2026-04-?? Ablation Phase 1: NoEncoder (#2) complete
- Training: /workspace/isaaclab/logs/rsl_rl/fulldof_albc/<ts>_ablation_noenc/
- Eval: eval_dr/ + eval_dr_switching/
- Notes: <1-2 line summary>
EOF
```

---

## Phase 2: Variant #3 (`TRPO-NoIPO`) — NEW IMPLEMENTATION

### Task 2.1: Create `ALBCNoConstraintEnvCfg`

**Files:**
- Create: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config_noconstraint.py`

- [ ] **Step 1: Write the file**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env cfg with empty constraints (used by TRPO-NoIPO ablation variant).

Inherits ALBCEnvCfg verbatim except constraints.terms is emptied. The
ConstraintEncoderRunner auto-sync will then set num_constraints=0 in both
algorithm and policy cfg, fully disabling IPO barrier and cost critic.
Reward, DR, DORAEMON, action space, observation space are unchanged.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .config import ALBCEnvCfg
from .mdp.constraints import ALBCConstraintCfg


@configclass
class ALBCNoConstraintEnvCfg(ALBCEnvCfg):
    """Empty constraint list; everything else same as ALBCEnvCfg."""

    constraints: ALBCConstraintCfg = ALBCConstraintCfg(terms=[])
```

- [ ] **Step 2: Verify it loads**

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -c "from isaaclab_tasks.direct.constrained_full_albc.config_noconstraint import ALBCNoConstraintEnvCfg; c = ALBCNoConstraintEnvCfg(); print('constraints:', len(c.constraints.terms), 'obs:', c.observation_space)"
```

Expected: `constraints: 0 obs: <current obs dim>`.

### Task 2.2: Create `FullDOFTRPONoIPORunnerCfg`

**Files:**
- Create: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/ablation_cfgs.py`

- [ ] **Step 1: Write the runner cfg file**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner configurations for ablation variants #3 (TRPO-NoIPO) and #4 (PPO-Enc).

Separated from rsl_rl_ppo_cfg.py to keep the main config file untouched.
Reuses _EncoderPolicyCfg / RslRlConstraintTRPOAlgorithmCfg for shape and
hyperparameter parity with the main method.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from .rsl_rl_ppo_cfg import (
    _FullDOFPolicyCfg,
    _FullDOFPPOPolicyCfg,
    RslRlConstraintTRPOAlgorithmCfg,
    _FullDOFPPOAlgorithmCfg,
)


# =============================================================================
# Variant #3: TRPO-NoIPO (encoder + TRPO, no IPO)
# =============================================================================


@configclass
class _FullDOFNoIPOPolicyCfg(_FullDOFPolicyCfg):
    """Encoder policy with num_constraints=0 (cost critic suppressed at build)."""

    num_constraints: int = 0


@configclass
class _FullDOFNoIPOAlgorithmCfg(RslRlConstraintTRPOAlgorithmCfg):
    """ConstraintTRPO with num_constraints=0. IPO barrier is skipped when K=0."""

    num_constraints: int = 0
    constraint_budgets: tuple[float, ...] = ()


@configclass
class FullDOFTRPONoIPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Encoder + TRPO, no IPO. Uses ALBCNoConstraintEnvCfg."""

    class_name: str = "FullDOFConstraintEncoderRunner"
    seed: int = 30
    num_steps_per_env: int = 64
    max_iterations: int = 2500
    save_interval: int = 100
    experiment_name: str = "fulldof_albc"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }
    normalize_value: bool = False

    algorithm = _FullDOFNoIPOAlgorithmCfg()
    policy = _FullDOFNoIPOPolicyCfg()


# =============================================================================
# Variant #4: PPO-Enc (encoder + PPO, no IPO)
# =============================================================================


@configclass
class _FullDOFPPOEncPolicyCfg(_FullDOFPolicyCfg):
    """Encoder policy class for PPO. num_constraints=0 skips cost critic."""

    num_constraints: int = 0


@configclass
class FullDOFPPOEncRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Encoder + PPO (no IPO). Uses ALBCNoConstraintEnvCfg.

    Uses standard rsl-rl OnPolicyRunner (default class_name) so the
    ConstraintEncoderRunner auto-sync does NOT force num_constraints back to
    the env's K. Encoder metrics logging is lost here -- acceptable since
    the encoder is only a representation layer for PPO in this variant.
    """

    seed: int = 30
    num_steps_per_env: int = 64
    max_iterations: int = 2500
    save_interval: int = 100
    experiment_name: str = "fulldof_albc"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _FullDOFPPOAlgorithmCfg()
    policy = _FullDOFPPOEncPolicyCfg()
```

- [ ] **Step 2: Export runner cfgs from `agents/__init__.py`**

Edit `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/__init__.py`:

```python
from .rsl_rl_ppo_cfg import (
    FullDOFTRPORunnerCfg,
    FullDOFNoEncoderRunnerCfg,
    FullDOFPPORunnerCfg,
)
from .ablation_cfgs import (
    FullDOFTRPONoIPORunnerCfg,
    FullDOFPPOEncRunnerCfg,
)
```

Exact operation: read existing file first, append the second import block (do not remove existing exports).

### Task 2.3: Register Isaac-FullDOF-TRPO-NoIPO-v0 and Isaac-FullDOF-PPO-Enc-v0

**Files:**
- Modify: `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/__init__.py`

- [ ] **Step 1: Read the existing file**

```bash
cat /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/__init__.py
```

- [ ] **Step 2: Append two new gym.register blocks after the last existing one**

Add at end of file (before final newline):

```python
gym.register(
    id="Isaac-FullDOF-TRPO-NoIPO-v0",
    entry_point="isaaclab_tasks.direct.constrained_full_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config_noconstraint:ALBCNoConstraintEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.ablation_cfgs:FullDOFTRPONoIPORunnerCfg",
    },
)

gym.register(
    id="Isaac-FullDOF-PPO-Enc-v0",
    entry_point="isaaclab_tasks.direct.constrained_full_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config_noconstraint:ALBCNoConstraintEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.ablation_cfgs:FullDOFPPOEncRunnerCfg",
    },
)
```

### Task 2.4: Smoke-test `Isaac-FullDOF-TRPO-NoIPO-v0`

- [ ] **Step 1: Run 5-iter smoke test**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-TRPO-NoIPO-v0 \
  --num_envs 64 --max_iterations 5 --headless
```

Expected: logs show `num_constraints=0`, no cost critic instantiation, 5 iters complete.

Known likely issue: `ConstraintEncoderRunner.__init__` may call `cost_critic.parameters()` unconditionally. If so, the runner log will show a crash on iteration 0. Fix by reading `constraint_encoder_runner.py:70-160` for cost critic references and guarding each call behind `if self.alg.num_constraints > 0`.

- [ ] **Step 2: If step 1 crashes, inspect and fix**

```bash
grep -n "cost_critic\|num_constraints\|evaluate_costs" \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/runners/constraint_encoder_runner.py \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/algorithms/constraint_trpo.py
```

For each unguarded reference, wrap in a conditional. Concrete fix pattern:

```python
# Before:
cost_grad = self.policy.evaluate_costs(obs)

# After:
if self.num_constraints > 0:
    cost_grad = self.policy.evaluate_costs(obs)
```

- [ ] **Step 3: Re-run smoke test**

Same command as Step 1. Expected: 5 iters complete cleanly.

### Task 2.5: Launch TRPO-NoIPO training (5000 iter)

- [ ] **Step 1: Start training**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-TRPO-NoIPO-v0 \
  --num_envs 2048 --max_iterations 5000 --headless \
  --logger wandb --log_project_name fulldof_albc --run_name ablation_nocstr \
  2>&1 | tee /workspace/ablation_nocstr.log
```

Expected runtime: ~5 hr.

- [ ] **Step 2: Monitor with train-analyze at milestones**

Same pattern as Task 1.2 Step 2.

- [ ] **Step 3: Verify model_4999.pt exists**

```bash
LATEST=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_ablation_nocstr | head -1)
ls "$LATEST/model_4999.pt"
```

### Task 2.6: Run eval_dr + eval_dr_switching for TRPO-NoIPO

Same steps as Task 1.3 but substitute task id `Isaac-FullDOF-TRPO-NoIPO-v0` and run prefix `ablation_nocstr`.

- [ ] **Step 1: eval_dr_fulldof**

```bash
cd /workspace/isaaclab
LATEST=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_ablation_nocstr | head -1)
CKPT="$LATEST/model_4999.pt"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_fulldof.py \
  --task Isaac-FullDOF-TRPO-NoIPO-v0 --num_envs 64 --headless --checkpoint "$CKPT"
```

- [ ] **Step 2: eval_dr_switching**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_switching.py \
  --task Isaac-FullDOF-TRPO-NoIPO-v0 --num_envs 64 --headless \
  --checkpoint "$CKPT" --seed 42 --segment_duration 5.0 --num_segments 10
```

- [ ] **Step 3: Inspect plots + update changelog**

Read `summary_att.png`, `summary_lin_vel.png`, `summary_yaw.png`. Append to `/workspace/isaaclab/changelog.md`.

---

## Phase 3: Variant #4 (`PPO-Enc`) — NEW IMPLEMENTATION

### Task 3.1: Smoke-test `Isaac-FullDOF-PPO-Enc-v0` (registered in Task 2.3)

- [ ] **Step 1: Run 5-iter smoke test**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-PPO-Enc-v0 \
  --num_envs 64 --max_iterations 5 --headless
```

Expected: logs show rsl_rl `OnPolicyRunner` (not ConstraintEncoderRunner), encoder module loaded via `class_name="FullDOFActorCriticEncoder"`, 5 iters complete.

Known risks:
- Standard rsl-rl `OnPolicyRunner` may not recognize `FullDOFActorCriticEncoder` class_name if it isn't registered in `rsl_rl.runners.on_policy_runner` namespace. `rsl_rl_ppo_cfg.py` already installs it at module import (`_runner_module.FullDOFActorCriticEncoder = ActorCriticEncoder`). Verify this by ensuring `ablation_cfgs.py` imports `rsl_rl_ppo_cfg` so the side effect runs. The file structure shows it already does via `from .rsl_rl_ppo_cfg import ...`.
- `RslRlPpoActorCriticCfg` default kwargs (`actor_hidden_dims`, `critic_hidden_dims`) may clash with encoder-style kwargs. `_FullDOFPolicyCfg` already includes both sets, so `to_dict()` should serialize all the encoder-specific fields too.
- PPO's `num_actor_obs` auto-calc from `obs_groups["policy"]` will sum `[policy, privileged]` = 87+24 = 111, while the encoder policy expects `policy_obs_dim=87` and `privileged_dim=24` separately. The standard rsl-rl `ActorCritic.__init__` takes `num_actor_obs` as an int; the custom `ActorCriticEncoder.__init__` ignores `num_actor_obs` and reads from obs_groups via `PolicyBase._init_base`. Collision-free as long as PolicyBase's obs-group handler processes the `obs` TensorDict (it does — see `_policy_base.py:64-78`).

- [ ] **Step 2: If step 1 fails**

Inspect the error. Common fixes:

(a) `KeyError: 'FullDOFActorCriticEncoder'` — add explicit import at top of `ablation_cfgs.py`:

```python
# Force registration of custom classes in rsl_rl.runners.on_policy_runner namespace
from . import rsl_rl_ppo_cfg  # noqa: F401
```

(b) `TypeError: __init__() got unexpected keyword 'num_actor_obs'` — add `num_actor_obs` and `num_critic_obs` as ignored kwargs in `ActorCriticEncoder.__init__` (already done via `**kwargs` with warning — check that the kwarg name isn't hard-checked).

(c) `ValueError: Policy obs dim X != expected Y` — indicates obs_groups mismatch. Expected: `policy` group's first key should be the 87D (or current) obs, second key the 24D privileged. Current `obs_groups={"policy": ["policy", "privileged"]}` satisfies this.

- [ ] **Step 3: If persistent incompatibility, implement `EncoderOnPolicyRunner`**

Create `/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/runners/encoder_ppo_runner.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner subclass that preserves PPO semantics and knows how to
instantiate ActorCriticEncoder without auto-syncing num_constraints.

Used by Variant #4 (PPO-Enc). Minimal code -- just bridges rsl-rl PPO to
our custom policy class that inherits from PolicyBase rather than ActorCritic.
"""

from __future__ import annotations

import logging

from rsl_rl.runners import OnPolicyRunner

logger = logging.getLogger(__name__)


class EncoderOnPolicyRunner(OnPolicyRunner):
    """PPO runner with encoder-aware logging.

    Does NOT auto-sync num_constraints (unlike ConstraintEncoderRunner).
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self._has_encoder = hasattr(self.alg.policy, "encoder")
        if self._has_encoder:
            logger.info("[EncoderOnPolicyRunner] Encoder detected. PPO + Encoder mode.")
```

Then register it:

```python
# In rsl_rl_ppo_cfg.py or ablation_cfgs.py
import rsl_rl.runners.on_policy_runner as _runner_module
from ..runners.encoder_ppo_runner import EncoderOnPolicyRunner
_runner_module.FullDOFEncoderOnPolicyRunner = EncoderOnPolicyRunner
```

And set in `FullDOFPPOEncRunnerCfg`:

```python
class_name: str = "FullDOFEncoderOnPolicyRunner"
```

Re-run smoke test.

### Task 3.2: Launch PPO-Enc training (5000 iter)

- [ ] **Step 1: Start training**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-PPO-Enc-v0 \
  --num_envs 2048 --max_iterations 5000 --headless \
  --logger wandb --log_project_name fulldof_albc --run_name ablation_ppoenc \
  2>&1 | tee /workspace/ablation_ppoenc.log
```

- [ ] **Step 2: Monitor + verify**

Same pattern as Task 1.2 Steps 2 and 3. Verify `model_4999.pt`.

### Task 3.3: Run eval_dr + eval_dr_switching for PPO-Enc

- [ ] **Step 1: eval_dr_fulldof**

```bash
cd /workspace/isaaclab
LATEST=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_ablation_ppoenc | head -1)
CKPT="$LATEST/model_4999.pt"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_fulldof.py \
  --task Isaac-FullDOF-PPO-Enc-v0 --num_envs 64 --headless --checkpoint "$CKPT"
```

- [ ] **Step 2: eval_dr_switching**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/analysis/eval_dr_switching.py \
  --task Isaac-FullDOF-PPO-Enc-v0 --num_envs 64 --headless \
  --checkpoint "$CKPT" --seed 42 --segment_duration 5.0 --num_segments 10
```

- [ ] **Step 3: Inspect plots + changelog**

---

## Phase 4: Variant #5 (`PurePPO`)

### Task 4.1: Verify compatibility

- [ ] **Step 1: Check `_FullDOFPPOPolicyCfg` auto-sizes for current env**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-PPO-v0 \
  --num_envs 64 --max_iterations 5 --headless
```

Expected: 5 iters pass. Actor input = 87D, Critic input = 111D, no constraint bookkeeping.

### Task 4.2: Launch PurePPO training

- [ ] **Step 1: Start training**

```bash
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FullDOF-PPO-v0 \
  --num_envs 2048 --max_iterations 5000 --headless \
  --logger wandb --log_project_name fulldof_albc --run_name ablation_pureppo \
  2>&1 | tee /workspace/ablation_pureppo.log
```

- [ ] **Step 2: Monitor + verify `model_4999.pt`**

### Task 4.3: Run eval_dr + eval_dr_switching for PurePPO

Same pattern as Task 3.3 with task id `Isaac-FullDOF-PPO-v0` and run prefix `ablation_pureppo`.

---

## Phase 5: Cross-variant Analysis

### Task 5.1: Run analyze_eval_dr.py across all 5 variants

**Files:**
- Write: `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/ablation_sweep/analysis_report.md`

- [ ] **Step 1: Gather eval_dr paths**

Per Phase 0.7 decision, `$MAIN` points to either r13_A's eval_dr or the challenger's eval_dr.

```bash
# If Phase 0.7 chose r13_A:
MAIN=/workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-20_20-08-38_r13_A/eval_dr
# If Phase 0.7 chose challenger:
# MAIN=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*_challenger_hist5_act3_enc16 | head -1)/eval_dr

NOENC=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*ablation_noenc | head -1)/eval_dr
NOCSTR=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*ablation_nocstr | head -1)/eval_dr
PPOENC=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*ablation_ppoenc | head -1)/eval_dr
PUREPPO=$(ls -td /workspace/isaaclab/logs/rsl_rl/fulldof_albc/*ablation_pureppo | head -1)/eval_dr
echo "MAIN:    $MAIN"
echo "NOENC:   $NOENC"
echo "NOCSTR:  $NOCSTR"
echo "PPOENC:  $PPOENC"
echo "PUREPPO: $PUREPPO"
```

- [ ] **Step 2: Run multi-run analyze across all DR levels**

```bash
python3 /workspace/isaaclab/scripts/analysis/analyze_eval_dr.py \
  "$MAIN" "$NOENC" "$NOCSTR" "$PPOENC" "$PUREPPO" \
  --labels main noenc nocstr ppoenc pureppo \
  --levels none soft medium hard \
  --save-hist /workspace/isaaclab/logs/rsl_rl/fulldof_albc/ablation_sweep/peak_roll_hist.png \
  --hist-axis roll
```

Expected: tables for heavy-tail, sample-mean divergence, axis decorrelation across 5 runs × 4 DR levels.

- [ ] **Step 3: Run compare_dr.py for side-by-side plot**

```bash
python3 /workspace/isaaclab/scripts/analysis/compare_dr.py \
  --dirs "$MAIN" "$NOENC" "$NOCSTR" "$PPOENC" "$PUREPPO" \
  --labels main noenc nocstr ppoenc pureppo \
  --output /workspace/isaaclab/logs/rsl_rl/fulldof_albc/ablation_sweep/compare_all.png
```

- [ ] **Step 4: Write claim-specific reports**

Create `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/ablation_sweep/analysis_report.md` with:

```markdown
# Ablation Sweep Report (2026-04-??)

## Variants
(paste the 5-run table from spec)

## Claim A: Encoder contribution (main vs noenc)

Per-level per-axis delta table:

| DR | roll Δ | pitch Δ | vx Δ | vy Δ | vz Δ | yaw Δ |
|---|---|---|---|---|---|---|
| none | ... | ... | ... | ... | ... | ... |
| soft | ... | ... | ... | ... | ... | ... |
| medium | ... | ... | ... | ... | ... | ... |
| hard | ... | ... | ... | ... | ... | ... |

- Where Δ = (main.ss_error - noenc.ss_error). Negative = encoder helps.
- Include env-level CV comparison.

## Claim B: IPO contribution (main vs nocstr)

(same structure)

## Algorithm effect

- `nocstr` vs `ppoenc`: TRPO vs PPO at encoder + no-IPO
- `main` vs `pureppo`: full method vs naive baseline

## Heavy-tail and sample-mean divergence

(paste analyze_eval_dr.py output tables)

## Switching eval summary

(peak transient error and settling time per variant per DR level from eval_dr_switching)
```

- [ ] **Step 5: Inspect all summary plots**

Read `summary_att.png`, `summary_lin_vel.png`, `summary_yaw.png` from all 5 `eval_dr/` directories. For each, note qualitative differences (overshoot, settling, divergence between mean and sample env trajectories).

### Task 5.2: Lifecycle cleanup

- [ ] **Step 1: Commit new ablation configs and registrations**

```bash
cd /workspace/isaaclab
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config_noconstraint.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/ablation_cfgs.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/__init__.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/__init__.py
# Only if EncoderOnPolicyRunner was needed:
# git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/runners/encoder_ppo_runner.py
git commit -m "ablation: add TRPO-NoIPO and PPO-Enc variants for encoder/IPO ablation sweep"
```

- [ ] **Step 2: Archive ablation sweep artifacts**

```bash
cd /workspace/isaaclab
git add logs/rsl_rl/fulldof_albc/ablation_sweep/ changelog.md
git commit -m "ablation: archive sweep analysis report and baseline selection rationale"
```

- [ ] **Step 3: Remove hist-series worktrees (skip if user wants to keep)**

Per operations rule `Experiment Worktree Lifecycle`:

```bash
# For each worktree:
# 1. Ensure logs already migrated to /workspace/isaaclab/logs/
# 2. Remove worktree
git worktree list
# Example for each of hist5, hist5_act3, hist10, layernorm, r13a_hist5 etc.:
# git worktree remove /workspace/isaaclab-r13a_hist5_act3
# Confirm before each removal.
```

Before removing: verify `model_4999.pt` (and any other needed artifact) is present in `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/<ts>_<run>/` and ask the user to confirm.

- [ ] **Step 4: Remove sweep launch script**

```bash
rm -f /workspace/run_r13a_hist.sh
```

Only after confirming the sweep is fully archived.

### Task 5.3: Deliver summary to user

- [ ] **Step 1: Present claim-specific findings**

Summarize in a Korean response:
- Claim A status (encoder helps: yes/no/partially, with numbers)
- Claim B status (IPO helps: yes/no/partially)
- Algorithm-effect note (TRPO vs PPO)
- Any regressions found
- Recommended next step (paper writing, additional seed, HP tuning)

---

## Self-Review

### Spec coverage

Spec → Plan task mapping:

| Spec section | Plan task |
|---|---|
| Claim priority | Claim-specific reports in Phase 5 Task 5.1 Step 4 |
| 5-variant matrix | Phases 1, 2, 3, 4 (one per variant; Phase 0 for baseline initial selection) |
| Baseline initial selection | Phase 0 Task 0.1 (r13_A from user analysis) |
| Baseline challenger (hist5_act3 + latent=16) | Phase 0.6 Tasks 0.6.1–0.6.4 |
| Final baseline decision + canonical cfg lock | Phase 0.7 Tasks 0.7.1–0.7.3 |
| Pre-flight sanity for #2/#5 | Phase 0.5 (runs AFTER Phase 0.7) |
| Hyper-parameter policy | Encoded in runner cfg files (Task 2.2); main's params inherited |
| Implementation for #3 (TRPO-NoIPO) | Phase 2 Tasks 2.1–2.6 |
| Implementation for #4 (PPO-Enc) | Phase 3 Tasks 3.1–3.3 |
| Evaluation protocol | Each phase's final task + Phase 5 Task 5.1 |
| Acceptance criteria | Phases 1–4 final tasks (`model_4999.pt` + eval artifacts), Phase 5 (analysis report) |
| Risk mitigations | Task 2.4 Step 2 (num_constraints guards), Task 3.1 Steps 2–3 (PPO+Encoder fallback) |
| Lifecycle | Task 5.2 |

### Phase order

Read top-to-bottom as: Phase 0 → Phase 0.6 → Phase 0.7 → Phase 0.5 → Phases 1–4 → Phase 5.
Phase 0.5 appears between 0 and 1 in document order but executes AFTER 0.7 by dependency.

### Placeholder scan

- No "TBD"/"TODO" in plan.
- All code blocks contain the concrete code to write/run.
- All bash commands list exact paths or parameterized via `$VAR`.
- Baseline-selection result fills in `<selected>` once Task 0.4 completes; this is a deliberate runtime variable, not a placeholder — the plan's procedure is concrete.

### Type consistency

- `FullDOFTRPONoIPORunnerCfg` uses `_FullDOFNoIPOPolicyCfg` (inherits `_FullDOFPolicyCfg`) and `_FullDOFNoIPOAlgorithmCfg` (inherits `RslRlConstraintTRPOAlgorithmCfg`).
- `FullDOFPPOEncRunnerCfg` uses `_FullDOFPPOEncPolicyCfg` (inherits `_FullDOFPolicyCfg`) and `_FullDOFPPOAlgorithmCfg`.
- `ALBCNoConstraintEnvCfg` inherits `ALBCEnvCfg`, used by both new tasks.
- Consistent naming: `ablation_noenc`, `ablation_nocstr`, `ablation_ppoenc`, `ablation_pureppo` across train.py `--run_name`, log directory prefix, and eval script arguments.

---

## Risks and Mitigations (runtime)

| Risk | When | Mitigation |
|---|---|---|
| Baseline selection table incomplete (some runs lack switching eval) | Task 0.3 | Fill N/A explicitly; if ss_error gap is small, skip the tied run rather than blocking |
| ConstraintEncoderRunner auto-sync panics on num_constraints=0 | Task 2.4 | Step 2 provides concrete fix pattern for unguarded evaluate_costs calls |
| rsl-rl PPO rejects ActorCriticEncoder class | Task 3.1 | Step 3 provides `EncoderOnPolicyRunner` subclass fallback |
| Training diverges (reward → negative sustained) | Phases 1–4 | Train-analyze skill run at milestones; stop training if confirmed diverged; do NOT adjust HPs mid-sweep (violates variable control); document and rerun |
| GPU 0 OOM at num_envs=2048 | Phases 1–4 | Lower to 1024; document in changelog; keep across all variants for parity |
| eval_dr crashes on a variant | Task X.3 | Re-run with `--num_envs 32`; if still crashes, investigate for variant-specific config bug |
