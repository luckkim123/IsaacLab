# Arm Freeze Root Cause Analysis (2026-03-26)

## Problem

Training run 2026-03-26_17-10-11 (tanh squashing + raw action storage + barrier_t=100):
policy converges to fixed max-extension position within 100 iterations.
action_size_mean=1.41 (near max sqrt(2)), rate_mean=0.
Roll error plateaued at 40deg, pitch at 20deg. No meaningful learning.

## Root Cause: H1 (tanh Saturation + Flat Reward Landscape)

action_size=1.41 means both axes at tanh boundary (|a|=0.99, pre-tanh mu~2.65).
With sigma=0.49, sampling at mu=2.65 produces:
- tanh output range: [0.94, 0.998] (compressed)
- EE position range: 0.022m (2.2cm) out of 0.922m total workspace
- Joint angle diversity: g1_std=0.81deg, **g2_std=0.00deg**
- All samples produce identical physical outcome -> advantage = noise -> no gradient signal

The policy reaches boundary because max extension = max restoring torque (physically true).
Once there, it cannot escape: no gradient signal + sigma shrinking + KL budget insufficient.

## Why Only Our System

| System         | Action meaning     | Squashing | Optimal at boundary? |
|----------------|-------------------|-----------|---------------------|
| RSL-RL PPO     | unbounded -> clamp | none      | no (mid-range)      |
| Hero Agent     | joint velocity     | clamp     | no (0 = stop)       |
| HORA           | unbounded -> clamp | none      | no                  |
| Factory (manip)| delta EE           | clamp     | no (0 = hold)       |
| SAC systems    | tanh-Gaussian      | tanh      | entropy prevents    |
| **Our system** | **absolute EE pos**| **tanh**  | **yes (max ext)**   |

Our unique combination: TRPO + absolute EE position + physical optimum at boundary.

## Hypothesis Verification Summary

| Hypothesis | Status | Key Evidence |
|-----------|--------|-------------|
| H1: tanh saturation -> flat landscape | **CONFIRMED** | g2_std=0.00deg at boundary |
| H2: smoothness -> stillness attractor | PARTIAL | rate 1.18->0.05 in 140 iters |
| H3: sigma -> exploration collapse | CONFIRMED | 0.99->0.49, Case 3 dynamics |
| H4: constraint -> implicit stillness | REJECTED | barrier grad 15% of total |
| H5: value function -> low SNR | MODERATE | shs=0.00028, step_norm=0.05 |
| H6: KL + tanh = action paralysis | CONFIRMED | delta_a_max=0.0015 at boundary |

## Solution: Delta EE Mode

Change from absolute EE position to delta EE position:
```
Current:  action -> x_des = action * R               (absolute, optimal at boundary)
Proposed: action -> delta = action * delta_scale      (delta, optimal at 0)
          EE_target = current_EE + delta
```

This centers the optimal action at (0, 0), eliminates boundary trap, and preserves
EE-space control intuition (x -> pitch, y -> roll mapping).

Code change scope: `_apply_ee_position_action()` in albc_env.py + new config params.
Existing IK + rate limiting reused entirely.

## Numerical Evidence

Track A (Checkpoint):
- log_std: [-0.33, -0.29], sigma: [0.72, 0.75]
- Actor final bias: [-0.27, -0.02] (MLP doesn't inherently output large mu)

Track B (Rewards, 683 iters):
- Episode command: -0.32 -> -2.50 (worsened due to longer episodes)
- Episode smoothness: -0.16 -> -0.01 (16x improved = not moving)
- TRPO step_norm: 0.24 -> 0.05 (5x smaller)

Track C (Barriers):
- Barrier gradient total ~0.023 (15% of TRPO grad_norm 0.15)
- All constraint margins large, not driving stillness

Track D (Physical diversity):
- mu=0.0: EE range 0.62m, g1_std=105deg, g2_std=34deg
- mu=2.65: EE range 0.022m, g1_std=0.81deg, g2_std=0.00deg
- 28x less physical diversity at boundary

Track E (Sigma):
- Decline rate: 0.0023/iter (early) -> 0.0003/iter (late)
- Estimated min_std arrival: iter ~1657
- Score function: Case 3 (boundary concentrates sigma downward)

## H2: Smoothness Penalty as Trigger

Smoothness penalty (weight=-0.5) accelerates H1 trap entry:
- iter 0~40: |smoothness/command| = 50% (half of total gradient)
- iter 40~80: act_rate drops 87% (1.18 -> 0.10), act_size jumps to 1.36 (boundary)
- iter 120+: smoothness converges to ~0 (arm frozen), only command gradient remains

Smoothness penalty provides immediate, direct gradient to reduce action changes.
Command reward provides delayed, indirect gradient (action -> IK -> torque -> attitude).
TRPO follows the easier signal first -> arm stops moving -> enters H1 trap.

Action item: reduce smoothness_weight from -0.5 to ~-0.05 with delta EE mode.
Same applies to torque_weight (-0.001, already small but review needed).

## H3: Sigma Monotonic Collapse

Sigma (noise_std) optimizer never increases sigma. Monotonic 1.00 -> 0.46 over 760 iters.
At tanh boundary, ALL directions produce similar outcomes, so sigma reduction
is always "locally optimal" (less noise = less variance = lower surrogate loss).
entropy_coef=0 provides no upward pressure.

Combined with tanh compression: effective action_std = tanh'(mu) * sigma.
sigma 2x reduction (1.0->0.5) + tanh compression (tanh'(2.65)=0.02)
= 60x reduction in exploration range.

Delta EE mode fixes this by eliminating tanh compression (action_std = sigma directly).
entropy_coef=0 initially, consider small positive value (0.001-0.01) if sigma
still collapses too fast in delta EE mode.
