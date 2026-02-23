# SAC-MPC Training Health Monitoring Guide

> **Status**: 2026-02-22 | **Source**: `runners/sac.py`, `runners/sac_mpc_runner.py`, `controllers/mpc.py`, `controllers/dynamics_mlp.py`, `encoder/actor_critic_mpc.py`, `utils/logging.py`, `base_env.py`
>
> SAC-MPC 학습 시 WandB 대시보드에서 모니터링해야 할 메트릭 가이드.
> 각 컴포넌트(Dynamics MLP, Encoder, Actor-Critic, MPC)의 건강 상태를 판단하는 기준을 정리한다.

---

## 1. Metric Naming Convention

기존 Hero Agent 로깅 인프라의 prefix 체계를 따른다.
SAC-MPC 고유 메트릭은 `SAC/`, `Dynamics/`, `MPC/` prefix를 사용한다.

| Prefix | Source | Description |
|:-------|:-------|:------------|
| `Episode_Reward/` | `base_env._collect_episode_metrics()` | Per-reward-term breakdown (episode-length normalized) |
| `Episode_Termination/` | `base_env._collect_episode_metrics()` | Terminated / timeout counts |
| `Attitude_Error/` | `base_env._collect_episode_metrics()` | Roll/pitch tracking errors (deg) |
| `Action/` | `base_env._collect_episode_metrics()` | Action magnitude diagnostics |
| `Loss/` | `runners/sac.py` | Actor, critic, alpha, dynamics losses |
| `SAC/` | `runners/sac.py` | SAC-specific diagnostics (Q values, alpha, grad norms) |
| `Dynamics/` | `runners/sac.py` or `dynamics_mlp.py` | Dynamics MLP prediction health |
| `MPC/` | `controllers/mpc.py` or `encoder/actor_critic_mpc.py` | MPC solver diagnostics |
| `Encoder/` | `utils/logging.py:log_encoder_metrics()` | Encoder latent z health |
| `TDC/` | `utils/logging.py:log_tdc_diagnostics()` | TDC controller health (TDC variant only) |
| `DR/` | `utils/logging.py:log_dr_metrics()` | Domain randomization parameter stats |
| `Train/` | `runners/sac_mpc_runner.py` | Training loop diagnostics (fps, buffer) |

---

## 2. WandB Dashboard: Recommended Panel Layout

### Panel 1: "Is Training Working?" (Primary)

학습이 전체적으로 잘 진행되는지 1분 안에 판단할 수 있는 핵심 지표들.

| Metric | Healthy Sign | Warning Sign |
|:-------|:------------|:-------------|
| `Episode_Reward/total` | 꾸준히 상승 | Warmup 이후 flat 또는 하락 |
| `Train/episode_length` | 점점 길어짐 (오래 생존) | 최소값에 고정 또는 감소 |
| `Attitude_Error/roll_deg` | 0-5 deg로 수렴 | 20 deg 이상 정체 |
| `Attitude_Error/pitch_deg` | 0-5 deg로 수렴 | 20 deg 이상 정체 |

### Panel 2: "Reward Breakdown" (Per-Term)

개별 보상 항목의 기여도를 분석. `base_env._collect_episode_metrics()`에서 자동 수집.

| Metric | Meaning | Healthy Range |
|:-------|:--------|:--------------|
| `Episode_Reward/tracking` | Gaussian tracking reward (weight=3.0) | 상승 중, dominant positive |
| `Episode_Reward/action_magnitude` | Control effort penalty (weight=-0.1) | 작은 음수, -0.01 ~ -0.05 |
| `Episode_Reward/action_rate` | Action smoothness penalty (weight=-0.01) | 0에 근접, -0.001 ~ -0.01 |

**진단 요점**: tracking이 상승하지 않으면서 penalty만 누적되면 reward shaping 문제.
penalty가 tracking의 2배 이상이면 penalty domination (가중치 조정 필요).

### Panel 3: "Actor-Critic Health"

SAC 핵심 컴포넌트의 학습 안정성 지표.

| Metric | Healthy Sign | Warning Sign |
|:-------|:------------|:-------------|
| `Loss/actor` | 감소 추세 (비단조) | Diverge (>100) 또는 NaN |
| `Loss/critic` | 감소 후 안정화 | Explode (>10) |
| `SAC/q1_mean`, `SAC/q2_mean` | 유사한 값, 점진적 상승 | Q1/Q2 간 큰 괴리 |
| `SAC/target_q_mean` | Q1/Q2를 부드럽게 추종 (Polyak lag) | 급격한 점프 |
| `SAC/alpha` | 0.1-0.5 부근 안정화 | 0으로 collapse (탐색 소멸) 또는 >5 (과잉 노이즈) |
| `SAC/log_prob_mean` | target_entropy (-2) 부근 안정화 | 매우 음수 (<-10, 과랜덤) 또는 0 근처 (collapsed) |
| `SAC/actor_grad_norm` | 안정적, clip threshold (1.0) 미만 | 지속적 1.0 (항상 clipping) |
| `SAC/critic_grad_norm` | 안정적, 1.0 미만 | 지속적 1.0 |

**핵심 인사이트**: `SAC/alpha`는 탐색-활용 균형의 핵심 지표.
alpha가 0으로 가면 policy가 deterministic으로 collapse되어 local optima에 갇힌다.
alpha가 너무 높으면 행동이 랜덤에 가까워 학습이 안 된다.

### Panel 4: "Dynamics MLP Health"

Dynamics MLP는 직접적인 supervised signal (MSE vs true next state)로 학습하므로
모든 컴포넌트 중 가장 빠르게 수렴해야 한다.

| Metric | Healthy Sign | Warning Sign |
|:-------|:------------|:-------------|
| `Loss/dynamics` | 첫 1K iter에서 급감 | 정체 또는 증가 |
| `Dynamics/pred_err_mean` | 초기 학습 후 <0.01 | >0.1 (모델이 학습 안 됨) |
| `Dynamics/grad_norm` | 안정적, 1.0 미만 | 지속적 1.0 (clipping = 어려운 문제) |

**핵심 인사이트**: Dynamics MLP가 수렴하지 않으면 MPC solver도 무의미.
Dynamics가 잘 수렴한 후에야 actor-critic이 reward improvement를 보이기 시작한다.

### Panel 5: "Encoder Health"

Encoder가 privileged information을 latent z로 적절히 압축하는지 확인.

| Metric | Healthy Sign | Warning Sign |
|:-------|:------------|:-------------|
| `Encoder/z_mean` | Non-zero, 안정적 | 0으로 collapse 또는 diverge |
| `Encoder/z_std` | >0.01 (env별 z 값 다름) | 0 근접 (모든 env에서 동일 z = DR 미인코딩) |
| `Encoder/grad_norm` | Non-zero, 안정적 | 0.0 (gradient가 encoder로 흐르지 않음) |

**핵심 인사이트**: SAC-MPC에서 encoder gradient는 다음 경로로 흐른다:
`actor_loss -> MPC solve -> cost_map -> encoder`.
`Encoder/grad_norm`이 0이면 MPC->encoder gradient chain이 끊긴 것.
MPC의 differentiable solve (1-step GD with `create_graph=True`)가 정상 작동하는지 확인 필요.

### Panel 6: "MPC Health"

MPC solver의 cost function 파라미터와 최적화 상태 모니터링.

| Metric | Healthy Sign | Warning Sign |
|:-------|:------------|:-------------|
| `MPC/Q_diag_mean` | >0, 학습 중 적응 | 0 또는 음수로 collapse |
| `MPC/R_diag_mean` | >0, state vs control 비용 균형 | 0 (제어 페널티 없음) 또는 매우 큰 값 |
| `MPC/solve_cost` | 학습 진행에 따라 감소 | 정체 또는 NaN |
| `MPC/state_err_total` | 감소 (tracking 개선) | 정체 |

**핵심 인사이트**: Q와 R의 비율이 중요.
Q >> R이면 MPC가 state tracking에만 집중 (jerky control).
R >> Q이면 MPC가 제어 최소화에 집중 (느린 반응).

### Panel 7: "Training Diagnostics"

학습 루프의 기본적인 건전성 확인.

| Metric | Description | Expected |
|:-------|:-----------|:---------|
| `Train/fps` | Throughput | 64 envs 기준 ~500-2000 |
| `Train/buffer_size` | Replay buffer 크기 | Capacity까지 성장 후 일정 |
| `Train/terminated` | 조기 종료 (angular_vel 또는 NaN) | 학습 진행에 따라 감소 |
| `Train/time_out` | Max-step timeout | 학습 진행에 따라 증가 (full episode 생존) |

### Panel 8: "DR Parameters" (Sanity Check)

Domain randomization이 정상 작동하는지 확인. 값에 분산이 있어야 한다.

| Metric | Expected |
|:-------|:---------|
| `DR/buoyancy_force_mean` | ~26N 부근, 분산 존재 |
| `DR/inertia_mean` | 분산 존재 (0.4x-2.5x range) |
| `DR/payload_mass_mean` | 0-2 kg 범위 |
| `DR/ocean_current_mag_mean` | 0-0.5 m/s 범위 |

**주의**: 모든 DR 메트릭이 상수면 randomization이 비활성화된 것.
Debug env (`Isaac-HeroAgent-v0`)에서는 정상적으로 상수.

---

## 3. Training Phase Timeline

```
Iter 0-10:       Warmup (random actions, gradient update 없음)
                 -> Reward: 낮고 불안정, dynamics loss: 높음

Iter 10-100:     Dynamics MLP 수렴기
                 -> Loss/dynamics: 급감, Dynamics/pred_err_mean < 0.01
                 -> Actor-critic: 아직 미학습 (reward 정체)

Iter 100-500:    Actor-critic 학습 시작
                 -> Reward 상승 시작, alpha 안정화
                 -> Q values 점진적 상승, MPC solve_cost 감소 시작

Iter 500-2000:   수렴기
                 -> Reward 꾸준히 상승, attitude error 감소
                 -> MPC/state_err_total 감소
                 -> Episode length 증가 (더 오래 생존)

Iter 2000+:      Plateau
                 -> Reward 수렴, errors 안정화
                 -> 추가 개선 여지: DR 강도 증가, horizon 연장
```

**주요 순서 관계**: Dynamics MLP >> Actor-Critic >> Encoder 순서로 수렴.
Dynamics가 수렴하지 않은 상태에서 actor-critic이 학습되면 잘못된 dynamics model에 overfitting.

---

## 4. Troubleshooting Flowchart

### Reward가 전혀 오르지 않을 때

```
1. Loss/dynamics 확인
   -> 높음 (>0.1): Dynamics MLP 문제 -> learning rate, network size 조정
   -> 낮음 (<0.01): Dynamics OK, 다음 단계로

2. SAC/alpha 확인
   -> 0 근접: Policy collapse -> alpha lower bound 추가 또는 target_entropy 조정
   -> >5: 과잉 탐색 -> 정상 범위로 내려올 때까지 대기

3. Encoder/grad_norm 확인
   -> 0.0: MPC -> Encoder gradient chain 끊김
     -> MPC differentiable solve (create_graph=True) 확인
     -> cost_map의 z 의존성 확인
   -> Non-zero: Encoder OK

4. MPC/solve_cost 확인
   -> NaN 또는 Inf: MPC solver 수치 불안정 -> PGD step size 축소
   -> 높지만 안정적: MPC converge 부족 -> PGD iterations 증가
```

### Reward가 오르다 갑자기 하락할 때

```
1. SAC/alpha 급변 확인
   -> 급락: Entropy collapse -> alpha learning rate 줄이기
   -> 급상승: 정상 (exploration burst), 보통 복구됨

2. Loss/critic 급등 확인
   -> 급등: Q value overestimation
     -> target update rate (tau) 줄이기
     -> twin Q clipping 확인

3. DR parameters 변화 확인
   -> DR curriculum이 급격히 강화되면 일시적 하락 정상
```

---

## 5. Metric Implementation Checklist

SAC-MPC 코드 구현 시, 아래 메트릭들이 올바른 위치에서 emit되는지 확인할 것.

### `runners/sac.py` (SAC update method)

```python
# Must emit after each gradient update:
metrics = {
    "Loss/actor": actor_loss.item(),
    "Loss/critic": critic_loss.item(),
    "Loss/alpha": alpha_loss.item(),
    "Loss/dynamics": dynamics_loss.item(),
    "SAC/alpha": alpha.item(),
    "SAC/q1_mean": q1.mean().item(),
    "SAC/q2_mean": q2.mean().item(),
    "SAC/target_q_mean": target_q.mean().item(),
    "SAC/log_prob_mean": log_prob.mean().item(),
    "SAC/actor_grad_norm": actor_grad_norm,
    "SAC/critic_grad_norm": critic_grad_norm,
}
```

### `runners/sac_mpc_runner.py` (Training loop)

```python
# Must emit per logging interval:
metrics = {
    "Train/fps": fps,
    "Train/buffer_size": buffer.size,
    "Train/terminated": terminated_count,
    "Train/time_out": timeout_count,
}
```

### `controllers/dynamics_mlp.py` or `sac.py` (Dynamics update)

```python
# Must emit after dynamics training step:
metrics = {
    "Dynamics/pred_err_mean": pred_error.mean().item(),
    "Dynamics/grad_norm": dynamics_grad_norm,
}
```

### `encoder/actor_critic_mpc.py` (MPC forward)

```python
# Must emit per MPC solve:
metrics = {
    "MPC/Q_diag_mean": Q_diag.mean().item(),
    "MPC/R_diag_mean": R_diag.mean().item(),
    "MPC/solve_cost": final_cost.mean().item(),
    "MPC/state_err_total": state_err.sum(-1).mean().item(),
}
```

### `base_env.py` + `utils/logging.py` (Already implemented)

아래 메트릭들은 기존 인프라에서 자동으로 emit됨:
- `Episode_Reward/*`: `base_env._collect_episode_metrics()`
- `Episode_Termination/*`: `base_env._collect_episode_metrics()`
- `Attitude_Error/*`: `base_env._collect_episode_metrics()`
- `Action/*`: `base_env._collect_episode_metrics()`
- `Encoder/*`: `logging.log_encoder_metrics()`
- `TDC/*`: `logging.log_tdc_diagnostics()`
- `DR/*`: `logging.log_dr_metrics()`

---

## 6. WandB Dashboard Setup

수동으로 패널을 구성하는 대신, `scripts/create_wandb_dashboard.py` 스크립트로
8-panel 레이아웃을 자동 생성할 수 있다.

```bash
cd /workspace/isaaclab
python scripts/create_wandb_dashboard.py \
    --project hero_agent \
    --run-id <RUN_ID>
```

자세한 사용법은 스크립트의 `--help` 참조.
