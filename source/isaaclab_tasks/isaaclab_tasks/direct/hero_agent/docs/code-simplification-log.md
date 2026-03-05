# Code Simplification Log

Hero Agent codebase (~7,700 lines, 27 Python files) simplification 진행 기록.

## Scope

- Dead code 제거 + 코드 중복 통합 + verbose 메서드 간소화 + 미사용 reward 함수 제거
- `encoder_tdc_env.py`: 유지 (참조 코드)
- 아키텍처 변경 (AdaptRunner 상속 등): 하지 않음

## Progress

### Step 1: Dead Code & Stale References 정리 -- DONE (2026-03-05)

Changes:
- `base_env.py`: `_cumulative_effort` 버퍼 제거 (초기화 L396, 업데이트 L831, 리셋 L1086)
- `base_env.py`: `HeroAgentEnvWindow` 클래스 제거 + `BaseEnvWindow` import 제거
- `controllers/__init__.py`: MPC docstring 참조 제거
- `encoder/__init__.py`: MPC docstring 참조 제거
- `runners/__init__.py`: MPC docstring 참조 제거
- `__pycache__` 8개 디렉토리 삭제

Verification: `ruff check` passed

### Step 2: 미사용 Reward 함수 제거 -- DONE (2026-03-05)

Changes:
- `mdp/rewards.py`: `action_rate_penalty()`, `angular_velocity_penalty()` 함수 삭제 (42 lines)
- `mdp/rewards.py`: ALBCRewardCfg에서 3 필드 + docstring 삭제: `action_rate_weight`, `termination_penalty`, `angular_velocity_weight`
- `mdp/__init__.py`: import 및 `__all__` export 제거
- `base_env.py`: import 제거, `_build_reward_terms()`에서 action_rate/angular_velocity term 빌드 블록 제거, `_get_rewards()`에서 termination_penalty 적용 코드 제거

Verification: `ruff check` passed, `grep` confirms no remaining references in code (docs stale refs -> Step 6)

### Step 3: Perturbation Update 중복 코드 통합 -- DONE (2026-03-05)

Changes:
- `base_env.py`: `_update_perturbation()` main body / buoy 동일 로직을 `_apply_perturbation_cycle()` helper로 통합

Verification: `ruff check` passed

### Step 4: Noise Config 중복 루프 통합 -- DONE (2026-03-05)

Changes:
- `base_env.py`: `_iter_noise_params()` static method 추가 (공통 iterator)
- `base_env.py`: `_pad_noise_cfg_for_tde()` 간소화 (4중 nested loop -> 1줄 for loop)
- `base_env.py`: `_convert_noise_cfg_tuples()` 간소화 (4중 nested loop -> 1줄 for loop)
- `config.py`: observation_noise_model 튜플을 `[val] * N` 패턴으로 가독성 향상

Verification: `ruff check` passed

### Step 5: DR Factory -- CoB/CoG DORAEMON 분기 통합 -- DONE (2026-03-05)

Changes:
- `mdp/events.py`: `_apply_xyz_offset_with_doraemon()` helper 추가 (XY uniform + Z DORAEMON override)
- `mdp/events.py`: CoB/CoG offset 각각의 DORAEMON 분기 (~16줄 x2) -> 2 helper 호출로 교체
- `mdp/events.py`: 미사용 `_apply_xyz_offset()` 함수 삭제 (26줄)

Verification: `ruff check` passed

### Step 6: 최종 정리 & 문서 업데이트 -- DONE (2026-03-05)

Changes:
- `ruff check` + `ruff format` 전체 hero_agent 디렉토리 통과
- 미사용 import 없음 확인 (F401 clean)
- 이 로그 파일 최종 업데이트

### Step 7: 500줄+ 파일 코드 리뷰 (config.py, tdc.py) -- DONE (2026-03-05)

config.py (569줄, 3건 micro-fix):
- L20: stale MPC docstring 참조 제거 (`hero_agent_mpc/` 삭제됨)
- L356-359: `HeroAgentTrainEnvCfg.ocean_current` 중복 override 제거 (부모 `HeroAgentEnvCfg`와 동일 값)
- L531: `HeroAgentEncoderTDCEnvCfg.enable_payload` 중복 override 제거 (상속으로 충분)

tdc.py (530줄, 2건 리팩터링):
- `_set_param()` static helper 추출 — `update_controller_params()`, `update_gains()`의 if/else 중복 제거
- `_zero_buffers` 리스트 + loop 기반 `reset()` — 11개 버퍼 개별 초기화를 2줄로 통합

Code Simplifier 리뷰 (3 parallel agents):
- Code Reuse: 이번 diff 범위 내 신규 중복 없음
- Code Quality: pre-existing 이슈만 (stringly-typed config 등)
- Efficiency: double `_compute_M_true` 호출은 pre-existing (리팩터링 전에도 inline으로 존재). 별도 최적화 task로 분리

Verification: `ruff check && ruff format` clean
