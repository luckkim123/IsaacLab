# ALBC Task Integration Design

## Overview

Integrate `ALBCAttitudeTask` class into `HeroAgentEnv` to simplify the codebase.

## Motivation

1. Unnecessary complexity from separate file/class
2. No future plan to add other tasks
3. Difficult to understand code flow across two files

## Changes

### Files to Delete
- `tasks/albc_attitude_task.py`
- `tasks/__init__.py`
- `tasks/__pycache__/`

### Files to Modify

#### 1. `hero_agent_env_cfg.py`
- Remove `from .tasks import ALBCAttitudeTaskCfg`
- Remove `task: ALBCAttitudeTaskCfg` field
- Add inline fields: `target_attitude`, `randomize_target_attitude`, `target_attitude_range`

#### 2. `hero_agent_env.py`
- Remove `from .tasks import ALBCAttitudeTask`
- Add `euler_xyz_from_quat` import
- Add attitude buffers to `_init_state_buffers()`
- Add methods: `_compute_attitude_error()`, `_get_attitude_error()`, `_update_potentials()`, `_reset_attitude_task()`, `_initialize_potentials()`
- Remove `_task` initialization
- Update all `self._task.xxx()` calls to direct methods

#### 3. `mdp/observations.py`
- Change `env._task.get_goal_observations(robot)` to `env._get_attitude_error()`

#### 4. `mdp/rewards.py`
- Remove `from ..tasks import ALBCAttitudeTask` TYPE_CHECKING import
- Change `task: ALBCAttitudeTask` parameter to use `env` directly
- Access `env._potentials` and `env._prev_potentials` instead of `task._potentials`

## Implementation Order

1. Config integration
2. Buffer and method integration
3. observations.py update
4. rewards.py update
5. Delete tasks directory
6. Test
