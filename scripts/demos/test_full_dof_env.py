"""Verification script for velocity tracking ALBC environment.

Tests:
    1. Smoke test: env create, reset, step without crash
    2. Observation dimensions match config (81D unified policy, 23D privileged)
    3. Thruster produces expected motion
    4. Reward responds to velocity tracking errors
    5. Manipulability index correctness
    6. Velocity command resampling
    7. Constraints fire at expected thresholds

Usage:
    ./isaaclab.sh -p scripts/demos/test_full_dof_env.py --headless
"""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Velocity tracking env verification")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Must import after Isaac Sim is initialized
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app

    from isaaclab_tasks.direct.constrained_full_albc import ALBCEnv
    from isaaclab_tasks.direct.constrained_full_albc.config import ALBCEnvCfg

    cfg = ALBCEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.randomization.enable = False  # Deterministic for testing
    cfg.doraemon.enable = False

    import math
    import sys

    def log(msg: str):
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    log("[VelTrack] Creating environment...")
    env = ALBCEnv(cfg)
    num_envs = args.num_envs
    device = env.device
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" -- {detail}"
        log(msg)
        if condition:
            passed += 1
        else:
            failed += 1

    # =========================================================================
    # Test 1: Smoke Test
    # =========================================================================
    log("\n=== Test 1: Smoke Test ===")
    obs, info = env.reset()
    check("Reset succeeds", obs is not None)

    has_nan = False
    for i in range(200):
        actions = torch.randn(num_envs, 8, device=device).clamp(-1, 1)
        obs, reward, terminated, truncated, info = env.step(actions)
        if torch.isnan(obs["policy"]).any():
            has_nan = True
            break
    check("200 random steps without NaN", not has_nan)

    # =========================================================================
    # Test 2: Observation Dimensions
    # =========================================================================
    log("\n=== Test 2: Observation Dimensions ===")
    obs_dict = env._get_observations()
    policy_shape = obs_dict["policy"].shape
    check("Policy obs shape", policy_shape == (num_envs, 81), f"got {policy_shape}")

    if "privileged" in obs_dict:
        priv_shape = obs_dict["privileged"].shape
        check("Privileged obs shape", priv_shape == (num_envs, 23), f"got {priv_shape}")
    else:
        check("Privileged obs present", False, "key missing")

    # Verify no separate history key (merged into policy obs)
    check("No separate proprio_hist key", "proprio_hist" not in obs_dict)

    # =========================================================================
    # Test 3: Thruster -> Motion
    # =========================================================================
    log("\n=== Test 3: Thruster -> Motion ===")
    env.reset()

    # Record initial position
    initial_pos = env._robot.data.root_pos_w[:, :3].clone()

    # Apply forward thrust for 50 steps (T0-T3 configured for surge)
    actions = torch.zeros(num_envs, 8, device=device)
    actions[:, 2] = 0.5  # T0
    actions[:, 3] = -0.5  # T1
    actions[:, 4] = 0.5  # T2
    actions[:, 5] = -0.5  # T3
    for _ in range(50):
        env.step(actions)

    pos_change = (env._robot.data.root_pos_w[:, :3] - initial_pos).norm(dim=-1)
    check("Forward thrust produces motion", pos_change.mean() > 0.01, f"mean delta={pos_change.mean():.4f}m")

    # Vertical thrust
    env.reset()
    initial_z = env._robot.data.root_pos_w[:, 2].clone()
    actions = torch.zeros(num_envs, 8, device=device)
    actions[:, 6] = 0.8  # T4 up
    actions[:, 7] = 0.8  # T5 up
    for _ in range(50):
        env.step(actions)
    z_change = env._robot.data.root_pos_w[:, 2] - initial_z
    check("Vertical thrust increases Z", z_change.mean() > 0.01, f"mean dz={z_change.mean():.4f}m")

    # =========================================================================
    # Test 4: Reward Directionality (velocity tracking)
    # =========================================================================
    log("\n=== Test 4: Reward Directionality ===")
    env.reset()

    # Set vel_cmd to zero -> robot stationary -> vel_err near zero -> reward near 0
    env._vel_cmd_lin[:] = 0.0
    env._vel_cmd_ang[:] = 0.0
    reward_zero_cmd = env._get_rewards().clone()

    # Set vel_cmd to large value -> vel_err large -> reward should be worse
    env._vel_cmd_lin[:, 0] = 0.5  # 0.5 m/s surge command
    reward_large_cmd = env._get_rewards().clone()
    check(
        "Large vel_cmd decreases reward (vel_err increases)",
        (reward_zero_cmd > reward_large_cmd).all().item(),
        f"zero_cmd={reward_zero_cmd.mean():.4f}, large_cmd={reward_large_cmd.mean():.4f}",
    )

    # =========================================================================
    # Test 5: Manipulability Index
    # =========================================================================
    log("\n=== Test 5: Manipulability Index ===")
    env.reset()

    # At nominal position theta2 = pi/2, w should be maximal (~1.0)
    env._robot.data.joint_pos[:, env._albc_joint_ids[1]] = math.pi / 2
    env._update_manipulability()
    w_nominal = env._manipulability.mean().item()
    check("Manipulability at nominal (pi/2)", w_nominal > 0.9, f"w={w_nominal:.4f}")

    # At singularity theta2 ≈ 0, w should be near 0
    env._robot.data.joint_pos[:, env._albc_joint_ids[1]] = 0.01
    env._update_manipulability()
    w_singular = env._manipulability.mean().item()
    check("Manipulability at singularity (0)", w_singular < 0.15, f"w={w_singular:.4f}")

    # =========================================================================
    # Test 6: Velocity Command Resampling
    # =========================================================================
    log("\n=== Test 6: Command Resampling ===")
    env.reset()

    # Record initial command
    initial_cmd = env._vel_cmd_lin.clone()

    # Step for resample_steps - 1 (should NOT change)
    resample_steps = cfg.vel_cmd_resample_steps
    for _ in range(resample_steps - 1):
        actions = torch.zeros(num_envs, 8, device=device)
        env.step(actions)
    cmd_before = env._vel_cmd_lin.clone()
    check("Command unchanged before resample", torch.allclose(initial_cmd, cmd_before))

    # One more step triggers resampling
    env.step(torch.zeros(num_envs, 8, device=device))
    cmd_after = env._vel_cmd_lin.clone()
    # Commands should differ (with high probability since uniform sampling)
    check("Command resampled after N steps", not torch.allclose(initial_cmd, cmd_after),
          f"diff={torch.norm(initial_cmd - cmd_after):.4f}")

    # =========================================================================
    # Test 7: Constraint Firing (9 constraints: 5 prob + 4 avg)
    # =========================================================================
    log("\n=== Test 7: Constraint Firing ===")
    from isaaclab_tasks.direct.constrained_full_albc.mdp.constraints import compute_all_costs

    env.reset()
    costs = compute_all_costs(env._robot, env, env._constraints_cfg)
    check("Costs tensor shape", costs.shape == (num_envs, 9), f"got {costs.shape}")

    # Manipulability (index 8): at singularity (theta2 near 0), cost > 0
    env.reset()
    env._robot.data.joint_pos[:, env._albc_joint_ids[1]] = 0.01
    env._update_manipulability()
    costs = compute_all_costs(env._robot, env, env._constraints_cfg)
    manip_cost = costs[:, 8].mean().item()
    check("Manipulability cost at singularity", manip_cost > 0.1, f"cost={manip_cost:.4f}")

    # Thruster utilization (index 7): after full thrust commands
    env.reset()
    full_thrust_actions = torch.zeros(num_envs, 8, device=device)
    full_thrust_actions[:, 2:] = 1.0  # all thrusters at full command
    for _ in range(10):
        env.step(full_thrust_actions)
    costs = compute_all_costs(env._robot, env, env._constraints_cfg)
    thr_cost = costs[:, 7].mean().item()
    check("Thruster util cost at full thrust", thr_cost > 0.5, f"cost={thr_cost:.4f}")

    # Yaw rate (index 5): zero command -> cost should be 0
    env.reset()
    costs = compute_all_costs(env._robot, env, env._constraints_cfg)
    yaw_cost_zero = costs[:, 5].mean().item()
    check("Yaw rate cost at rest is ~0", yaw_cost_zero < 0.01, f"cost={yaw_cost_zero:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    total = passed + failed
    log(f"\n{'='*50}")
    log(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'='*50}")

    env.close()
    simulation_app.close()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
