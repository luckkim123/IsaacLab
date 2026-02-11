# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark evaluation engine for Hero Agent controllers.

Provides BenchmarkRunner for systematic evaluation of multiple Hero Agent models
across controlled scenarios, and BenchmarkAggregator for result summarization.

Supports all 5 task types with appropriate model loading:
    - Base-v0, Encoder-Base-v0: OnPolicyRunner + standard inference
    - TDC-v0: No checkpoint (classical controller, zero RL actions)
    - Encoder-TDC-v0: OnPolicyRunner + encoder wire
    - Adapt-TDC-v0: build_adapt_policy + Phase 2 checkpoint
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

from ..config_benchmark import BenchmarkScenario
from ..utils.env_utils import connect_encoder_to_env, unwrap_env
from . import apply_cli_overrides, resolve_task_configs

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEntry:
    """A model to evaluate: task ID + optional checkpoint."""

    task_id: str
    checkpoint_path: str | None = None  # None for TDC-v0
    label: str = ""

    def __post_init__(self):
        if not self.label:
            # Derive label from task_id: "Isaac-HeroAgent-TDC-v0" -> "TDC"
            parts = self.task_id.replace("Isaac-HeroAgent-", "").replace("-v0", "")
            self.label = parts or self.task_id


@dataclass
class EpisodeResult:
    """Metrics for a single completed episode."""

    mean_attitude_error_deg: float
    roll_error_deg: float
    pitch_error_deg: float
    attitude_error_std_deg: float  # temporal stability within episode
    time_to_1deg_s: float | None  # convergence speed (None if never reached)
    episode_reward: float
    episode_length_s: float
    timed_out: bool
    action_rms: float
    # Control standard metrics
    settling_time_5deg_s: float | None = None  # time to reach and stay within 5 deg
    overshoot_deg: float = 0.0  # max error after first crossing below target
    steady_state_error_deg: float = 0.0  # mean error over last 20% of episode
    # DR snapshot (captured at episode start, associated at episode end)
    dr_payload_mass: float | None = None
    dr_buoyancy_force: float | None = None
    dr_water_density: float | None = None
    dr_current_magnitude: float | None = None
    dr_joint_stiffness: float | None = None


@dataclass
class ScenarioSummary:
    """Aggregated metrics for one (model, scenario) combination."""

    label: str
    scenario: str
    num_episodes: int
    # Attitude error (primary metric)
    mean_error_deg: float
    std_error_deg: float
    median_error_deg: float
    p95_error_deg: float
    # Convergence
    mean_time_to_1deg_s: float | None
    convergence_rate: float  # fraction of episodes reaching < 1 deg
    # Reward
    mean_reward: float
    # Stability
    mean_error_std_deg: float  # average temporal std within episodes
    # Action effort
    mean_action_rms: float
    # Control standard metrics
    mean_settling_time_5deg_s: float | None = None
    mean_overshoot_deg: float = 0.0
    mean_steady_state_error_deg: float = 0.0


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Evaluate a single (entry, scenario) pair by running episodes in Isaac Lab."""

    def __init__(self, num_envs: int = 256, device: str = "cuda:0", seed: int = 42):
        self.num_envs = num_envs
        self.device = device
        self.seed = seed

    def evaluate(
        self,
        entry: BenchmarkEntry,
        scenario: BenchmarkScenario,
        num_episodes: int,
    ) -> list[EpisodeResult]:
        """Run num_episodes and collect EpisodeResult for each.

        Args:
            entry: Model entry (task_id + checkpoint).
            scenario: Benchmark scenario with DR/current/payload settings.
            num_episodes: Total episodes to collect across all envs.

        Returns:
            List of EpisodeResult, one per completed episode.
        """
        # --- Resolve and override config ---
        env_cfg, agent_cfg = resolve_task_configs(entry.task_id)
        apply_cli_overrides(env_cfg, self.num_envs, self.seed, self.device)

        # Apply scenario overrides
        _apply_scenario(env_cfg, scenario)

        # Disable debug visualization for benchmarking
        env_cfg.debug_vis = False

        # --- Create environment ---
        env = gym.make(entry.task_id, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        raw_env = unwrap_env(env)

        # --- Load policy ---
        policy_fn, cleanup_fn = _load_policy(entry, env, agent_cfg, self.device)

        # --- Metric collection ---
        step_dt = raw_env.step_dt
        results: list[EpisodeResult] = []

        # Per-env accumulators (reset on episode boundary)
        error_accum: list[list[float]] = [[] for _ in range(self.num_envs)]
        reward_accum = torch.zeros(self.num_envs, device=self.device)
        action_sq_accum = torch.zeros(self.num_envs, device=self.device)
        step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Track first time error < 1 deg
        converged_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        # DR snapshot buffer (captured after each reset, used at episode end)
        dr_snapshot: dict[str, list[float | None]] = {
            "payload_mass": [None] * self.num_envs,
            "buoyancy_force": [None] * self.num_envs,
            "water_density": [None] * self.num_envs,
            "current_magnitude": [None] * self.num_envs,
            "joint_stiffness": [None] * self.num_envs,
        }
        _capture_dr_snapshot(raw_env, dr_snapshot, list(range(self.num_envs)))

        # Safety: max steps to prevent infinite loop
        max_ep_len = int(raw_env.max_episode_length) if hasattr(raw_env, "max_episode_length") else 3000
        max_steps = (num_episodes // max(self.num_envs, 1) + 2) * max_ep_len

        obs = env.get_observations()
        global_step = 0

        with torch.inference_mode():
            while len(results) < num_episodes and global_step < max_steps:
                global_step += 1

                # Act
                actions = policy_fn(obs)

                # Step
                obs, rewards, dones, infos = env.step(actions)

                # Read attitude error from env internals
                att_err = raw_env._attitude_error[:, :2].abs()  # (num_envs, 2) radians
                att_err_deg = torch.rad2deg(att_err)
                total_err_deg = att_err_deg.sum(dim=-1)  # (num_envs,)

                # Accumulate per-env
                for i in range(self.num_envs):
                    error_accum[i].append(total_err_deg[i].item())
                reward_accum += rewards.squeeze(-1) if rewards.dim() > 1 else rewards
                action_sq_accum += (actions**2).mean(dim=-1)
                step_count += 1

                # Check convergence (first time total error < 1 deg)
                newly_converged = (total_err_deg < 1.0) & (converged_step < 0)
                converged_step[newly_converged] = step_count[newly_converged]

                # Process completed episodes
                done_mask = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
                if done_mask.any():
                    done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

                    for idx in done_ids:
                        i = idx.item()
                        if len(results) >= num_episodes:
                            break

                        ep_steps = step_count[i].item()
                        if ep_steps == 0:
                            continue

                        # Gather per-step errors for this specific env
                        ep_errors = torch.tensor(error_accum[i], dtype=torch.float32)

                        # Time to 1 deg
                        conv_step = converged_step[i].item()
                        time_to_1deg = conv_step * step_dt if conv_step >= 0 else None

                        # Control standard metrics
                        settling_time, overshoot, steady_state = _compute_control_metrics(
                            ep_errors, step_dt, threshold_deg=5.0
                        )

                        # Check timeout from env infos
                        timed_out = False
                        if "time_outs" in infos:
                            to = infos["time_outs"]
                            timed_out = bool(to[i].item()) if hasattr(to, "__getitem__") else False

                        results.append(
                            EpisodeResult(
                                mean_attitude_error_deg=ep_errors.mean().item(),
                                roll_error_deg=torch.rad2deg(raw_env._attitude_error[i, 0].abs()).item(),
                                pitch_error_deg=torch.rad2deg(raw_env._attitude_error[i, 1].abs()).item(),
                                attitude_error_std_deg=ep_errors.std().item() if ep_steps > 1 else 0.0,
                                time_to_1deg_s=time_to_1deg,
                                episode_reward=reward_accum[i].item(),
                                episode_length_s=ep_steps * step_dt,
                                timed_out=timed_out,
                                action_rms=(action_sq_accum[i] / ep_steps).sqrt().item(),
                                settling_time_5deg_s=settling_time,
                                overshoot_deg=overshoot,
                                steady_state_error_deg=steady_state,
                                dr_payload_mass=dr_snapshot["payload_mass"][i],
                                dr_buoyancy_force=dr_snapshot["buoyancy_force"][i],
                                dr_water_density=dr_snapshot["water_density"][i],
                                dr_current_magnitude=dr_snapshot["current_magnitude"][i],
                                dr_joint_stiffness=dr_snapshot["joint_stiffness"][i],
                            )
                        )

                    # Reset accumulators for done envs
                    done_id_list = [idx.item() for idx in done_ids]
                    for ii in done_id_list:
                        error_accum[ii] = []
                    reward_accum[done_ids] = 0.0
                    action_sq_accum[done_ids] = 0.0
                    step_count[done_ids] = 0
                    converged_step[done_ids] = -1

                    # Recapture DR snapshot for newly reset envs
                    _capture_dr_snapshot(raw_env, dr_snapshot, done_id_list)

                    # Reset policy hidden states
                    if cleanup_fn is not None:
                        cleanup_fn(dones)

        if global_step >= max_steps and len(results) < num_episodes:
            print(f"[WARN] Benchmark hit step limit ({max_steps}), collected {len(results)}/{num_episodes} episodes")

        env.close()
        return results[:num_episodes]


# ---------------------------------------------------------------------------
# BenchmarkAggregator
# ---------------------------------------------------------------------------


class BenchmarkAggregator:
    """Aggregate and output benchmark results."""

    def __init__(self):
        self._all_results: list[tuple[str, str, list[EpisodeResult]]] = []

    def add(self, label: str, scenario: str, results: list[EpisodeResult]) -> None:
        """Add results for one (model, scenario) pair."""
        self._all_results.append((label, scenario, results))

    def summarize(self) -> list[ScenarioSummary]:
        """Compute summary statistics for all collected results."""
        summaries = []
        for label, scenario, results in self._all_results:
            if not results:
                continue

            errors = torch.tensor([r.mean_attitude_error_deg for r in results])
            convergence_times = [r.time_to_1deg_s for r in results if r.time_to_1deg_s is not None]
            converge_rate = len(convergence_times) / len(results) if results else 0.0

            # Control standard metrics
            settling_times = [r.settling_time_5deg_s for r in results if r.settling_time_5deg_s is not None]
            mean_settling = sum(settling_times) / len(settling_times) if settling_times else None
            mean_overshoot = sum(r.overshoot_deg for r in results) / len(results)
            mean_ss_error = sum(r.steady_state_error_deg for r in results) / len(results)

            summaries.append(
                ScenarioSummary(
                    label=label,
                    scenario=scenario,
                    num_episodes=len(results),
                    mean_error_deg=errors.mean().item(),
                    std_error_deg=errors.std().item() if len(errors) > 1 else 0.0,
                    median_error_deg=errors.median().item(),
                    p95_error_deg=errors.quantile(0.95).item() if len(errors) >= 2 else errors.max().item(),
                    mean_time_to_1deg_s=(
                        sum(convergence_times) / len(convergence_times) if convergence_times else None
                    ),
                    convergence_rate=converge_rate,
                    mean_reward=sum(r.episode_reward for r in results) / len(results),
                    mean_error_std_deg=sum(r.attitude_error_std_deg for r in results) / len(results),
                    mean_action_rms=sum(r.action_rms for r in results) / len(results),
                    mean_settling_time_5deg_s=mean_settling,
                    mean_overshoot_deg=mean_overshoot,
                    mean_steady_state_error_deg=mean_ss_error,
                )
            )
        return summaries

    def to_csv(self, output_dir: str) -> tuple[str, str]:
        """Write per-episode and summary CSVs.

        Args:
            output_dir: Directory for output files.

        Returns:
            Tuple of (per_episode_csv_path, summary_csv_path).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Per-episode CSV
        ep_path = os.path.join(output_dir, "results.csv")
        with open(ep_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "scenario", "episode"] + list(EpisodeResult.__dataclass_fields__.keys()))
            for label, scenario, results in self._all_results:
                for i, r in enumerate(results):
                    writer.writerow([label, scenario, i] + list(asdict(r).values()))

        # Summary CSV
        summaries = self.summarize()
        summary_path = os.path.join(output_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(ScenarioSummary.__dataclass_fields__.keys()))
            for s in summaries:
                writer.writerow(list(asdict(s).values()))

        return ep_path, summary_path

    def print_comparison_table(self) -> None:
        """Print a formatted comparison table to console."""
        summaries = self.summarize()
        if not summaries:
            print("[Benchmark] No results to display.")
            return

        # Group by scenario
        scenarios = sorted(set(s.scenario for s in summaries))
        labels = sorted(set(s.label for s in summaries))

        # Header
        col_w = 14
        header = f"{'Model':<20}"
        for sc in scenarios:
            header += f"| {sc:^{col_w}} "
        print("\n" + "=" * len(header))
        print("  Attitude Error (|roll| + |pitch|) in degrees")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Build lookup
        lookup = {(s.label, s.scenario): s for s in summaries}

        for label in labels:
            row = f"{label:<20}"
            for sc in scenarios:
                s = lookup.get((label, sc))
                if s:
                    cell = f"{s.mean_error_deg:.2f}+/-{s.std_error_deg:.2f}"
                    row += f"| {cell:^{col_w}} "
                else:
                    row += f"| {'N/A':^{col_w}} "
            print(row)

        # Convergence rate row
        print("-" * len(header))
        print(f"{'Conv. Rate (<1deg)':<20}", end="")
        for sc in scenarios:
            rates = []
            for label in labels:
                s = lookup.get((label, sc))
                if s:
                    rates.append(f"{s.convergence_rate:.0%}")
                else:
                    rates.append("N/A")
            # Show first model's rate as example
            cell = " / ".join(rates)
            if len(cell) > col_w:
                # Fall back to per-row display
                break
        else:
            print()

        # Detailed convergence per model
        print()
        for label in labels:
            row = f"{'  ' + label + ' conv%':<20}"
            for sc in scenarios:
                s = lookup.get((label, sc))
                if s:
                    row += f"| {s.convergence_rate:^{col_w}.0%} "
                else:
                    row += f"| {'N/A':^{col_w}} "
            print(row)

        # Settling time per model
        print()
        for label in labels:
            row = f"{'  ' + label + ' settle':<20}"
            for sc in scenarios:
                s = lookup.get((label, sc))
                if s and s.mean_settling_time_5deg_s is not None:
                    cell = f"{s.mean_settling_time_5deg_s:.2f}s"
                    row += f"| {cell:^{col_w}} "
                else:
                    row += f"| {'N/A':^{col_w}} "
            print(row)

        # Overshoot per model
        for label in labels:
            row = f"{'  ' + label + ' ovrsht':<20}"
            for sc in scenarios:
                s = lookup.get((label, sc))
                if s:
                    cell = f"{s.mean_overshoot_deg:.2f}deg"
                    row += f"| {cell:^{col_w}} "
                else:
                    row += f"| {'N/A':^{col_w}} "
            print(row)

        # Steady-state error per model
        for label in labels:
            row = f"{'  ' + label + ' ss_err':<20}"
            for sc in scenarios:
                s = lookup.get((label, sc))
                if s:
                    cell = f"{s.mean_steady_state_error_deg:.2f}deg"
                    row += f"| {cell:^{col_w}} "
                else:
                    row += f"| {'N/A':^{col_w}} "
            print(row)

        print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_scenario(env_cfg: object, scenario: BenchmarkScenario) -> None:
    """Override env_cfg DR, ocean current, and payload from scenario."""
    env_cfg.randomization = scenario.dr_cfg
    env_cfg.ocean_current = scenario.ocean_current_cfg
    env_cfg.enable_payload = scenario.enable_payload

    # TDC envs need joint gain ranges from scenario DR
    if hasattr(env_cfg, "tdc"):
        # Preserve TDC config but update joint gains to match scenario
        pass


def _load_policy(
    entry: BenchmarkEntry,
    env: RslRlVecEnvWrapper,
    agent_cfg: object,
    device: str,
) -> tuple[callable, callable | None]:
    """Load the appropriate policy for the given entry.

    Returns:
        Tuple of (policy_fn, reset_fn).
        policy_fn: obs -> actions
        reset_fn: dones -> None (resets hidden states), or None
    """
    task_id = entry.task_id

    # TDC-v0 (pure classical control): no checkpoint, zero actions (TDC runs internally).
    # Must NOT match Encoder-TDC or Adapt-TDC which need trained policies.
    is_pure_tdc = "TDC" in task_id and "Encoder" not in task_id and "Adapt" not in task_id
    if is_pure_tdc and entry.checkpoint_path is None:
        num_envs = env.unwrapped.num_envs
        action_dim = env.unwrapped.cfg.action_space
        zero_actions = torch.zeros(num_envs, action_dim, device=device)
        return lambda _obs: zero_actions, None

    # Adapt-TDC-v0: special Phase 2 loading
    if "Adapt-TDC" in task_id:
        return _load_adapt_policy(entry, env, agent_cfg, device)

    # Standard RSL-RL: OnPolicyRunner
    return _load_rsl_rl_policy(entry, env, agent_cfg, device)


def _load_rsl_rl_policy(
    entry: BenchmarkEntry,
    env: RslRlVecEnvWrapper,
    agent_cfg: object,
    device: str,
) -> tuple[callable, callable | None]:
    """Load policy via OnPolicyRunner (Base, Encoder-Base, Encoder-TDC)."""
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(entry.checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Extract neural network for reset
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Wire encoder if applicable
    if hasattr(policy_nn, "get_last_z"):
        connect_encoder_to_env(env, policy_nn, caller_name="Benchmark")

    def reset_fn(dones):
        policy_nn.reset(dones)

    return policy, reset_fn


def _load_adapt_policy(
    entry: BenchmarkEntry,
    env: RslRlVecEnvWrapper,
    agent_cfg: object,
    device: str,
) -> tuple[callable, callable | None]:
    """Load Phase 2 adaptation policy (Adapt-TDC)."""
    from . import build_adapt_policy, get_proprio_history_shape, load_phase2_checkpoint

    obs_dict = env.get_observations()
    num_actions = env.unwrapped.cfg.action_space
    policy = build_adapt_policy(obs_dict, agent_cfg, num_actions)

    proprio_history_shape = get_proprio_history_shape(env.unwrapped.cfg)
    hist_normalizer, _ = load_phase2_checkpoint(
        policy,
        entry.checkpoint_path,
        device,
        proprio_history_shape,
    )

    connect_encoder_to_env(env, policy, caller_name="Benchmark")

    def policy_fn(obs):
        obs_normalized = dict(obs)
        if "proprio_hist" in obs:
            obs_normalized["proprio_hist"] = hist_normalizer(obs["proprio_hist"])
        actions = policy.act_inference(obs_normalized)
        return actions.clamp(-1.0, 1.0)

    def reset_fn(dones):
        policy.reset(dones)

    return policy_fn, reset_fn


def _compute_control_metrics(
    ep_errors: torch.Tensor,
    step_dt: float,
    threshold_deg: float = 5.0,
) -> tuple[float | None, float, float]:
    """Compute settling time, overshoot, and steady-state error from error trace.

    Args:
        ep_errors: Per-step total attitude error in degrees. Shape: (T,).
        step_dt: Timestep in seconds.
        threshold_deg: Settling threshold in degrees.

    Returns:
        Tuple of (settling_time_s, overshoot_deg, steady_state_error_deg).
    """
    if len(ep_errors) == 0:
        return None, 0.0, 0.0

    below = ep_errors < threshold_deg

    # Settling time: last step where error exceeded threshold, then +1
    if below.all():
        settling_time = 0.0
    else:
        last_violation = (~below).nonzero(as_tuple=False)[-1].item()
        if last_violation < len(ep_errors) - 1:
            settling_time = (last_violation + 1) * step_dt
        else:
            settling_time = None  # never settled

    # Overshoot: max error after first crossing below threshold
    if below.any():
        first_below = below.nonzero(as_tuple=False)[0].item()
        overshoot = ep_errors[first_below:].max().item()
    else:
        overshoot = ep_errors.max().item()

    # Steady-state: mean of last 20% of episode
    tail_start = int(0.8 * len(ep_errors))
    steady_state = ep_errors[tail_start:].mean().item()

    return settling_time, overshoot, steady_state


def _capture_dr_snapshot(
    raw_env: object,
    dr_snapshot: dict[str, list[float | None]],
    env_ids: list[int],
) -> None:
    """Capture current DR parameters for specified environments.

    Reads from env internals (hydro model, payload, joints). Values are stored
    in per-env lists so they can be associated with episodes at completion time.

    Args:
        raw_env: Unwrapped HeroAgentEnv instance.
        dr_snapshot: Dict of DR buffers to update.
        env_ids: Environment indices to capture.
    """
    for i in env_ids:
        if hasattr(raw_env, "_hydro"):
            dr_snapshot["buoyancy_force"][i] = raw_env._hydro.buoyancy_force[i].item()
            dr_snapshot["water_density"][i] = raw_env._hydro.water_density[i].item()
            if hasattr(raw_env._hydro, "_current_velocity") and raw_env._hydro._current_velocity is not None:
                dr_snapshot["current_magnitude"][i] = torch.linalg.norm(raw_env._hydro._current_velocity[i, :3]).item()
            else:
                dr_snapshot["current_magnitude"][i] = None
        if hasattr(raw_env, "_payload_mass") and raw_env._payload_mass is not None:
            dr_snapshot["payload_mass"][i] = raw_env._payload_mass[i].item()
        else:
            dr_snapshot["payload_mass"][i] = None
        if hasattr(raw_env, "_robot") and hasattr(raw_env, "_albc_joint_ids"):
            dr_snapshot["joint_stiffness"][i] = (
                raw_env._robot.data.joint_stiffness[i, raw_env._albc_joint_ids].mean().item()
            )
