# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark CLI for systematic evaluation of Hero Agent controllers.

Evaluates one or more Hero Agent models across controlled benchmark scenarios
and outputs comparison tables, CSVs, and optional WandB logs.

Usage:
    ./isaaclab.sh -p .../hero_agent/workflows/benchmark.py \
        --entries "Isaac-HeroAgent-TDC-v0:" \
                  "Isaac-HeroAgent-Base-v0:path/to/model_600.pt" \
        --scenarios nominal hard \
        --num_episodes 50 \
        --num_envs 256 \
        --output_dir benchmarks/run1 \
        --headless

Entry format: "TASK_ID:CHECKPOINT_PATH" (empty path for TDC-v0 which needs no checkpoint).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark Hero Agent controllers across scenarios.")
parser.add_argument(
    "--entries",
    nargs="+",
    required=True,
    help='Model entries as "TASK_ID:CHECKPOINT_PATH" (empty path for no-checkpoint tasks like TDC-v0).',
)
parser.add_argument(
    "--scenarios",
    nargs="+",
    default=["nominal", "easy", "hard", "extreme"],
    help="Scenario names to evaluate (default: all four).",
)
parser.add_argument("--num_episodes", type=int, default=50, help="Episodes to collect per (model, scenario) pair.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--output_dir", type=str, default="benchmarks/latest", help="Output directory for CSVs.")
parser.add_argument("--logger", type=str, default=None, choices=["wandb"], help="Optional WandB logging.")
parser.add_argument("--log_project_name", type=str, default="hero_agent_benchmark", help="WandB project name.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.hero_agent.config_benchmark import get_benchmark_scenarios
from isaaclab_tasks.direct.hero_agent.workflows.benchmark_runner import (
    BenchmarkAggregator,
    BenchmarkEntry,
    BenchmarkRunner,
)


def _parse_entries(raw_entries: list[str]) -> list[BenchmarkEntry]:
    """Parse "TASK_ID:CHECKPOINT_PATH" strings into BenchmarkEntry objects."""
    entries = []
    for raw in raw_entries:
        if ":" not in raw:
            raise ValueError(
                f"Invalid entry format: '{raw}'. Expected 'TASK_ID:CHECKPOINT_PATH' "
                f"(use empty path for no-checkpoint: 'Isaac-HeroAgent-TDC-v0:')."
            )
        task_id, ckpt = raw.split(":", 1)
        checkpoint_path = ckpt.strip() if ckpt.strip() else None
        entries.append(BenchmarkEntry(task_id=task_id.strip(), checkpoint_path=checkpoint_path))
    return entries


def main():
    """Run benchmark evaluation."""
    entries = _parse_entries(args_cli.entries)
    all_scenarios = get_benchmark_scenarios()

    # Filter to requested scenarios
    scenario_names = args_cli.scenarios
    scenarios = {}
    for name in scenario_names:
        if name not in all_scenarios:
            raise ValueError(f"Unknown scenario '{name}'. Available: {list(all_scenarios.keys())}")
        scenarios[name] = all_scenarios[name]

    # Setup WandB if requested
    if args_cli.logger == "wandb":
        try:
            import wandb

            wandb.init(
                project=args_cli.log_project_name,
                config={
                    "entries": [e.label for e in entries],
                    "scenarios": scenario_names,
                    "num_episodes": args_cli.num_episodes,
                    "num_envs": args_cli.num_envs,
                    "seed": args_cli.seed,
                },
            )
        except ImportError:
            print("[WARN] wandb not available, skipping.")

    runner = BenchmarkRunner(
        num_envs=args_cli.num_envs,
        device=args_cli.device if args_cli.device else "cuda:0",
        seed=args_cli.seed,
    )
    aggregator = BenchmarkAggregator()

    # Evaluate each (entry, scenario) pair
    total = len(entries) * len(scenarios)
    current = 0
    for entry in entries:
        for sc_name, scenario in scenarios.items():
            current += 1
            print(f"\n[Benchmark {current}/{total}] {entry.label} x {sc_name}")
            print(f"  Scenario: {scenario.description}")
            if entry.checkpoint_path:
                print(f"  Checkpoint: {entry.checkpoint_path}")

            results = runner.evaluate(entry, scenario, args_cli.num_episodes)
            aggregator.add(entry.label, sc_name, results)

            # Print interim stats
            if results:
                errors = [r.mean_attitude_error_deg for r in results]
                mean_err = sum(errors) / len(errors)
                print(f"  -> {len(results)} episodes, mean error: {mean_err:.2f} deg")

    # Output results
    output_dir = os.path.abspath(args_cli.output_dir)
    ep_csv, summary_csv = aggregator.to_csv(output_dir)
    print(f"\n[Benchmark] Per-episode results: {ep_csv}")
    print(f"[Benchmark] Summary results: {summary_csv}")

    aggregator.print_comparison_table()

    # WandB table logging
    if args_cli.logger == "wandb":
        try:
            import wandb

            summaries = aggregator.summarize()
            table = wandb.Table(
                columns=["model", "scenario", "mean_error_deg", "std_error_deg", "convergence_rate", "mean_reward"]
            )
            for s in summaries:
                table.add_data(
                    s.label, s.scenario, s.mean_error_deg, s.std_error_deg, s.convergence_rate, s.mean_reward
                )
            wandb.log({"benchmark_summary": table})
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
    simulation_app.close()
