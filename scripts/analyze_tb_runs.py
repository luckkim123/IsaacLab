#!/usr/bin/env python3
"""Analyze TensorBoard event files for Hero Agent encoder training runs."""

import os
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Define runs
RUNS = {
    "Run4 (priv=26, 600it)": "/workspace/isaaclab/logs/rsl_rl/hero_agent_albc_encoder/2026-02-20_03-14-43_encoder-base/",
    "Run3 (priv=18)": "/workspace/isaaclab/logs/rsl_rl/hero_agent_albc_encoder/2026-02-20_03-04-44_encoder-base/",
    "Run2 (prev compare)": "/workspace/isaaclab/logs/rsl_rl/hero_agent_albc_encoder/2026-02-20_02-47-18_encoder-base/",
    "Run1 (hist best 600)": "/workspace/isaaclab/logs/rsl_rl/hero_agent_albc_encoder/2026-02-20_02-03-29_encoder-base/",
}

# Target metrics
TARGET_METRICS = [
    "Episode/mean_tracking_error_roll",
    "Episode/mean_tracking_error_pitch",
    "Episode_Reward/mean",
    "Loss/value_function",
    "Loss/surrogate_loss",
    "Train/mean_reward",
]

# Target iterations for comparison
TARGET_ITERS = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500]


def load_run(path):
    """Load a TensorBoard event file and return scalar data."""
    ea = EventAccumulator(path)
    ea.Reload()
    return ea


def get_scalar_data(ea, tag):
    """Get scalar data as list of (step, value) tuples."""
    try:
        events = ea.Scalars(tag)
        return [(e.step, e.value) for e in events]
    except KeyError:
        return []


def find_closest_value(data, target_step, tolerance=10):
    """Find value at the closest step to target_step within tolerance."""
    if not data:
        return None
    best = None
    best_dist = float("inf")
    for step, value in data:
        dist = abs(step - target_step)
        if dist < best_dist:
            best_dist = dist
            best = (step, value)
    if best and best_dist <= tolerance:
        return best
    return None


def main():
    # Load all runs
    accumulators = {}
    for name, path in RUNS.items():
        if not os.path.isdir(path):
            print(f"WARNING: {path} does not exist, skipping")
            continue
        print(f"Loading: {name}")
        print(f"  Path: {path}")
        accumulators[name] = load_run(path)

    print("\n" + "=" * 120)

    # Print available tags for each run
    print("\nAVAILABLE SCALAR TAGS PER RUN:")
    print("-" * 120)
    for name, ea in accumulators.items():
        tags = sorted(ea.Tags().get("scalars", []))
        print(f"\n  {name} ({len(tags)} tags):")
        for t in tags:
            data = get_scalar_data(ea, t)
            if data:
                steps = [d[0] for d in data]
                print(f"    {t:55s}  steps: {min(steps):5d} - {max(steps):5d}  ({len(data)} points)")

    # Extract data for target metrics
    print("\n" + "=" * 120)
    print("\nDETAILED METRICS PER RUN:")
    print("=" * 120)

    all_run_data = {}  # {run_name: {metric: [(step, value), ...]}}

    for name, ea in accumulators.items():
        print(f"\n{'=' * 120}")
        print(f"  {name}")
        print(f"{'=' * 120}")

        run_data = {}
        for metric in TARGET_METRICS:
            data = get_scalar_data(ea, metric)
            if not data:
                # Try alternative names
                alt_names = {
                    "Episode_Reward/mean": ["Episode Reward/mean", "Reward/mean", "Train/mean_reward"],
                    "Train/mean_reward": ["Train/mean reward", "Episode_Reward/mean"],
                }
                for alt in alt_names.get(metric, []):
                    data = get_scalar_data(ea, alt)
                    if data:
                        print(f"  Note: Using '{alt}' instead of '{metric}'")
                        break

            run_data[metric] = data

            if data:
                print(f"\n  {metric}:")
                print(f"  {'Iter':>6s}  {'Value':>12s}")
                print(f"  {'-'*6}  {'-'*12}")
                for target in TARGET_ITERS:
                    result = find_closest_value(data, target)
                    if result:
                        step, val = result
                        print(f"  {step:6d}  {val:12.6f}")
                # Also print last available point
                last_step, last_val = data[-1]
                if last_step > TARGET_ITERS[-1]:
                    print(f"  {last_step:6d}  {last_val:12.6f}  (last)")
                # Print min/max
                values = [v for _, v in data]
                min_val = min(values)
                max_val = max(values)
                min_step = [s for s, v in data if v == min_val][0]
                max_step = [s for s, v in data if v == max_val][0]
                print(f"  --- Min: {min_val:.6f} (iter {min_step}), Max: {max_val:.6f} (iter {max_step})")
            else:
                print(f"\n  {metric}: NOT FOUND")

        all_run_data[name] = run_data

    # Comparison tables
    print("\n" + "=" * 120)
    print("\nCOMPARISON TABLES (side by side)")
    print("=" * 120)

    run_names = list(accumulators.keys())

    for metric in TARGET_METRICS:
        # Check if any run has this metric
        has_data = any(all_run_data.get(r, {}).get(metric) for r in run_names)
        if not has_data:
            continue

        print(f"\n  {metric}")
        print(f"  {'-' * (8 + 18 * len(run_names))}")

        # Header
        header = f"  {'Iter':>6s}"
        for name in run_names:
            short = name[:16]
            header += f"  {short:>16s}"
        print(header)
        print(f"  {'----':>6s}" + "  " + ("  ".join(["-" * 16] * len(run_names))))

        # Use target iterations
        for target in TARGET_ITERS:
            row = f"  {target:6d}"
            any_value = False
            for name in run_names:
                data = all_run_data.get(name, {}).get(metric, [])
                result = find_closest_value(data, target)
                if result:
                    _, val = result
                    row += f"  {val:16.6f}"
                    any_value = True
                else:
                    row += f"  {'--':>16s}"
            if any_value:
                print(row)

        # Print last value for each run
        row_last = f"  {'LAST':>6s}"
        for name in run_names:
            data = all_run_data.get(name, {}).get(metric, [])
            if data:
                last_step, last_val = data[-1]
                row_last += f"  {last_val:12.4f}@{last_step:<3d}"
            else:
                row_last += f"  {'--':>16s}"
        print(row_last)

    # Summary table: final values
    print("\n" + "=" * 120)
    print("\nFINAL SUMMARY (last available value per metric)")
    print("=" * 120)

    header = f"  {'Metric':55s}"
    for name in run_names:
        short = name[:16]
        header += f"  {short:>16s}"
    print(header)
    print(f"  {'-'*55}" + "  " + ("  ".join(["-" * 16] * len(run_names))))

    for metric in TARGET_METRICS:
        row = f"  {metric:55s}"
        for name in run_names:
            data = all_run_data.get(name, {}).get(metric, [])
            if data:
                _, last_val = data[-1]
                row += f"  {last_val:16.6f}"
            else:
                row += f"  {'N/A':>16s}"
        print(row)

    # Also check for any additional interesting metrics
    print("\n" + "=" * 120)
    print("\nADDITIONAL METRICS (encoder-related, if available)")
    print("=" * 120)

    encoder_metrics = []
    for name, ea in accumulators.items():
        tags = ea.Tags().get("scalars", [])
        for t in tags:
            if any(k in t.lower() for k in ["encoder", "z_", "latent", "adaptation", "reconstruct"]):
                if t not in encoder_metrics:
                    encoder_metrics.append(t)

    if encoder_metrics:
        for metric in sorted(encoder_metrics):
            print(f"\n  {metric}")
            header = f"  {'Iter':>6s}"
            for name in run_names:
                short = name[:16]
                header += f"  {short:>16s}"
            print(header)
            print(f"  {'----':>6s}" + "  " + ("  ".join(["-" * 16] * len(run_names))))

            for target in [50, 100, 200, 300, 500, 600, 1000, 1500]:
                row = f"  {target:6d}"
                any_value = False
                for name in run_names:
                    data = get_scalar_data(accumulators[name], metric)
                    result = find_closest_value(data, target)
                    if result:
                        _, val = result
                        row += f"  {val:16.6f}"
                        any_value = True
                    else:
                        row += f"  {'--':>16s}"
                if any_value:
                    print(row)

            # Last value
            row_last = f"  {'LAST':>6s}"
            for name in run_names:
                data = get_scalar_data(accumulators[name], metric)
                if data:
                    last_step, last_val = data[-1]
                    row_last += f"  {last_val:12.4f}@{last_step:<3d}"
                else:
                    row_last += f"  {'--':>16s}"
            print(row_last)
    else:
        print("  No encoder-specific metrics found.")

    print("\n" + "=" * 120)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
