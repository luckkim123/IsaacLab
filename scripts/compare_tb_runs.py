"""Compare TensorBoard metrics between two runs at the same iteration count."""

import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Run paths ──
RUN_A_DIR = "/workspace/isaaclab/logs/rsl_rl/full_dof_trpo/2026-04-04_16-06-48"
RUN_B_DIR = "/workspace/isaaclab/logs/rsl_rl/full_dof_trpo/2026-04-04_21-10-13"

# ── Tag prefixes to extract ──
TAG_PREFIXES = [
    "Episode_Reward/",
    "Thruster/",
    "Action/",
    "Velocity_Tracking/",
    "Command/",
    "Constraint/",
    "Train/",
    "DORAEMON/",
    "Loss/",
    "Encoder/",
]


def load_event_accumulator(log_dir: str) -> EventAccumulator:
    """Load TB event files from a directory."""
    # Find event file
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        print(f"ERROR: No event files found in {log_dir}")
        sys.exit(1)

    ea = EventAccumulator(log_dir)
    ea.Reload()
    return ea


def get_metric_at_step(ea: EventAccumulator, tag: str, target_step: int) -> float | None:
    """Get the metric value at the closest step <= target_step."""
    try:
        events = ea.Scalars(tag)
    except KeyError:
        return None

    if not events:
        return None

    # Find the event at or closest to target_step
    best = None
    best_diff = float("inf")
    for e in events:
        diff = abs(e.step - target_step)
        if diff < best_diff:
            best_diff = diff
            best = e
    return best.value if best is not None else None


def get_max_step(ea: EventAccumulator) -> int:
    """Get the maximum step in the event accumulator."""
    max_step = 0
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        if events:
            max_step = max(max_step, max(e.step for e in events))
    return max_step


def main():
    print("=" * 100)
    print("TensorBoard Metric Comparison: Run A vs Run B")
    print("=" * 100)
    print(f"Run A: {RUN_A_DIR}")
    print(f"Run B: {RUN_B_DIR}")
    print()

    # Load events
    ea_a = load_event_accumulator(RUN_A_DIR)
    ea_b = load_event_accumulator(RUN_B_DIR)

    # Determine comparison iteration (last step of Run B)
    max_step_a = get_max_step(ea_a)
    max_step_b = get_max_step(ea_b)
    print(f"Run A max iteration: {max_step_a}")
    print(f"Run B max iteration: {max_step_b}")

    # Compare at the last iter of run B
    target_step = max_step_b
    print(f"\nComparing at iteration: {target_step}")
    print()

    # ── Config diff ──
    print("-" * 100)
    print("CONFIG DIFFERENCES (from agent.yaml)")
    print("-" * 100)
    print(f"  {'Parameter':<40s} {'Run A':<20s} {'Run B':<20s}")
    print(f"  {'entropy_coef':<40s} {'0.001':<20s} {'0.003':<20s}")
    print()

    # Collect all matching tags from both runs
    tags_a = set(ea_a.Tags().get("scalars", []))
    tags_b = set(ea_b.Tags().get("scalars", []))

    # Filter tags by prefix
    relevant_tags = set()
    for tag in tags_a | tags_b:
        for prefix in TAG_PREFIXES:
            if tag.startswith(prefix):
                relevant_tags.add(tag)
                break

    # Sort tags by category
    relevant_tags = sorted(relevant_tags)

    # Group by prefix
    current_prefix = ""
    print("-" * 100)
    print(f"  {'Metric':<55s} {'Run A':>15s} {'Run B':>15s} {'Diff (B-A)':>15s}")
    print("-" * 100)

    for tag in relevant_tags:
        # Print section header
        prefix = tag.split("/")[0] + "/"
        if prefix != current_prefix:
            current_prefix = prefix
            print(f"\n  [{prefix[:-1]}]")

        val_a = get_metric_at_step(ea_a, tag, target_step)
        val_b = get_metric_at_step(ea_b, tag, target_step)

        # Format values
        str_a = f"{val_a:>15.6f}" if val_a is not None else f"{'N/A':>15s}"
        str_b = f"{val_b:>15.6f}" if val_b is not None else f"{'N/A':>15s}"

        if val_a is not None and val_b is not None:
            diff = val_b - val_a
            # Color indication
            str_diff = f"{diff:>+15.6f}"
        else:
            str_diff = f"{'N/A':>15s}"

        # Short tag name
        short_tag = tag
        print(f"  {short_tag:<55s} {str_a} {str_b} {str_diff}")

    print()
    print("=" * 100)

    # ── Additional: print ALL available tags per run for reference ──
    print("\nAll scalar tags in Run A but NOT in Run B:")
    only_a = tags_a - tags_b
    for t in sorted(only_a):
        print(f"  {t}")

    print("\nAll scalar tags in Run B but NOT in Run A:")
    only_b = tags_b - tags_a
    for t in sorted(only_b):
        print(f"  {t}")

    print()


if __name__ == "__main__":
    main()
