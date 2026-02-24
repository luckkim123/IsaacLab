"""Create a WandB Report with 8-panel SAC-MPC monitoring dashboard.

Usage:
    # Preview panel layout (no WandB dependency):
    ./isaaclab.sh -p scripts/hero_agent/create_wandb_dashboard.py --dry-run

    # Create WandB report (requires: pip install wandb[workspaces]):
    ./isaaclab.sh -p scripts/hero_agent/create_wandb_dashboard.py \
        --entity <WANDB_ENTITY> --project hero_agent

    # With specific run filter:
    ./isaaclab.sh -p scripts/hero_agent/create_wandb_dashboard.py \
        --entity <WANDB_ENTITY> --project hero_agent \
        --run-filter "SAC-MPC*"

See docs/SAC_MPC_MONITORING.md for metric interpretation guide.
"""

from __future__ import annotations

import argparse
import sys


# =============================================================================
# Panel Definitions (source of truth for all 8 panels)
# =============================================================================

PANELS: list[dict[str, str | list[str]]] = [
    {
        "title": "1. Is Training Working? (Primary)",
        "metrics": [
            "Episode_Reward/total",
            "Train/episode_length",
            "Attitude_Error/roll_deg",
            "Attitude_Error/pitch_deg",
        ],
    },
    {
        "title": "2. Reward Breakdown (Per-Term)",
        "metrics": [
            "Episode_Reward/tracking",
            "Episode_Reward/action_magnitude",
            "Episode_Reward/action_rate",
        ],
    },
    {
        "title": "3. Actor-Critic Health",
        "metrics": [
            "Loss/actor",
            "Loss/critic",
            "SAC/alpha",
            "SAC/q1_mean",
            "SAC/q2_mean",
            "SAC/target_q_mean",
            "SAC/log_prob_mean",
            "SAC/actor_grad_norm",
            "SAC/critic_grad_norm",
        ],
    },
    {
        "title": "4. Dynamics MLP Health",
        "metrics": [
            "Loss/dynamics",
            "Dynamics/pred_err_mean",
            "Dynamics/grad_norm",
        ],
    },
    {
        "title": "5. Encoder Health",
        "metrics": [
            "Encoder/z_mean",
            "Encoder/z_std",
            "Encoder/grad_norm",
        ],
    },
    {
        "title": "6. MPC Health",
        "metrics": [
            "MPC/Q_diag_mean",
            "MPC/R_diag_mean",
            "MPC/solve_cost",
            "MPC/state_err_total",
        ],
    },
    {
        "title": "7. Training Diagnostics",
        "metrics": [
            "Train/fps",
            "Train/buffer_size",
            "Train/terminated",
            "Train/time_out",
        ],
    },
    {
        "title": "8. DR Parameters (Sanity Check)",
        "metrics": [
            "DR/buoyancy_force_mean",
            "DR/inertia_mean",
            "DR/payload_mass_mean",
            "DR/ocean_current_mag_mean",
        ],
    },
]


def dry_run() -> None:
    """Print the panel layout without creating a WandB report."""
    total = 0
    print("=" * 60)
    print("SAC-MPC Training Health Dashboard - Panel Layout")
    print("=" * 60)
    for panel in PANELS:
        metrics = panel["metrics"]
        print(f"\n## {panel['title']} ({len(metrics)} metrics)")
        for m in metrics:
            print(f"  - {m}")
        total += len(metrics)
    print(f"\n{'=' * 60}")
    print(f"Total: {total} metrics across {len(PANELS)} panels")
    print("=" * 60)


def create_report(entity: str, project: str, run_filter: str | None = None) -> str:
    """Create WandB report with 8-panel SAC-MPC dashboard.

    Args:
        entity: WandB entity (username or team).
        project: WandB project name.
        run_filter: Optional run name glob filter (e.g., "SAC-MPC*").

    Returns:
        URL of the created report.
    """
    try:
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        print(
            "ERROR: wandb_workspaces not installed.\n"
            "Install with: pip install wandb[workspaces]\n"
            "Or use --dry-run to preview panel layout without creating a report.",
            file=sys.stderr,
        )
        sys.exit(1)

    blocks: list = [
        wr.H1("SAC-MPC Training Health Dashboard"),
        wr.P(
            "Auto-generated monitoring dashboard. "
            "See docs/SAC_MPC_MONITORING.md for metric interpretation guide."
        ),
    ]

    # Build runset with optional filter
    runset_kwargs: dict = {"entity": entity, "project": project}
    if run_filter:
        runset_kwargs["filters"] = {"display_name": {"$regex": run_filter}}

    runset = wr.Runset(**runset_kwargs)

    # Add each panel section
    for panel_def in PANELS:
        metrics = panel_def["metrics"]
        blocks.append(wr.H2(str(panel_def["title"])))
        blocks.append(
            wr.PanelGrid(
                panels=[
                    wr.LinePlot(
                        title=str(panel_def["title"]),
                        y=[m for m in metrics],
                        smoothing_factor=0.6,
                    )
                ],
                runsets=[runset],
            )
        )

    report = wr.Report(
        entity=entity,
        project=project,
        title="SAC-MPC Training Health Dashboard",
        description="8-panel monitoring dashboard for SAC-MPC training health assessment",
        blocks=blocks,
    )

    report.save()
    url = report.url
    print(f"Report created: {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create WandB Report with SAC-MPC monitoring dashboard"
    )
    parser.add_argument("--entity", default=None, help="WandB entity (username or team)")
    parser.add_argument("--project", default="hero_agent", help="WandB project name")
    parser.add_argument(
        "--run-filter",
        default=None,
        help="Optional run name glob filter (e.g., 'SAC-MPC*')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print panel layout without creating WandB report",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        if args.entity is None:
            parser.error("--entity is required when not using --dry-run")
        create_report(args.entity, args.project, args.run_filter)


if __name__ == "__main__":
    main()
