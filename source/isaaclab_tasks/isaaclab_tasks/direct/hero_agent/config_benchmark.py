# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark scenario presets for systematic evaluation of Hero Agent controllers.

Each scenario defines a controlled combination of domain randomization intensity,
ocean current strength, and payload mass. This enables fair comparison across:
    - nominal: No randomization (can it stabilize at all?)
    - easy: Mild perturbation (+/-10% hydro, 0.1 m/s current)
    - hard: Training distribution (+/-30-50% hydro, 0.3 m/s current)
    - extreme: Out-of-distribution stress test (beyond training range)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_assets.robots.uuv import OceanCurrentCfg

from .config import DomainRandomizationCfg


@dataclass
class BenchmarkScenario:
    """A controlled evaluation scenario combining DR, ocean current, and payload settings."""

    name: str
    dr_cfg: DomainRandomizationCfg = field(default_factory=DomainRandomizationCfg)
    ocean_current_cfg: OceanCurrentCfg = field(default_factory=OceanCurrentCfg)
    enable_payload: bool = True
    description: str = ""


def get_benchmark_scenarios() -> dict[str, BenchmarkScenario]:
    """Return the four standard benchmark scenarios.

    Returns:
        Dict mapping scenario name to BenchmarkScenario.
    """
    return {
        "nominal": _make_nominal(),
        "easy": _make_easy(),
        "hard": _make_hard(),
        "extreme": _make_extreme(),
    }


# ---------------------------------------------------------------------------
# Scenario factories
# ---------------------------------------------------------------------------


def _make_nominal() -> BenchmarkScenario:
    """No randomization. Fixed 0.5kg payload. No ocean current."""
    return BenchmarkScenario(
        name="nominal",
        dr_cfg=DomainRandomizationCfg.fixed_pose(),
        ocean_current_cfg=OceanCurrentCfg(
            max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        enable_payload=True,
        description="Baseline: fixed parameters, no disturbances, 0.5kg payload",
    )


def _make_easy() -> BenchmarkScenario:
    """Mild perturbation: +/-10% hydro, 0.1 m/s current, 0-0.3kg payload."""
    return BenchmarkScenario(
        name="easy",
        dr_cfg=DomainRandomizationCfg(
            enable=True,
            # Mild hydro variation
            added_mass_scale=(0.9, 1.1),
            linear_damping_scale=(0.9, 1.1),
            quadratic_damping_scale=(0.9, 1.1),
            volume_scale=(0.95, 1.05),
            inertia_scale=(0.9, 1.1),
            body_mass_scale=(0.95, 1.05),
            water_density_range=(997.0, 1003.0),
            # Small CoB/CoG offsets
            cob_offset_x=(-0.005, 0.005),
            cob_offset_y=(-0.005, 0.005),
            cob_offset_z=(-0.02, 0.02),
            cog_offset_x=(-0.005, 0.005),
            cog_offset_y=(-0.005, 0.005),
            cog_offset_z=(-0.02, 0.02),
            # Mild joint variation
            joint_stiffness_range=(90.0, 110.0),
            joint_damping_range=(2.7, 3.3),
            joint_static_friction_range=(0.0, 0.02),
            joint_viscous_friction_range=(0.0, 0.1),
            # Light payload
            payload_mass_range=(0.0, 0.3),
            payload_cog_offset_x=(-0.1, 0.1),
            payload_cog_offset_y=(-0.1, 0.1),
            payload_cog_offset_z=(-0.1, 0.0),
        ),
        ocean_current_cfg=OceanCurrentCfg(
            max_velocity=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
            noise_scale=(0.02, 0.02, 0.01, 0.0, 0.0, 0.0),
        ),
        enable_payload=True,
        description="Mild perturbation: +/-10% hydro, 0.1 m/s current, 0-0.3kg payload",
    )


def _make_hard() -> BenchmarkScenario:
    """Training distribution: matches HeroAgentTrainEnvCfg DR ranges."""
    return BenchmarkScenario(
        name="hard",
        dr_cfg=DomainRandomizationCfg(
            enable=True,
            # Same as HeroAgentTrainEnvCfg defaults
            added_mass_scale=(0.7, 1.3),
            linear_damping_scale=(0.7, 1.3),
            quadratic_damping_scale=(0.6, 1.4),
            volume_scale=(0.9, 1.1),
            inertia_scale=(0.8, 1.2),
            body_mass_scale=(0.9, 1.1),
            water_density_range=(995.0, 1025.0),
            cob_offset_x=(-0.01, 0.01),
            cob_offset_y=(-0.01, 0.01),
            cob_offset_z=(-0.04, 0.04),
            cog_offset_x=(-0.01, 0.01),
            cog_offset_y=(-0.01, 0.01),
            cog_offset_z=(-0.04, 0.04),
            joint_stiffness_range=(80.0, 120.0),
            joint_damping_range=(2.4, 3.6),
            joint_static_friction_range=(0.0, 0.05),
            joint_viscous_friction_range=(0.0, 0.3),
            payload_mass_range=(0.0, 1.0),
            payload_cog_offset_x=(-0.50, 0.50),
            payload_cog_offset_y=(-0.50, 0.50),
            payload_cog_offset_z=(-0.20, 0.0),
        ),
        ocean_current_cfg=OceanCurrentCfg(
            max_velocity=(0.2, 0.2, 0.1, 0.0, 0.0, 0.0),
            noise_scale=(0.05, 0.05, 0.02, 0.0, 0.0, 0.0),
        ),
        enable_payload=True,
        description="Training distribution: matches HeroAgentTrainEnvCfg DR ranges",
    )


def _make_extreme() -> BenchmarkScenario:
    """Out-of-distribution: +/-50-100% hydro, 0.5 m/s current, 0.5-2.5kg payload."""
    return BenchmarkScenario(
        name="extreme",
        dr_cfg=DomainRandomizationCfg(
            enable=True,
            # Beyond training range
            added_mass_scale=(0.5, 2.0),
            linear_damping_scale=(0.5, 2.0),
            quadratic_damping_scale=(0.3, 2.5),
            volume_scale=(0.7, 1.3),
            inertia_scale=(0.5, 2.0),
            body_mass_scale=(0.7, 1.3),
            water_density_range=(980.0, 1040.0),
            # Large CoB/CoG offsets
            cob_offset_x=(-0.03, 0.03),
            cob_offset_y=(-0.03, 0.03),
            cob_offset_z=(-0.08, 0.08),
            cog_offset_x=(-0.03, 0.03),
            cog_offset_y=(-0.03, 0.03),
            cog_offset_z=(-0.08, 0.08),
            # Wide joint variation
            joint_stiffness_range=(60.0, 150.0),
            joint_damping_range=(1.5, 5.0),
            joint_static_friction_range=(0.0, 0.1),
            joint_viscous_friction_range=(0.0, 0.6),
            # Heavy payload (OOD)
            payload_mass_range=(0.5, 2.5),
            payload_cog_offset_x=(-0.80, 0.80),
            payload_cog_offset_y=(-0.80, 0.80),
            payload_cog_offset_z=(-0.40, 0.0),
        ),
        ocean_current_cfg=OceanCurrentCfg(
            max_velocity=(0.5, 0.5, 0.25, 0.0, 0.0, 0.0),
            noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
        ),
        enable_payload=True,
        description="OOD stress test: +/-50-100% hydro, 0.5 m/s current, 0.5-2.5kg payload",
    )
