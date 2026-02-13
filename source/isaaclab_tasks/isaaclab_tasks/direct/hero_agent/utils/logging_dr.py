# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization metrics logging for Hero Agent environment.

Provides log_dr_metrics for tracking the distribution of randomized parameters
across parallel environments. Called from base_env._collect_episode_metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..base_env import HeroAgentEnv


def log_dr_metrics(
    extras: dict,
    env: HeroAgentEnv,
    robot: Articulation,
    joint_ids: list[int],
) -> None:
    """Log domain randomization parameter statistics to extras["log"].

    Reads per-env DR state from hydrodynamic models, joint parameters, and
    payload config. All values are logged as scalar summaries (mean/std) so
    TensorBoard/WandB can track the distribution of randomized parameters
    across parallel environments.

    Args:
        extras: Environment extras dictionary (must have "log" key).
        env: HeroAgentEnv instance with _hydro, _buoy_hydro, etc.
        robot: Robot articulation for joint parameter access.
        joint_ids: ALBC joint indices for stiffness/damping readout.
    """
    log = extras["log"]

    with torch.no_grad():
        # Main body hydrodynamics
        hydro = env._hydro
        log["DR/volume_mean"] = hydro.volume.mean().item()
        log["DR/water_density_mean"] = hydro.water_density.mean().item()
        log["DR/water_density_std"] = hydro.water_density.std().item()
        log["DR/buoyancy_force_mean"] = hydro.buoyancy_force.mean().item()
        log["DR/cob_z_mean"] = hydro.center_of_buoyancy[:, 2].mean().item()
        log["DR/cog_z_mean"] = hydro.center_of_gravity[:, 2].mean().item()
        log["DR/inertia_mean"] = hydro.rigid_body_inertia.mean().item()

        # Buoy hydrodynamics
        buoy = env._buoy_hydro
        log["DR/buoy_volume_mean"] = buoy.volume.mean().item()
        log["DR/buoy_buoyancy_mean"] = buoy.buoyancy_force.mean().item()

        # Joint actuator gains
        stiffness = robot.data.joint_stiffness[:, joint_ids]
        damping = robot.data.joint_damping[:, joint_ids]
        log["DR/joint_stiffness_mean"] = stiffness.mean().item()
        log["DR/joint_stiffness_std"] = stiffness.std().item()
        log["DR/joint_damping_mean"] = damping.mean().item()

        # Buoyancy force distribution (critical for TDC lambda)
        log["DR/buoyancy_force_std"] = hydro.buoyancy_force.std().item()

        # Payload (if enabled)
        if env._payload_mass is not None:
            log["DR/payload_mass_mean"] = env._payload_mass.mean().item()
            log["DR/payload_mass_std"] = env._payload_mass.std().item()
        if env._payload_cog_offset is not None:
            log["DR/payload_cog_offset_norm_mean"] = torch.linalg.norm(env._payload_cog_offset, dim=-1).mean().item()

        # Ocean current (if model has current velocity)
        if hasattr(hydro, "_current_velocity") and hydro._current_velocity is not None:
            current_mag = torch.linalg.norm(hydro._current_velocity[:, :3], dim=-1)
            log["DR/ocean_current_mag_mean"] = current_mag.mean().item()
