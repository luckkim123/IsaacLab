# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug visualization utilities for Hero Agent environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from isaaclab_tasks.models import HydrodynamicsModel


class DebugVisualization:
    """Manages debug visualization markers for underwater robots.

    Provides visualization for:
    - Center of Mass (CoM) - red sphere
    - Center of Buoyancy (CoB) - blue sphere
    - Payload attachment point - green sphere (optional)
    """

    def __init__(self, num_envs: int, device: str):
        """Initialize debug visualization manager.

        Args:
            num_envs: Number of parallel environments.
            device: Torch device for computations.
        """
        self._num_envs = num_envs
        self._device = device
        self._markers_created = False

        # Marker instances (created lazily)
        self._com_marker: VisualizationMarkers | None = None
        self._cob_marker: VisualizationMarkers | None = None
        self._payload_marker: VisualizationMarkers | None = None
        self._frame_marker: VisualizationMarkers | None = None

    def setup(self, enable_payload: bool = False) -> None:
        """Create visualization markers.

        Args:
            enable_payload: Whether to create payload marker.
        """
        if self._markers_created:
            return

        self._com_marker = self._create_sphere_marker("/Visuals/CoM", color=(1.0, 0.0, 0.0))
        self._cob_marker = self._create_sphere_marker("/Visuals/CoB", color=(0.0, 0.0, 1.0))

        if enable_payload:
            self._payload_marker = self._create_sphere_marker("/Visuals/Payload", color=(0.0, 1.0, 0.0))

        # World frame marker at origin (XYZ axes)
        self._frame_marker = self._create_frame_marker("/Visuals/WorldFrame", scale=0.5)

        self._markers_created = True

    def set_visibility(self, visible: bool) -> None:
        """Set visibility for all markers.

        Args:
            visible: Whether markers should be visible.
        """
        if not self._markers_created:
            if visible:
                self.setup()
            return

        for marker in (self._com_marker, self._cob_marker, self._payload_marker, self._frame_marker):
            if marker is not None:
                marker.set_visibility(visible)

    def update(
        self,
        robot: Articulation,
        body_id: list[int],
        buoy_body_id: list[int],
        hydro: HydrodynamicsModel,
        buoy_hydro: HydrodynamicsModel,
        payload_mass: torch.Tensor | None = None,
        payload_offset: torch.Tensor | None = None,
        default_payload_mass: float = 1.0,
    ) -> None:
        """Update marker positions based on current robot state.

        Computes system-level CoM and CoB by combining main body and buoy,
        then transforms from body frame to world frame.

        Args:
            robot: Robot articulation.
            body_id: Main body index.
            buoy_body_id: Buoy body index.
            hydro: Main body hydrodynamics model.
            buoy_hydro: Buoy hydrodynamics model.
            payload_mass: Payload mass per environment (optional).
            payload_offset: Payload attachment offset per environment (optional).
            default_payload_mass: Default payload mass for scaling visualization.
        """
        if not self._markers_created or robot.root_physx_view is None:
            return

        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w

        # Get mass and volume parameters
        body_masses = robot.root_physx_view.get_masses()[0].to(self._device)
        base_idx, buoy_idx = body_id[0], buoy_body_id[0]

        m_main = body_masses[base_idx].expand(self._num_envs)
        m_buoy = body_masses[buoy_idx].expand(self._num_envs)
        V_main = hydro.volume
        V_buoy = buoy_hydro.volume

        # Compute buoy offset in body frame
        buoy_pos_w = robot.data.body_pos_w[:, buoy_idx]
        buoy_offset_b = quat_apply_inverse(root_quat, buoy_pos_w - root_pos)

        # System CoM (mass-weighted average)
        m_total = m_main + m_buoy
        r_cg_system_b = (
            m_main.unsqueeze(-1) * hydro.center_of_gravity
            + m_buoy.unsqueeze(-1) * (buoy_offset_b + buoy_hydro.center_of_gravity)
        ) / m_total.unsqueeze(-1)

        # System CoB (volume-weighted average)
        V_total = V_main + V_buoy
        r_cb_system_b = (
            V_main.unsqueeze(-1) * hydro.center_of_buoyancy
            + V_buoy.unsqueeze(-1) * (buoy_offset_b + buoy_hydro.center_of_buoyancy)
        ) / V_total.unsqueeze(-1)

        # Transform to world frame and update markers
        if self._com_marker is not None:
            self._com_marker.visualize(translations=root_pos + quat_apply(root_quat, r_cg_system_b))
        if self._cob_marker is not None:
            self._cob_marker.visualize(translations=root_pos + quat_apply(root_quat, r_cb_system_b))

        # Payload marker (position + mass-scaled size)
        if self._payload_marker is not None and payload_mass is not None and payload_offset is not None:
            payload_pos_world = root_pos + quat_apply(root_quat, payload_offset)
            scale_factor = (payload_mass / default_payload_mass).unsqueeze(-1).expand(-1, 3)
            self._payload_marker.visualize(translations=payload_pos_world, scales=scale_factor)

        # World frame marker at origin (single instance, not per-environment)
        if self._frame_marker is not None:
            origin = torch.zeros(1, 3, device=self._device)
            self._frame_marker.visualize(translations=origin)

    @staticmethod
    def _create_sphere_marker(
        prim_path: str, color: tuple[float, float, float], radius: float = 0.03
    ) -> VisualizationMarkers:
        """Create a sphere marker for debug visualization.

        Args:
            prim_path: USD prim path for the marker.
            color: RGB color tuple (0-1 range).
            radius: Sphere radius in meters.

        Returns:
            Configured VisualizationMarkers instance.
        """
        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        return VisualizationMarkers(cfg)

    @staticmethod
    def _create_frame_marker(prim_path: str, scale: float = 0.5) -> VisualizationMarkers:
        """Create a coordinate frame marker (XYZ axes) for debug visualization.

        Args:
            prim_path: USD prim path for the marker.
            scale: Scale factor for the frame size.

        Returns:
            Configured VisualizationMarkers instance with XYZ axes.
        """
        cfg = FRAME_MARKER_CFG.copy()
        cfg.prim_path = prim_path
        cfg.markers["frame"].scale = (scale, scale, scale)
        return VisualizationMarkers(cfg)
