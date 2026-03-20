# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug visualization utilities for constrained ALBC environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from isaaclab_tasks.models import HydrodynamicsModel


class DebugVisualization:
    """Manages debug visualization markers for underwater robots.

    Provides visualization for:
    - Center of Mass (CoM) - red sphere
    - Center of Buoyancy (CoB) - blue sphere
    - Payload CoG - yellow sphere (mass-scaled) with green stem from gripper (optional)
    """

    _PAYLOAD_COG_RADIUS: float = 0.012
    _PAYLOAD_STEM_RADIUS: float = 0.008
    _MIN_STEM_DIST: float = 1e-3
    _HIDDEN_POS: float = 1e6

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
        self._payload_cog_marker: VisualizationMarkers | None = None
        self._payload_stem_marker: VisualizationMarkers | None = None
        self._frame_marker: VisualizationMarkers | None = None

    def setup(self, enable_payload: bool = False) -> None:
        """Create visualization markers.

        Args:
            enable_payload: Whether to create payload markers.
        """
        if self._markers_created:
            return

        self._com_marker = self._create_sphere_marker("/Visuals/CoM", color=(1.0, 0.0, 0.0))
        self._cob_marker = self._create_sphere_marker("/Visuals/CoB", color=(0.0, 0.0, 1.0))

        if enable_payload:
            self._payload_cog_marker = self._create_sphere_marker(
                "/Visuals/PayloadCoG", color=(1.0, 0.85, 0.0), radius=self._PAYLOAD_COG_RADIUS
            )
            self._payload_stem_marker = self._create_cylinder_marker(
                "/Visuals/PayloadStem", color=(0.0, 0.8, 0.0), radius=self._PAYLOAD_STEM_RADIUS, height=1.0
            )

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

        for marker in (
            self._com_marker,
            self._cob_marker,
            self._payload_cog_marker,
            self._payload_stem_marker,
            self._frame_marker,
        ):
            if marker is not None:
                marker.set_visibility(visible)

    def update(
        self,
        robot: Articulation,
        body_id: list[int],
        buoy_body_id: list[int],
        hydro: HydrodynamicsModel,
        buoy_hydro: HydrodynamicsModel,
        gripper_body_id: list[int] | None = None,
        payload_mass: torch.Tensor | None = None,
        payload_offset: torch.Tensor | None = None,
        payload_cog_offset: torch.Tensor | None = None,
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
            gripper_body_id: Gripper body index (for payload visualization).
            payload_mass: Payload mass per environment (optional).
            payload_offset: Payload attachment offset in gripper frame (optional).
            payload_cog_offset: Payload CoG offset from attachment in gripper frame (optional).
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

        # Payload visualization: stem (cylinder) from attachment to CoG, sphere at CoG
        self._update_payload_markers(
            robot, gripper_body_id, payload_mass, payload_offset, payload_cog_offset, default_payload_mass
        )

        # World frame marker at origin (single instance, not per-environment)
        if self._frame_marker is not None:
            origin = torch.zeros(1, 3, device=self._device)
            self._frame_marker.visualize(translations=origin)

    def _update_payload_markers(
        self,
        robot: Articulation,
        gripper_body_id: list[int] | None,
        payload_mass: torch.Tensor | None,
        payload_offset: torch.Tensor | None,
        payload_cog_offset: torch.Tensor | None,
        default_payload_mass: float,
    ) -> None:
        """Update payload stem and CoG sphere markers.

        Draws a thin green cylinder from the gripper attachment point to the
        effective CoG position, with a yellow sphere at the CoG scaled by mass.
        """
        if gripper_body_id is None or payload_mass is None or payload_offset is None:
            return

        gripper_idx = gripper_body_id[0]
        gripper_pos_w = robot.data.body_pos_w[:, gripper_idx]
        gripper_quat_w = robot.data.body_quat_w[:, gripper_idx]

        # Attachment point in world frame
        attach_pos_w = gripper_pos_w + quat_apply(gripper_quat_w, payload_offset)

        # Effective CoG in world frame (attachment + cog_offset)
        if payload_cog_offset is not None:
            effective_offset = payload_offset + payload_cog_offset
        else:
            effective_offset = payload_offset
        cog_pos_w = gripper_pos_w + quat_apply(gripper_quat_w, effective_offset)

        # CoG sphere (yellow, mass-scaled)
        if self._payload_cog_marker is not None:
            scale_factor = (payload_mass / default_payload_mass).unsqueeze(-1).expand(-1, 3)
            self._payload_cog_marker.visualize(translations=cog_pos_w, scales=scale_factor)

        # Stem cylinder from attachment to CoG
        if self._payload_stem_marker is not None and payload_cog_offset is not None:
            self._update_stem(attach_pos_w, cog_pos_w)

    def _update_stem(
        self,
        start_w: torch.Tensor,
        end_w: torch.Tensor,
    ) -> None:
        """Update stem cylinder between two world-frame points.

        The cylinder marker has default height=1.0 and axis=Z. We compute
        the midpoint, orientation (Z -> direction), and z-scale (= distance).
        """
        assert self._payload_stem_marker is not None

        direction = end_w - start_w
        dist = direction.norm(dim=-1, keepdim=True)

        # Hide stem for near-zero cog_offset (< 1mm)
        valid = dist.squeeze(-1) > self._MIN_STEM_DIST
        if not valid.any():
            # Move all stems out of view
            far_away = torch.full_like(start_w, self._HIDDEN_POS)
            self._payload_stem_marker.visualize(translations=far_away)
            return

        # Midpoint
        midpoint = (start_w + end_w) * 0.5

        # Orientation: rotate Z-axis to match direction
        dir_norm = direction / dist.clamp(min=self._MIN_STEM_DIST)
        orientations = _quat_from_z_to_direction(dir_norm, self._device)

        # Scale: (1, 1, dist) since cylinder height=1.0 along Z
        scales = torch.ones(self._num_envs, 3, device=self._device)
        scales[:, 2] = dist.squeeze(-1)

        # For near-zero offset envs, hide by moving far away
        midpoint[~valid] = self._HIDDEN_POS

        self._payload_stem_marker.visualize(translations=midpoint, orientations=orientations, scales=scales)

    @staticmethod
    def _create_sphere_marker(
        prim_path: str, color: tuple[float, float, float], radius: float = 0.03
    ) -> VisualizationMarkers:
        """Create a sphere marker for debug visualization."""
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
    def _create_cylinder_marker(
        prim_path: str, color: tuple[float, float, float], radius: float = 0.004, height: float = 1.0
    ) -> VisualizationMarkers:
        """Create a cylinder marker for debug visualization."""
        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        return VisualizationMarkers(cfg)

    @staticmethod
    def _create_frame_marker(prim_path: str, scale: float = 0.5) -> VisualizationMarkers:
        """Create a coordinate frame marker (XYZ axes) for debug visualization."""
        cfg = FRAME_MARKER_CFG.copy()
        cfg.prim_path = prim_path
        cfg.markers["frame"].scale = (scale, scale, scale)
        return VisualizationMarkers(cfg)


def _quat_from_z_to_direction(direction: torch.Tensor, device: str) -> torch.Tensor:
    """Compute quaternion that rotates Z-axis to the given direction.

    Args:
        direction: Normalized direction vectors. Shape: (N, 3).
        device: Torch device.

    Returns:
        Quaternions (w, x, y, z). Shape: (N, 4).
    """
    n = direction.shape[0]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, -1)

    # cos(angle) = dot(z, dir)
    cos_angle = (z_axis * direction).sum(dim=-1)

    # Cross product gives rotation axis (and sin of angle)
    cross = torch.cross(z_axis, direction, dim=-1)
    sin_angle = cross.norm(dim=-1)

    # angle between z and direction
    angle = torch.atan2(sin_angle, cos_angle)

    # Rotation axis (handle parallel/anti-parallel cases)
    # When direction ~= +Z, axis is arbitrary (angle ~= 0, rotation is identity)
    # When direction ~= -Z, pick arbitrary perpendicular axis
    fallback_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, -1)
    axis = torch.where(sin_angle.unsqueeze(-1) > 1e-6, cross / sin_angle.unsqueeze(-1).clamp(min=1e-8), fallback_axis)

    return quat_from_angle_axis(angle, axis)
