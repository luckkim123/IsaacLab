# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export utilities for Hero Agent deploy module (Phase 3).

Provides functions to export the HeroAgentDeployModule as:
    - TorchScript (.pt) via torch.jit.script
    - ONNX (.onnx) via torch.onnx.export
    - Config JSON (.json) for C++ TDC controller parameters

Reference: isaaclab_rl/rsl_rl/exporter.py (JIT/ONNX patterns)
"""

from __future__ import annotations

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)


def export_deploy_module_jit(
    module: torch.nn.Module,
    path: str,
    filename: str = "deploy_model.pt",
) -> str:
    """Export deploy module as TorchScript via torch.jit.script.

    Args:
        module: HeroAgentDeployModule instance (eval mode, frozen).
        path: Directory to save the exported model.
        filename: Output filename.

    Returns:
        Full path to the saved TorchScript file.

    Raises:
        RuntimeError: If JIT script or verification fails.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)

    module_cpu = module.cpu()
    module_cpu.eval()
    scripted = torch.jit.script(module_cpu)
    scripted.save(filepath)

    # Verify round-trip: reload and compare outputs
    loaded = torch.jit.load(filepath)
    loaded.eval()
    dummy_obs = torch.zeros(1, 13)
    dummy_hist = torch.zeros(1, 30, 12)

    with torch.no_grad():
        kp_orig, kd_orig, m_hat_orig = module_cpu(dummy_obs, dummy_hist)
        kp_load, kd_load, m_hat_load = loaded(dummy_obs, dummy_hist)

    pairs = [("kp", kp_orig, kp_load), ("kd", kd_orig, kd_load), ("m_hat", m_hat_orig, m_hat_load)]
    for name, orig, load in pairs:
        if not torch.allclose(orig, load, atol=1e-5):
            max_diff = torch.max(torch.abs(orig - load)).item()
            raise RuntimeError(f"JIT round-trip verification failed for {name}: max diff={max_diff:.2e}")

    logger.info("JIT model saved and verified: %s", filepath)
    return filepath


def export_deploy_module_onnx(
    module: torch.nn.Module,
    path: str,
    filename: str = "deploy_model.onnx",
    policy_obs_dim: int = 13,
    history_len: int = 30,
    history_dim: int = 12,
) -> str:
    """Export deploy module as ONNX.

    Args:
        module: HeroAgentDeployModule instance (eval mode, frozen).
        path: Directory to save the exported model.
        filename: Output filename.
        policy_obs_dim: Policy observation dimension.
        history_len: Proprioception history length.
        history_dim: Proprioception feature dimension.

    Returns:
        Full path to the saved ONNX file.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)

    module_cpu = module.cpu()
    module_cpu.eval()

    dummy_obs = torch.zeros(1, policy_obs_dim)
    dummy_hist = torch.zeros(1, history_len, history_dim)

    torch.onnx.export(
        module_cpu,
        (dummy_obs, dummy_hist),
        filepath,
        export_params=True,
        opset_version=18,
        input_names=["policy_obs", "proprio_hist"],
        output_names=["kp", "kd", "m_hat"],
        dynamic_axes={},
    )

    logger.info("ONNX model saved: %s", filepath)
    return filepath


def export_deploy_config(
    env_cfg: object,
    tdc_cfg: object,
    path: str,
    filename: str = "deploy_config.json",
) -> str:
    """Export TDC controller configuration as JSON for C++ consumer.

    Args:
        env_cfg: HeroAgentEncoderTDCEnvCfg (or subclass) instance.
        tdc_cfg: TDCControllerCfg instance.
        path: Directory to save the config file.
        filename: Output filename.

    Returns:
        Full path to the saved JSON file.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)

    config = {
        "tdc": {
            "m_hat_default": list(tdc_cfg.m_hat),
            "kp_default": tdc_cfg.kp,
            "kd_default": tdc_cfg.kd,
            "h": tdc_cfg.h,
            "dls_lambda_damping": tdc_cfg.dls_lambda_damping,
            "nu_dot_ema_alpha": tdc_cfg.nu_dot_ema_alpha,
            "base_position": list(tdc_cfg.base_position),
            "ik_dls_lambda": tdc_cfg.ik_dls_lambda,
            "max_joint_velocity": tdc_cfg.max_joint_velocity,
            "link1_length": tdc_cfg.link1_length,
            "link2_length": tdc_cfg.link2_length,
            "joint_stiffness": tdc_cfg.joint_stiffness,
            "joint_damping": tdc_cfg.joint_damping,
        },
        "gain_bounds": {
            "kp_range": list(getattr(env_cfg, "kp_range", (10.0, 100.0))),
            "kd_range": list(getattr(env_cfg, "kd_range", (2.0, 30.0))),
        },
        "timing": {
            "decimation": env_cfg.decimation,
            "control_decimation": env_cfg.control_decimation,
            "sim_dt": env_cfg.sim.dt,
            "step_dt": env_cfg.sim.dt * env_cfg.decimation,
            "control_dt": env_cfg.sim.dt * env_cfg.decimation * env_cfg.control_decimation,
        },
        "observation": {
            "policy_obs_dim": env_cfg.observation_space,
            "action_dim": env_cfg.action_space,
        },
    }

    if hasattr(env_cfg, "proprio_history_len"):
        config["adaptation"] = {
            "proprio_history_len": env_cfg.proprio_history_len,
            "proprio_feature_dim": env_cfg.proprio_feature_dim,
        }

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Config saved: %s", filepath)
    return filepath
