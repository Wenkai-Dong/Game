# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


"""
Joint penalties.
"""


def joint_torque_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.applied_torque.torch[:, asset_cfg.joint_ids])
        - asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to min=0.0
    out_of_limits = out_of_limits.clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


"""
Robot.
"""


def joint_deviation_l2(
    env: ManagerBasedRLEnv, threshold: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one beyond a threshold."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    squared_error = torch.sum(torch.square(angle), dim=1)
    return (squared_error - threshold).clip(min=0.0)


def straight_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    selected_bodies_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    # (num_envs, num_bodies, 3) via broadcasting, no memory copy
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    gravity_vec = gravity_vec.expand(selected_bodies_quat.shape[0], selected_bodies_quat.shape[1], 3)
    projected_gravity = quat_apply_inverse(selected_bodies_quat, gravity_vec)
    return torch.sum(torch.square(projected_gravity[..., :2]), dim=(1, 2))


def stand_still_velocity(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.05
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_vel.torch), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < threshold)