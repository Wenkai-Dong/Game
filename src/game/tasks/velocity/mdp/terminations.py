# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RayCaster, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    terrain_height = torch.amax(sensor.data.ray_hits_w.torch[..., 2], dim=1)
    terrain_height = torch.nan_to_num(terrain_height, nan=-1e6, posinf=-1e6, neginf=-1e6)
    relative_height = asset.data.root_pos_w.torch[:, 2] - terrain_height
    return relative_height < minimum_height
