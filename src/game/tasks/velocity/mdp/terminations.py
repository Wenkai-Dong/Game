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

def subterrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        subterrain_origins = env.scene.env_origins[:, :2]
        pos_rel = asset.data.root_pos_w.torch[:, :2] - subterrain_origins

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(pos_rel[:, 0]) > 0.5 * grid_width - distance_buffer
        y_out_of_bounds = torch.abs(pos_rel[:, 1]) > 0.5 * grid_length - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
