# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the velocity-tracking locomotion environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def sub_terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> dict[str, torch.Tensor]:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.linalg.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain levels of sub-terrain
    gen_cfg = terrain.cfg.terrain_generator
    if not hasattr(env, "_col_to_sub"):
        props = torch.tensor([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=torch.float64)
        cum = torch.cumsum(props / props.sum(), dim=0)
        cols = torch.arange(gen_cfg.num_cols, dtype=torch.float64) / gen_cfg.num_cols + 0.001
        env._col_to_sub = torch.searchsorted(cum, cols, right=True).to(terrain.terrain_types.device)
    sub_type = env._col_to_sub[terrain.terrain_types]

    levels = {}
    for i, name in enumerate(gen_cfg.sub_terrains.keys()):
        mask = sub_type == i
        if mask.any():
            levels[name] = terrain.terrain_levels[mask].float().mean()
    levels["all"] = terrain.terrain_levels.float().mean()
    return levels