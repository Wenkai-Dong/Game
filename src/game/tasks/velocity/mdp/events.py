# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


class randomize_virtual_floor(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        vfloor_range = cfg.params.get("vfloor_range", (-0.4, -1.0))
        env.vfloor_z = math_utils.sample_uniform(*vfloor_range, (env.scene.num_envs,), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        vfloor_range: tuple[float, float] = (-0.4, -1.0),
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.int32)
        else:
            env_ids = env_ids.to(env.device)

        env.vfloor_z[env_ids] = math_utils.sample_uniform(*vfloor_range, (len(env_ids),), device=env.device)