# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import RayCaster


"""
Sensors.
"""


def elevation_map(
        env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, size: tuple[int, int], offset: float = 0.825,
        z_noise: float = 0.0,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    relative_pos_w = sensor.data.ray_hits_w.torch - sensor.data.pos_w.torch.unsqueeze(1)  # (N, B, 3)
    sensor_quat_expanded = yaw_quat(sensor.data.quat_w.torch).unsqueeze(1).expand(-1, relative_pos_w.shape[1], -1)
    relative_pos_s = quat_apply_inverse(sensor_quat_expanded, relative_pos_w)

    relative_pos_s = torch.nan_to_num(relative_pos_s, nan=0.0, posinf=3.0, neginf=-3.0)
    # Z-axis height: height = hit_point_z - sensor_height + offset + noise
    relative_pos_s[..., 2] += offset + (torch.rand_like(relative_pos_s[..., 2]) - 0.5) * 2 * z_noise
    return relative_pos_s.reshape(relative_pos_w.shape[0], size[0], size[1], 3).permute(0, 3, 1, 2).contiguous()
