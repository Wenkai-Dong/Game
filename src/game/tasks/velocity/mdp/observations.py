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
    vfloor_z = env.vfloor_z.unsqueeze(1)  # (N, 1)

    pos_z = sensor.data.pos_w.torch[:, 2:3]  # (N, 1)
    ray_hits_z = sensor.data.ray_hits_w.torch[..., 2]   # (N, B)
    miss = ~torch.isfinite(ray_hits_z)  # (N, B)
    ray_hits_z = torch.where(miss, vfloor_z, ray_hits_z)

    # Z-axis height: height = sensor_height - hit_point_z - offset + noise
    height_scan = pos_z - ray_hits_z - offset
    height_scan += (torch.rand_like(height_scan) - 0.5) * 2 * z_noise  # (N, B)
    height_scan = height_scan.reshape(height_scan.shape[0], size[0], size[1]).unsqueeze(-1)
    elevation_map = sensor.ray_starts.torch.reshape(height_scan.shape[0], size[0], size[1], -1)[..., :2]

    return torch.cat([elevation_map, height_scan], dim=-1).permute(0, 3, 1, 2).contiguous()
