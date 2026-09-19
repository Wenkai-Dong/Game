# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function: str = "{DIR}.mesh_terrains:stepping_stones_height_terrain"

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""

    platform_width: tuple[float, float] | float = 2.0
    """The width of the square platform at the center of the terrain. Defaults to 2.0."""

    border_width: float = 1.0
    """The width of the border around the terrain. Defaults to 1.0."""

    max_yx_angle: tuple[float, float] | bool = False
    """Whether to use Euler angles for rotation. Defaults to False."""


@configclass
class MeshPalletsNarrowTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a star pattern."""

    function: str = "{DIR}.mesh_terrains:pallets_narrow_terrain"

    num_bars: int = MISSING
    """The number of bars per-side the star. Must be greater than 2."""

    bar_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the bars in the star (in m)."""

    platform_width: tuple[float, float] | float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""

    stone_length_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    stone_height_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""


@configclass
class MeshPalletsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a stepping pallets height field terrain."""

    function: str = "{DIR}.mesh_terrains:pallets_terrain"

    pallet_height: float = MISSING
    """The maximum height of the pallet (in m)."""

    pallet_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the pallet (in m)."""

    pallet_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between pallet (in m)."""

    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""

    platform_width: tuple[float, float] | float = 2.0
    """The width of the square platform at the center of the terrain. Defaults to 2.0."""

    border_width: float = 1.0
    """The width of the border around the terrain. Defaults to 1.0."""
