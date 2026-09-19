# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
import game.terrains as game_terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=10.0,
    num_rows=10,
    num_cols=15,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": game_terrain_gen.HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1, noise_range=(-0.075, 0.075), noise_step=0.02, downsampled_scale=0.1, border_width=1.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.3),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.3),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=1.0,
            convert_to_heightfield=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.1,
            gap_width_range=(0.05,0.7),
            platform_width=1.0,
        ),
        "stepping_stones": game_terrain_gen.MeshSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=(0, 0.15),
            stone_width_range=(0.15, 0.5),
            stone_distance_range=(0.05, 0.3),
            platform_width=(2.0, 1.0),
            border_width=1.0,
            max_yx_angle=False,
        ),
        "stepping_stones_angle": game_terrain_gen.MeshSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=(0, 0.15),
            stone_width_range=(0.15, 0.5),
            stone_distance_range=(0.02, 0.3),
            platform_width=(2.0, 1.0),
            border_width=1.0,
            max_yx_angle=(0, 10),
        ),
        "stepping_stones_narrow": game_terrain_gen.MeshPalletsNarrowTerrainCfg(
            proportion=0.1,
            num_bars=5,
            bar_width_range=(0.15, 1.0),
            stone_length_range=(0.15, 1.0),
            stone_distance_range=(0.05, 0.2),
            stone_height_range=(0.0, 0.1),
            platform_width=(2.0, 1.0),
        ),
        "pallets": game_terrain_gen.MeshPalletsTerrainCfg(
            proportion=0.1,
            pallet_height=(0.0, 0.2),
            pallet_width_range=(0.15, 0.5),
            pallet_distance_range=(0.05, 0.5),
            platform_width=(2.0, 1.0),
            border_width=1.0,
        ),
        "pallets_narrow": game_terrain_gen.MeshPalletsNarrowTerrainCfg(
            proportion=0.1,
            num_bars=5,
            bar_width_range=(0.4, 1.0),
            stone_length_range=(0.2, 0.5),
            stone_distance_range=(0.05, 0.3),
            stone_height_range=(0.0, 0.15),
            platform_width=(2.0, 1.0),
        ),
        "pits": terrain_gen.MeshPitTerrainCfg(
            proportion=0.1,
            pit_depth_range=(0.05,0.55),
            double_pit=True,
            platform_width=2.0,
        ),
        "pits_inv": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.1,
            box_height_range=(0.05,0.7),
            double_box=True,
            platform_width=2.0,
        ),
        "beams": terrain_gen.MeshStarTerrainCfg(
            proportion=0.1,
            num_bars=5,
            bar_width_range=(0.1, 1.0),
            bar_height_range=(10.0, 10.0),
            platform_width=2.0,
        ),
    },
)
"""Rough terrains configuration."""
