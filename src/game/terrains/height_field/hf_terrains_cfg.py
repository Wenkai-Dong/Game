# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains import HfRandomUniformTerrainCfg, SubTerrainBaseCfg


"""
Different height field terrain configurations.
"""


@configclass
class HfRandomUniformDifficultyTerrainCfg(HfRandomUniformTerrainCfg):
    """Configuration for a random uniform height field terrain."""

    function: str = "{DIR}.hf_terrains:random_uniform_difficulty_terrain"
