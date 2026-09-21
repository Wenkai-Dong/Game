# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlPpoAlgorithmCfg


#########################
# Model configurations #
#########################


@configclass
class RslRlGameModelCfg(RslRlCNNModelCfg):
    """Configuration for CNN model."""

    class_name: str = "game.rsl_rl.models.game_model:GameModel"

    init_weights: float | tuple[float] = 2 ** 0.5,

    dil_cnn_cfg: RslRlCNNModelCfg.CNNCfg = MISSING

    @configclass
    class MLPCfg:
        output_dim: int = MISSING

        hidden_dims: int | tuple[int] | list[int] = MISSING

        activation: str = MISSING

        init_weights: float | tuple[float] = MISSING

    pos_cfg: MLPCfg = MISSING

    est_cfg: MLPCfg = MISSING

    @configclass
    class MHACfg:
        num_heads: int = MISSING

        dropout: int | tuple[int] | list[int] = MISSING

        bias: bool = MISSING

        kdim: int | None = None

        vdim: int | None = None

        gated_position: str = MISSING

    mha_cfg: MHACfg = MISSING


############################
# Algorithm configurations #
############################


@configclass
class RslRlGamePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    class_name: str = "game.rsl_rl.algorithms.game_ppo:GamePPO"
    """The algorithm class name. Defaults to Game PPO."""

    est_learning_rate: float = MISSING
    """The learning rate for the policy."""

    est_max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    use_mixed_precision: bool = MISSING,
