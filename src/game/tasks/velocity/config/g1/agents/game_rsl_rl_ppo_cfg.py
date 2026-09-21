# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg, RslRlCNNModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg
)
from game.rsl_rl import RslRlGameModelCfg, RslRlGamePpoAlgorithmCfg
from game.tasks.velocity.mdp.symmetry import g1, g1_history


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "Game/game"
    obs_groups = {"actor": ["actor", "actor_map", "actor_history"], "critic": ["critic", "critic_map"]}
    # logger = "wandb"
    # wandb_project = "Game"
    actor = RslRlGameModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        init_weights=(2**0.5, 0, 2**0.5, 0, 2**0.5, 0, 0.01),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=5,
            stride=1,
            dilation=1,
            padding="replicate",
            norm="layer",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=False,
        ),
        dil_cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 16],
            kernel_size=5,
            stride=1,
            dilation=[2, 3],
            padding="replicate",
            norm="layer",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=False,
        ),
        pos_cfg=RslRlGameModelCfg.MLPCfg(
            output_dim=16,
            hidden_dims=[64, 32],
            activation="elu",
            init_weights=(2 ** 0.5, 0, 2 ** 0.5, 0, 1.0),
        ),
        est_cfg=RslRlGameModelCfg.MLPCfg(
            output_dim=3,
            hidden_dims=[512, 256, 128],
            activation="elu",
            init_weights=(2 ** 0.5, 0, 2 ** 0.5, 0, 2 ** 0.5, 0, 1.0),
        ),
        mha_cfg=RslRlGameModelCfg.MHACfg(
            num_heads=16,
            dropout=0.0,
            bias=True,
            kdim=None,
            vdim=None,
            gated_position="sdpa",
        ),
    )
    critic = actor.replace(
        distribution_cfg=None,
        init_weights=(2**0.5, 0, 2**0.5, 0, 2**0.5, 0, 1.0),
    )
    algorithm = RslRlGamePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=4,
        num_mini_batches=3,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=g1_history.compute_symmetric_states
        ),
        share_cnn_encoders=False,
        use_mixed_precision=True,
        est_learning_rate=1.0e-3,
        est_max_grad_norm=10.0,
    )
