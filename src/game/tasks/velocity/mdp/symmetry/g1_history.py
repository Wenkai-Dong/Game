# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # actor observation group
        # -- original
        obs_aug["actor"][:batch_size] = obs["actor"][:]
        # -- left-right
        obs_aug["actor"][batch_size : 2 * batch_size] = _transform_actor_obs_left_right(obs["actor"])
        # critic observation group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(obs["critic"])
        if obs.get("actor_map") is not None:
            # actor map observation group
            # -- original
            obs_aug["actor_map"][:batch_size] = obs["actor_map"][:]
            # -- left-right
            obs_aug["actor_map"][batch_size: 2 * batch_size] = _transform_map_obs_left_right(obs["actor_map"])
        if obs.get("critic_map") is not None:
            # critic map observation group
            # -- original
            obs_aug["critic_map"][:batch_size] = obs["critic_map"][:]
            # -- left-right
            obs_aug["critic_map"][batch_size: 2 * batch_size] = _transform_map_obs_left_right(obs["critic_map"])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_actor_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # ang vel
    obs[..., :3] = obs[..., :3] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[..., 3:6] = obs[..., 3:6] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[..., 6:9] = obs[..., 6:9] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[..., 9:38] = _switch_g1_joints_left_right(obs[..., 9:38])
    # joint vel
    obs[..., 38:67] = _switch_g1_joints_left_right(obs[..., 38:67])
    # last actions
    obs[..., 67:96] = _switch_g1_joints_left_right(obs[..., 67:96])

    return obs

def _transform_critic_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:,3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 12:41] = _switch_g1_joints_left_right(obs[:, 12:41])
    # joint vel
    obs[:, 41:70] = _switch_g1_joints_left_right(obs[:, 41:70])
    # last actions
    obs[:, 70:99] = _switch_g1_joints_left_right(obs[:, 70:99])

    return obs

def _transform_map_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    obs[:, 2:3, :, :] = torch.flip(obs[:, 2:3, :, :], dims=[2])

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_g1_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'
]

Correspondingly, the joint ordering for the ANYmal robot is:

* LF = left front --> [0, 4, 8]
* LH = left hind --> [1, 5, 9]
* RF = right front --> [2, 6, 10]
* RH = right hind --> [3, 7, 11]
"""


def _switch_g1_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [0, 9, 11, 13, 21, 25]] = joint_data[..., [1, 10, 12, 14, 22, 26]]
    joint_data_switched[..., [3, 6, 15, 17, 19, 23, 27]] = - joint_data[..., [4, 7, 16, 18, 20, 24, 28]]
    # right <-- left
    joint_data_switched[..., [1, 10, 12, 14, 22, 26, ]] = joint_data[..., [0, 9, 11, 13, 21, 25, ]]
    joint_data_switched[..., [4, 7, 16, 18, 20, 24, 28]] = - joint_data[..., [3, 6, 15, 17, 19, 23, 27]]

    # Flip the sign of the HAA joints
    joint_data_switched[..., [8, ]] = joint_data[..., [8, ]]
    joint_data_switched[..., [2, 5]] = - joint_data[..., [2, 5]]

    return joint_data_switched
