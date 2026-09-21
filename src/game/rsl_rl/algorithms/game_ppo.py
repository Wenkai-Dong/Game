# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    compile_model,
    reduce_gradients_in_buckets,
    resolve_class,
    resolve_obs_groups,
    resolve_optimizer,
)


class GamePPO(PPO):
    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        est_learning_rate: float = 0.001,
        est_max_grad_norm: float = 1.0,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, learning_rate=learning_rate, optimizer=optimizer, **kwargs)

        est_params = list(self.actor.ests.parameters())
        est_ids = {id(p) for p in est_params}
        ppo_params = [
            p for p in chain(self.actor.parameters(), self.critic.parameters())
            if id(p) not in est_ids
        ]

        # re Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(ppo_params, lr=learning_rate)  # type: ignore
        self.est_optimizer = resolve_optimizer(optimizer)(est_params, lr=est_learning_rate)  # type: ignore

        # PPO parameters
        self.est_max_grad_norm = est_max_grad_norm

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self.actor.ests:
            loss_dict["est"] = self._est_update()
        return loss_dict

    def _est_update(self) -> dict[str, float]:
        # RND loss
        mean_est_loss = 0
        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                # Recompute actions log prob and entropy for current batch of transitions
                # Note: We need to do this because we updated the policy with new parameters
                vel_est = self.actor.est_vel(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                )
                vel_real = batch.observations["critic"][:, :3]
                # Velocity Estimator Loss
                est_loss = F.mse_loss(vel_est, vel_real)

            self.est_optimizer.zero_grad()
            est_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.ests.parameters(), self.est_max_grad_norm)
            self.est_optimizer.step()

            mean_est_loss += est_loss.item()
        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_est_loss /= num_updates
        return mean_est_loss


