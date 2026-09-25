# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState, MLP

from game.rsl_rl.modules import GatedMHA


class GameModel(MLPModel):
    """CNN-based neural model.

    This model uses one or more convolutional neural network (CNN) encoders to process one or more 2D observation groups
    before passing the resulting latent to an MLP. Any 1D observation groups are directly concatenated with the CNN
    latent and passed to the MLP. 1D observations can be normalized before being passed to the MLP. The output of the
    model can be either deterministic or stochastic, in which case a distribution module is used to sample the outputs.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        dil_cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        pos_cfg: dict[str, dict] | dict[str, Any] | None = None,
        est_cfg: dict[str, dict] | dict[str, Any] | None = None,
        mha_cfg: dict[str, dict] | dict[str, Any] | None = None,
        init_weights: float | tuple[float] = 2**0.5,
    ) -> None:
        # Resolve observation groups and dimensions
        self.obs_set = obs_set
        _, obs_dim_1d = self._get_obs_dim(obs, obs_groups, obs_set)

        # Create Velocity Estimator encoders
        if "actor" in self.obs_set:
            if est_cfg is None:
                raise ValueError("Velocity Estimator configurations must be provided.")
            # Create a est config for each history observation group in case only one is provided
            if not all(isinstance(v, dict) for v in est_cfg.values()):
                est_cfg = {group: est_cfg for group in self.obs_groups_his}
            # Check that the number of configs matches the number of observation groups
            if len(est_cfg) != len(self.obs_groups_his):
                raise ValueError("The number of Velocity Estimator configurations must match the number of history observation groups.")
            # Create Velocity Estimators for each history observation
            ests = {}
            for idx, obs_group in enumerate(self.obs_groups_his):
                est_init = est_cfg[obs_group].pop("init_weights", None)
                ests[obs_group] = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    MLP(input_dim=self.obs_dims_his[idx] * self.obs_channels_his[idx], **est_cfg[obs_group])
                )
                ests[obs_group][-1].init_weights(est_init)

        # Create or validate CNN encoders
        if cnns is not None:
            # Check compatibility if CNNs are provided
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            # Create a cnn config for each 2D observation group in case only one is provided
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
            # Create CNNs for each 2D observation
            cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx] - 2,
                    **cnn_cfg[obs_group],
                )
                cnns[obs_group].init_weights()
            # Check configuration
            if dil_cnn_cfg is None:
                raise ValueError("Dilation CNN configurations must be provided if Dilation CNNs are not shared.")
            # Create a cnn config for each 2D observation group in case only one is provided
            if not all(isinstance(v, dict) for v in dil_cnn_cfg.values()):
                dil_cnn_cfg = {group: dil_cnn_cfg for group in self.obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            if len(dil_cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of Dilation CNN configurations must match the number of 2D observation groups.")
            # Create Dilation CNNs for each 2D observation
            dil_cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                dil_cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx] - 2,
                    **dil_cnn_cfg[obs_group],
                )
                dil_cnns[obs_group].init_weights()
            # Check configuration
            if pos_cfg is None:
                raise ValueError("Positions configurations must be provided if CNNs are not shared.")
            # Create a cnn config for each 2D observation group in case only one is provided
            if not all(isinstance(v, dict) for v in pos_cfg.values()):
                pos_cfg = {group: pos_cfg for group in self.obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            if len(pos_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of Positions configurations must match the number of 2D observation groups.")
            # Create Dilation CNNs for each 2D observation
            positions = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                pos_init = pos_cfg[obs_group].pop("init_weights", None)
                positions[obs_group] = MLP(input_dim=self.obs_channels_2d[idx], **pos_cfg[obs_group])
                positions[obs_group].init_weights(pos_init)

        # Compute latent dimension of the CNNs
        self.cnn_latent_channels = 0
        for cnn in cnns.values():
            self.cnn_latent_channels += int(cnn.output_channels)  # type: ignore
        for dil_cnn in dil_cnns.values():
            self.cnn_latent_channels += int(dil_cnn.output_channels)  # type: ignore
        for position in positions.values():
            self.cnn_latent_channels += int(position[-1].out_features)  # type: ignore

        # Initialize the parent MLP model
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.mlp.init_weights(init_weights)

        # Register CNN encoders
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)
        # Register Dilation CNN encoders
        if isinstance(dil_cnns, nn.ModuleDict):
            self.dil_cnns = dil_cnns
        else:
            self.dil_cnns = nn.ModuleDict(dil_cnns)
        # Register Position encoders
        if isinstance(positions, nn.ModuleDict):
            self.positions = positions
        else:
            self.positions = nn.ModuleDict(positions)

        # Register Estimator encoders
        if "actor" in self.obs_set:
            if isinstance(ests, nn.ModuleDict):
                self.ests = ests
            else:
                self.ests = nn.ModuleDict(ests)

        # Register Pool encoders
        self.pool = nn.Sequential(
            nn.Linear(self.cnn_latent_channels, 1),
            nn.Softmax(dim=1),
        )
        if "actor" in self.obs_set:
            pool_linear_input_dim = self.cnn_latent_channels + obs_dim_1d + 3
        else:
            pool_linear_input_dim = self.cnn_latent_channels + obs_dim_1d
        self.pool_linear = nn.Linear(pool_linear_input_dim , 64)

        # Register MHA encoders
        self.q_norm = nn.LayerNorm(self.cnn_latent_channels)
        self.k_norm = nn.LayerNorm(self.cnn_latent_channels)
        self.mha = GatedMHA(embed_dim=self.cnn_latent_channels, **mha_cfg)
        self.o_norm = nn.LayerNorm(self.cnn_latent_channels)
        self.need_weights = False

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by combining (optional) normalized 1D and CNN-encoded 2D observation groups."""
        # Select 1D observation groups and normalize
        latent_1d = super().get_latent(obs)
        # Estimate Velocity
        if "actor" in self.obs_set:
            latent_est = self.est_vel(obs).detach()
            latent_1d = torch.cat((latent_est, latent_1d), dim=-1)
        # Process 2D observation groups with CNNs
        latent_cnn = torch.cat(
            [self.cnns[obs_group](obs[obs_group][:, -1:, ...]) for obs_group in self.obs_groups_2d], dim=-1
        ) # (N, 32, 13, 18)
        latent_cnn = latent_cnn.flatten(2).permute(0, 2, 1) # (N, 234, 32)
        latent_dil_cnn = torch.cat(
            [self.dil_cnns[obs_group](obs[obs_group][:, -1:, ...]) for obs_group in self.obs_groups_2d], dim=-1
        ) # (N, 16, 13, 18)
        latent_dil_cnn = latent_dil_cnn.flatten(2).permute(0, 2, 1) # (N, 234, 16)
        latent_pos = torch.cat(
            [self.positions[obs_group](obs[obs_group].flatten(2).permute(0, 2, 1)) for obs_group in self.obs_groups_2d], dim=-1
        ) # (N, 234, 16)
        latent_mapping = torch.cat([latent_cnn, latent_dil_cnn, latent_pos], dim=-1)   # (N, 234, 64)
        # pool
        mapping_pool = self.pool(latent_mapping)    # (N, 234, 1)
        mapping_pool = torch.sum(mapping_pool * latent_mapping, dim=1, keepdim=True)   # (N, 1, 64)
        mapping_pool_enc = self.pool_linear(torch.concat([latent_1d.unsqueeze(1), mapping_pool], dim=-1))   # (N, 1, 64)
        # gated_mha
        query = self.q_norm(mapping_pool_enc)
        key = self.k_norm(latent_mapping)
        latent_mha, self.attn_output_weights = self.mha(query, key, latent_mapping, need_weights=self.need_weights)    # (N, 1, 64)
        latent_mha = self.o_norm(latent_mha).flatten(1) # (N, 64)
        return torch.cat([latent_1d, latent_mha], dim=-1)

    def est_vel(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        velocity = torch.cat(
            [self.ests[obs_group](self.obs_normalizer(obs[obs_group])) for obs_group in self.obs_groups_his], dim=-1
        ) # (N, 3)
        return velocity

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchGameModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxGameModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []
        obs_dims_his = []
        obs_channels_his = []
        obs_groups_his = []

        # Iterate through active observation groups and separate 1D and 2D observations
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            elif len(obs[obs_group].shape) == 3:  # B, H, C
                obs_groups_his.append(obs_group)
                obs_dims_his.append(obs[obs_group].shape[-1])
                obs_channels_his.append(obs[obs_group].shape[1])
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided. If this is intentional, use the MLP model instead.")

        # Store active 2D observation groups and dimensions directly as attributes
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d
        # Store active history observation groups and dimensions directly as attributes
        self.obs_dims_his = obs_dims_his
        self.obs_channels_his = obs_channels_his
        self.obs_groups_his = obs_groups_his
        # Return active 1D observation groups and dimension for parent class
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        if "actor" in self.obs_set:
            return self.obs_dim + self.cnn_latent_channels + 3
        else:
            return self.obs_dim + self.cnn_latent_channels

class _TorchGameModel(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: GameModel) -> None:
        """Create a TorchScript-friendly copy of a CNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.dil_cnns = nn.ModuleList([copy.deepcopy(model.dil_cnns[g]) for g in model.obs_groups_2d])
        self.positions = nn.ModuleList([copy.deepcopy(model.positions[g]) for g in model.obs_groups_2d])
        self.ests = nn.ModuleList([copy.deepcopy(model.ests[g]) for g in model.obs_groups_his])
        self.pool = copy.deepcopy(model.pool)
        self.pool_linear = copy.deepcopy(model.pool_linear)
        self.q_norm = copy.deepcopy(model.q_norm)
        self.k_norm = copy.deepcopy(model.k_norm)
        self.mha = copy.deepcopy(model.mha)
        self.o_norm = copy.deepcopy(model.o_norm)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor], obs_his: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and 2D inputs."""
        latent_1d = self.obs_normalizer(obs_1d)

        latent_est_list = []
        for i, est in enumerate(self.ests):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_est_list.append(est(self.obs_normalizer(obs_his[i])))
        latent_est = torch.cat(latent_est_list, dim=-1)
        latent_1d = torch.cat((latent_est, latent_1d), dim=-1)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_cnn_list.append(cnn(obs_2d[i][:, -1:, ...]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1).flatten(2).permute(0, 2, 1)

        latent_dil_cnn_list = []
        for i, dil_cnn in enumerate(self.dil_cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_dil_cnn_list.append(dil_cnn(obs_2d[i][:, -1:, ...]))
        latent_dil_cnn = torch.cat(latent_dil_cnn_list, dim=-1).flatten(2).permute(0, 2, 1)

        latent_pos_list = []
        for i, position in enumerate(self.positions):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_pos_list.append(position(obs_2d[i].flatten(2).permute(0, 2, 1)))
        latent_pos = torch.cat(latent_pos_list, dim=-1)
        latent_mapping = torch.cat([latent_cnn, latent_dil_cnn, latent_pos], dim=-1)   # (N, 234, 64)

        mapping_pool = self.pool(latent_mapping)    # (N, 234, 1)
        mapping_pool = torch.sum(mapping_pool * latent_mapping, dim=1, keepdim=True)   # (N, 1, 64)
        mapping_pool_enc = self.pool_linear(torch.concat([latent_1d.unsqueeze(1), mapping_pool], dim=-1))   # (N, 1, 64)
        # gated_mha
        query = self.q_norm(mapping_pool_enc)
        key = self.k_norm(latent_mapping)
        latent_mha, _ = self.mha(query, key, latent_mapping, need_weights=False)    # (N, 1, 64)
        latent_mha = self.o_norm(latent_mha).flatten(1) # (N, 64)
        latent = torch.cat([latent_1d, latent_mha], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxGameModel(nn.Module):
    """Exportable CNN model for ONNX."""

    def __init__(self, model: GameModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.dil_cnns = nn.ModuleList([copy.deepcopy(model.dil_cnns[g]) for g in model.obs_groups_2d])
        self.positions = nn.ModuleList([copy.deepcopy(model.positions[g]) for g in model.obs_groups_2d])
        self.ests = nn.ModuleList([copy.deepcopy(model.ests[g]) for g in model.obs_groups_his])
        self.pool = copy.deepcopy(model.pool)
        self.pool_linear = copy.deepcopy(model.pool_linear)
        self.q_norm = copy.deepcopy(model.q_norm)
        self.k_norm = copy.deepcopy(model.k_norm)
        self.mha = copy.deepcopy(model.mha)
        self.o_norm = copy.deepcopy(model.o_norm)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim
        self.obs_groups_his = model.obs_groups_his
        self.obs_dims_his = model.obs_dims_his
        self.obs_channels_his = model.obs_channels_his

    def forward(self, obs_1d: torch.Tensor, *obs_else: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        obs_2d = [x for x in obs_else if x.dim() == 4]
        obs_his = [x for x in obs_else if x.dim() == 3]
        latent_1d = self.obs_normalizer(obs_1d)

        latent_est_list = []
        for i, est in enumerate(self.ests):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_est_list.append(est(self.obs_normalizer(obs_his[i])))
        latent_est = torch.cat(latent_est_list, dim=-1)
        latent_1d = torch.cat((latent_est, latent_1d), dim=-1)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_cnn_list.append(cnn(obs_2d[i][:, -1:, ...]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1).flatten(2).permute(0, 2, 1)

        latent_dil_cnn_list = []
        for i, dil_cnn in enumerate(self.dil_cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_dil_cnn_list.append(dil_cnn(obs_2d[i][:, -1:, ...]))
        latent_dil_cnn = torch.cat(latent_dil_cnn_list, dim=-1).flatten(2).permute(0, 2, 1)

        latent_pos_list = []
        for i, position in enumerate(self.positions):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_pos_list.append(position(obs_2d[i].flatten(2).permute(0, 2, 1)))
        latent_pos = torch.cat(latent_pos_list, dim=-1)
        latent_mapping = torch.cat([latent_cnn, latent_dil_cnn, latent_pos], dim=-1)   # (N, 234, 64)

        mapping_pool = self.pool(latent_mapping)    # (N, 234, 1)
        mapping_pool = torch.sum(mapping_pool * latent_mapping, dim=1, keepdim=True)   # (N, 1, 64)
        mapping_pool_enc = self.pool_linear(torch.concat([latent_1d.unsqueeze(1), mapping_pool], dim=-1))   # (N, 1, 64)
        # gated_mha
        query = self.q_norm(mapping_pool_enc)
        key = self.k_norm(latent_mapping)
        latent_mha, _ = self.mha(query, key, latent_mapping, need_weights=False)    # (N, 1, 64)
        latent_mha = self.o_norm(latent_mha).flatten(1) # (N, 64)
        latent = torch.cat([latent_1d, latent_mha], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        dummy_his = []
        for i in range(len(self.obs_groups_his)):
            c = self.obs_dims_his[i]
            h = self.obs_channels_his[i]
            dummy_his.append(torch.zeros(1, h, c))
        return (dummy_1d, *dummy_2d, *dummy_his)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d, *self.obs_groups_his]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
