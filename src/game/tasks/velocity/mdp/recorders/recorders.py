# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence

import argparse
import torch
import os
import sys

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.datasets import EpisodeData
from isaaclab_rl.entrypoints.common import _physics_backend_name
from isaaclab_rl.utils.wandb import _WANDB_URI_PATTERN, _WANDB_URL_PATTERN


class PostStepRootLinearVelocityBaseRecorder(RecorderTerm):

    def record_post_step(self):
        return "root_lin_vel_b", self._env.scene["robot"].data.root_lin_vel_b.torch


class PostStepRootAngularVelocityBaseRecorder(RecorderTerm):

    def record_post_step(self):
        return "root_ang_vel_b", self._env.scene["robot"].data.root_ang_vel_b.torch


class PostStepCommandBaseVelocityRecorder(RecorderTerm):

    def record_post_step(self):
        return "command_base_velocity", self._env.command_manager.get_command("base_velocity")


class PreResetTerminationRecorder(RecorderTerm):

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        if not hasattr(self, "_startup_done"):
            self._startup_done = True
            return None, None
        reasons = torch.zeros(len(env_ids), dtype=torch.long, device=self._env.device)
        for idx, term_name in enumerate(self._env.termination_manager.active_terms):
            triggered = self._env.termination_manager.get_term(term_name)[env_ids] > 0.5
            reasons[triggered] = idx
        return "Termination", reasons


class PostResetSubTerrainIndexRecorder(RecorderTerm):

    def record_post_reset(self, env_ids: Sequence[int] | None):
        return "sub_terrain", self._env.scene.terrain.terrain_types[env_ids]


class PostResetEpisodeCountRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.episode_count = torch.zeros(self._env.scene.num_envs, dtype=torch.long, device=self._env.device)
        self.max_episode = cfg.max_episode

    def record_pre_reset(self, env_ids):
        manager = self._env.recorder_manager
        counts = self.episode_count[env_ids].tolist()
        for env_id, count in zip(env_ids, counts):
            if count > self.max_episode:
                manager._episodes[env_id] = EpisodeData()
        return None, None

    def record_post_reset(self, env_ids: Sequence[int] | None):
        self.episode_count[env_ids] += 1
        if bool((self.episode_count > self.max_episode).all()):
            raise KeyboardInterrupt
        return "episode_count", self.episode_count[env_ids]


def launch_arg(name: str) -> str | None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(name)
    args, _ = parser.parse_known_args(sys.orig_argv[1:])
    return getattr(args, name.lstrip("-"))


def wandb_run_path_from_checkpoint() -> str | None:
    checkpoint = launch_arg("--checkpoint")
    if not checkpoint:
        return None
    match = _WANDB_URL_PATTERN.match(checkpoint) or _WANDB_URI_PATTERN.match(checkpoint)
    return "{entity}/{project}/{run_id}".format(**match.groupdict()) if match else None


class CloseSaveWandBRecorder(RecorderTerm):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        sim_name = _physics_backend_name(env.cfg.sim.physics)
        vx = env.cfg.commands.base_velocity.ranges.lin_vel_x[0]
        diff = env.cfg.scene.terrain.terrain_generator.difficulty_range[0]

        env.cfg.recorders.dataset_filename = f"{sim_name}_vx{vx}_diff{diff}"
        self._file_path = os.path.abspath(
            os.path.join(env.cfg.recorders.dataset_export_dir_path, env.cfg.recorders.dataset_filename)
        )

    def close(self, file_path):
        file_path = self._file_path
        run_path = wandb_run_path_from_checkpoint()
        if  run_path is None:
            print(f"[INFO] Evaluation dataset saved locally: {os.path.abspath(file_path)}")
            return
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        import wandb
        wandb.Api().run(run_path).upload_file(file_path, root=os.path.dirname(file_path))
