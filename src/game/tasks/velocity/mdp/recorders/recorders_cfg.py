# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import TYPE_CHECKING

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .recorders import (
        PostStepRootLinearVelocityBaseRecorder,
        PostStepRootAngularVelocityBaseRecorder,
        PostStepCommandBaseVelocityRecorder,
        PreResetTerminationRecorder,
        PostResetSubTerrainIndexRecorder,
        PostResetEpisodeCountRecorder,
        CloseSaveWandBRecorder,
    )

##
# State recorders.
##


@configclass
class PostStepRootLinearVelocityBaseRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term."""

    class_type: type["PostStepRootLinearVelocityBaseRecorder"] | str = "{DIR}.recorders:PostStepRootLinearVelocityBaseRecorder"


@configclass
class PostStepRootAngularVelocityBaseRecorderCfg(RecorderTermCfg):

    class_type: type["PostStepRootAngularVelocityBaseRecorder"] | str = "{DIR}.recorders:PostStepRootAngularVelocityBaseRecorder"


@configclass
class PostStepCommandBaseVelocityRecorderCfg(RecorderTermCfg):

    class_type: type["PostStepCommandBaseVelocityRecorder"] | str = "{DIR}.recorders:PostStepCommandBaseVelocityRecorder"


@configclass
class PreResetTerminationRecorderCfg(RecorderTermCfg):

    class_type: type["PreResetTerminationRecorder"] | str = "{DIR}.recorders:PreResetTerminationRecorder"


@configclass
class PostResetSubTerrainIndexRecorderCfg(RecorderTermCfg):

    class_type: type["PostResetSubTerrainIndexRecorder"] | str = "{DIR}.recorders:PostResetSubTerrainIndexRecorder"


@configclass
class PostResetEpisodeCountRecorderCfg(RecorderTermCfg):

    class_type: type["PostResetEpisodeCountRecorder"] | str = "{DIR}.recorders:PostResetEpisodeCountRecorder"

    max_episode: int = 4


@configclass
class CloseSaveWandBRecorderCfg(RecorderTermCfg):

    class_type: type["CloseSaveWandBRecorder"] | str = "{DIR}.recorders:CloseSaveWandBRecorder"


##
# Recorder manager configurations.
##


@configclass
class VelocityRecorderManagerCfg(RecorderManagerBaseCfg):

    record_post_step_root_linear_velocity_base = PostStepRootLinearVelocityBaseRecorderCfg()
    record_post_step_root_angular_velocity_base = PostStepRootAngularVelocityBaseRecorderCfg()
    record_post_step_command_base_velocity = PostStepCommandBaseVelocityRecorderCfg()
    record_pre_reset_termination = PreResetTerminationRecorderCfg()
    record_post_reset_subterrain_index = PostResetSubTerrainIndexRecorderCfg()
    record_post_reset_episode_count = PostResetEpisodeCountRecorderCfg()
    record_close_save_wandb = CloseSaveWandBRecorderCfg()
