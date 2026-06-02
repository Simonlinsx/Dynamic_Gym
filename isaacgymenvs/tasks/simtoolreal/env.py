# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime
import io
import math
import os
import random
import tempfile
import time
from copy import copy
from os.path import join
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgym import gymapi, gymtorch, gymutil
import torch
from torch import Tensor
# isort: on

import json

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.simtoolreal.utils import (
    populate_generic_dof_properties,
    populate_dof_properties,
    tolerance_curriculum,
    tolerance_successes_objective,
)
from isaacgymenvs.utils.observation_action_utils_sharpa import (
    OBS_NAMES,
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)
from isaacgymenvs.utils.torch_jit_utils import (
    get_axis_params,
    quat_rotate,
    scale,
    tensor_clamp,
    to_torch,
    torch_rand_float,
    unscale,
    quat_rotate_inverse,
)

DATETIME_STR = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


DEFAULT_DOMINO_OBJECT_VARIANTS = [
    {
        "label": "mug",
        "asset_modelname": "039_mug",
        "asset_model_id": 0,
        "quat": [0.70710678, 0.70710678, 0.0, 0.0],
        "mass": 0.035,
    },
    {
        "label": "cup",
        "asset_modelname": "021_cup",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.025,
    },
    {
        "label": "bottle",
        "asset_modelname": "001_bottle",
        "asset_model_id": 13,
        "quat": [0.70710678, 0.0, 0.0, 0.70710678],
        "mass": 0.025,
        "random_yaw": False,
    },
    {
        "label": "drill",
        "asset_modelname": "030_drill",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.035,
    },
    {
        "label": "pill_bottle",
        "asset_modelname": "080_pillbottle",
        "asset_model_id": 1,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.025,
    },
    {
        "label": "milk_box",
        "asset_modelname": "038_milk-box",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.020,
    },
    {
        "label": "can",
        "asset_modelname": "071_can",
        "asset_model_id": 0,
        "quat": [0.707225, 0.706849, -0.0100455, -0.00982061],
        "mass": 0.020,
    },
    {
        "label": "milk_tea",
        "asset_modelname": "101_milk-tea",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.025,
    },
    {
        "label": "sauce_can",
        "asset_modelname": "105_sauce-can",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.025,
    },
    {
        "label": "tea_box",
        "asset_modelname": "112_tea-box",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.018,
    },
    {
        "label": "bowl",
        "asset_modelname": "002_bowl",
        "asset_model_id": 4,
        "quat": [0.70710678, 0.70710678, 0.0, 0.0],
        "mass": 0.025,
    },
    {
        "label": "plate",
        "asset_modelname": "003_plate",
        "asset_model_id": 0,
        "quat": [0.70710678, 0.70710678, 0.0, 0.0],
        "mass": 0.018,
    },
    {
        "label": "hammer",
        "asset_modelname": "020_hammer",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.035,
    },
    {
        "label": "screwdriver",
        "asset_modelname": "032_screwdriver",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.018,
    },
    {
        "label": "apple",
        "asset_modelname": "035_apple",
        "asset_model_id": 0,
        "quat": [1.0, 0.0, 0.0, 0.0],
        "mass": 0.025,
    },
    {
        "label": "book",
        "asset_modelname": "043_book",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.025,
    },
    {
        "label": "mouse",
        "asset_modelname": "047_mouse",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.018,
    },
    {
        "label": "stapler",
        "asset_modelname": "048_stapler",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.025,
    },
    {
        "label": "dumbbell",
        "asset_modelname": "052_dumbbell",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.035,
    },
    {
        "label": "kettle",
        "asset_modelname": "091_kettle",
        "asset_model_id": 0,
        "quat": [0.5, 0.5, 0.5, 0.5],
        "mass": 0.035,
    },
]

DEFAULT_DYNAMIC_GRASP_AFFORDANCE_SURFACE_OFFSETS = {
    "mug": 0.045,
    "cup": 0.040,
    "bottle": 0.040,
    "drill": 0.050,
    "pill_bottle": 0.035,
    "milk_box": 0.042,
    "can": 0.038,
    "milk_tea": 0.040,
    "sauce_can": 0.038,
    "tea_box": 0.040,
    "bowl": 0.060,
    "plate": 0.065,
    "hammer": 0.055,
    "screwdriver": 0.050,
    "apple": 0.040,
    "book": 0.045,
    "mouse": 0.040,
    "stapler": 0.045,
    "dumbbell": 0.055,
    "kettle": 0.060,
}


def assert_equals(a, b):
    assert a == b, f"a: {a}, b: {b}"


def cfg_pose_to_transform(pose, name: str) -> gymapi.Transform:
    assert len(pose) == 7, f"{name} must be [x, y, z, qx, qy, qz, qw]"
    transform = gymapi.Transform()
    transform.p = gymapi.Vec3(float(pose[0]), float(pose[1]), float(pose[2]))
    transform.r = gymapi.Quat(
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
        float(pose[6]),
    )
    return transform


class SimToolReal(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg
        self.task_mode = cfg["env"].get("taskMode", "goal_pose_reaching")
        valid_task_modes = {"goal_pose_reaching", "dynamic_tabletop_grasp"}
        if self.task_mode not in valid_task_modes:
            raise ValueError(
                f"Unknown taskMode {self.task_mode}; expected one of {valid_task_modes}"
            )
        self.dynamic_tabletop_grasp = self.task_mode == "dynamic_tabletop_grasp"
        self.create_goal_object = bool(cfg["env"].get("createGoalObject", True))

        # Goal related variables
        self.goal_object_indices = []
        self.goal_assets = []
        self.goal_sampling_type = cfg["env"]["goalSamplingType"]
        self.target_volume_region_scale = cfg["env"]["targetVolumeRegionScale"]
        self.delta_goal_distance = cfg["env"]["deltaGoalDistance"]
        self.delta_rotation_degrees = cfg["env"]["deltaRotationDegrees"]

        self.frame_since_restart: int = (
            0  # number of control steps since last restart across all actors
        )

        self.robot_asset_file: str = self.cfg["env"]["asset"]["robot"]
        self.robot_asset_root_cfg = self.cfg["env"].get("robotAssetRoot", None)
        self.robot_embodiment = str(
            self.cfg["env"].get("robotEmbodiment", "sharpa_legacy")
        )
        self.arm_embodiment = str(self.cfg["env"].get("armEmbodiment", "kuka_legacy"))
        self.hand_embodiment = str(
            self.cfg["env"].get("handEmbodiment", "sharpa_legacy")
        )
        self.policy_action_interface = str(
            self.cfg["env"].get("policyActionInterface", "legacy_joint_target")
        )
        valid_action_interfaces = {"legacy_joint_target", "joint_target"}
        if self.policy_action_interface not in valid_action_interfaces:
            raise NotImplementedError(
                f"policyActionInterface={self.policy_action_interface} is not "
                f"implemented; expected one of {sorted(valid_action_interfaces)}."
            )
        self.robot_dof_property_preset = str(
            self.cfg["env"].get(
                "robotDofPropertyPreset",
                "legacy_kuka_sharpa" if self.use_sharpa else "generic_position",
            )
        )

        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]

        self.privileged_actions = self.cfg["env"]["privilegedActions"]
        self.privileged_actions_torque = self.cfg["env"]["privilegedActionsTorque"]

        configured_fingertips = self.cfg["env"].get("fingertipBodyNames", None)
        default_num_fingertips = len(configured_fingertips or [])
        if default_num_fingertips == 0:
            default_num_fingertips = 5 if self.use_sharpa else 4

        self.num_arm_dofs = int(self.cfg["env"].get("armDofs", 7))
        self.num_fingertips = int(
            self.cfg["env"].get("numFingertips", default_num_fingertips)
        )
        self.num_hand_dofs = int(
            self.cfg["env"].get("handDofs", 22 if self.use_sharpa else 16)
        )
        self.num_finger_dofs = (
            None if self.num_fingertips == 0 else self.num_hand_dofs / self.num_fingertips
        )
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs

        self.num_robot_actions = self.num_hand_arm_dofs
        if self.privileged_actions:
            self.num_robot_actions += 3

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        self.hand_actions_penalty_scale = self.cfg["env"]["handActionsPenaltyScale"]
        self.object_lin_vel_penalty_scale = self.cfg["env"]["objectLinVelPenaltyScale"]
        self.object_ang_vel_penalty_scale = self.cfg["env"]["objectAngVelPenaltyScale"]

        # Dynamic tabletop grasping parameters. These are only used when
        # taskMode == "dynamic_tabletop_grasp"; defaults preserve legacy configs.
        self.dynamic_grasp_initial_speed_range = self.cfg["env"].get(
            "dynamicGraspInitialSpeedRange", [0.05, 0.25]
        )
        self.dynamic_grasp_initial_yaw_rate_range = self.cfg["env"].get(
            "dynamicGraspInitialYawRateRange", [-2.0, 2.0]
        )
        self.dynamic_grasp_speed_curriculum = self.cfg["env"].get(
            "dynamicGraspSpeedCurriculum", False
        )
        self.dynamic_grasp_start_speed_range = self.cfg["env"].get(
            "dynamicGraspStartSpeedRange", self.dynamic_grasp_initial_speed_range
        )
        self.dynamic_grasp_start_yaw_rate_range = self.cfg["env"].get(
            "dynamicGraspStartYawRateRange",
            self.dynamic_grasp_initial_yaw_rate_range,
        )
        self.dynamic_grasp_speed_curriculum_steps = self.cfg["env"].get(
            "dynamicGraspSpeedCurriculumSteps", 0
        )
        self.dynamic_grasp_persistent_motion = self.cfg["env"].get(
            "dynamicGraspPersistentMotion", True
        )
        self.dynamic_grasp_bounce_at_workspace = self.cfg["env"].get(
            "dynamicGraspBounceAtWorkspace", True
        )
        self.dynamic_grasp_workspace_x = self.cfg["env"].get(
            "dynamicGraspWorkspaceX", [-0.45, 0.45]
        )
        self.dynamic_grasp_workspace_y = self.cfg["env"].get(
            "dynamicGraspWorkspaceY", [-0.45, 0.45]
        )
        self.dynamic_grasp_velocity_match_scale = self.cfg["env"].get(
            "dynamicGraspVelocityMatchScale", 0.25
        )
        self.dynamic_grasp_near_scale = self.cfg["env"].get(
            "dynamicGraspNearScale", 0.20
        )
        self.dynamic_grasp_enclosure_distance_scale = self.cfg["env"].get(
            "dynamicGraspEnclosureDistanceScale", 0.12
        )
        self.dynamic_grasp_intercept_lead_time = self.cfg["env"].get(
            "dynamicGraspInterceptLeadTime", 0.0
        )
        self.dynamic_grasp_intercept_distance_scale = self.cfg["env"].get(
            "dynamicGraspInterceptDistanceScale", 0.20
        )
        self.dynamic_grasp_pregrasp_ahead_distance = self.cfg["env"].get(
            "dynamicGraspPregraspAheadDistance", 0.0
        )
        self.dynamic_grasp_pregrasp_distance_scale = self.cfg["env"].get(
            "dynamicGraspPregraspDistanceScale", 0.20
        )
        self.dynamic_grasp_stable_fingertip_distance = self.cfg["env"].get(
            "dynamicGraspStableFingertipDistance", 0.12
        )
        self.dynamic_grasp_stable_object_palm_vel = self.cfg["env"].get(
            "dynamicGraspStableObjectPalmVel", 0.20
        )
        self.dynamic_grasp_success_steps_required = self.cfg["env"].get(
            "dynamicGraspSuccessSteps", 30
        )
        self.dynamic_grasp_velocity_match_rew_scale = self.cfg["env"].get(
            "dynamicGraspVelocityMatchRewScale", 2.0
        )
        self.dynamic_grasp_intercept_rew_scale = self.cfg["env"].get(
            "dynamicGraspInterceptRewScale", 0.0
        )
        self.dynamic_grasp_pregrasp_alignment_rew_scale = self.cfg["env"].get(
            "dynamicGraspPregraspAlignmentRewScale", 0.0
        )
        self.dynamic_grasp_enclosure_rew_scale = self.cfg["env"].get(
            "dynamicGraspEnclosureRewScale", 5.0
        )
        self.dynamic_grasp_lifted_enclosure_rew_scale = self.cfg["env"].get(
            "dynamicGraspLiftedEnclosureRewScale",
            self.dynamic_grasp_enclosure_rew_scale,
        )
        self.dynamic_grasp_lift_progress_rew_scale = self.cfg["env"].get(
            "dynamicGraspLiftProgressRewScale", 0.0
        )
        self.dynamic_grasp_stable_hold_rew_scale = self.cfg["env"].get(
            "dynamicGraspStableHoldRewScale", 20.0
        )
        self.dynamic_grasp_success_bonus = self.cfg["env"].get(
            "dynamicGraspSuccessBonus", self.cfg["env"].get("reachGoalBonus", 1000.0)
        )
        self.dynamic_grasp_stable_progress_rew_scale = self.cfg["env"].get(
            "dynamicGraspStableProgressRewScale", 0.0
        )
        self.dynamic_grasp_lifted_rel_vel_rew_scale = self.cfg["env"].get(
            "dynamicGraspLiftedRelVelRewScale", 0.0
        )
        self.dynamic_grasp_lifted_rel_vel_scale = self.cfg["env"].get(
            "dynamicGraspLiftedRelVelScale",
            max(self.dynamic_grasp_stable_object_palm_vel, 1e-3),
        )
        self.dynamic_grasp_lifted_centering_rew_scale = self.cfg["env"].get(
            "dynamicGraspLiftedCenteringRewScale", 0.0
        )
        self.dynamic_grasp_lifted_centering_distance_scale = self.cfg["env"].get(
            "dynamicGraspLiftedCenteringDistanceScale", 0.12
        )
        self.dynamic_grasp_lifted_height_hold_rew_scale = self.cfg["env"].get(
            "dynamicGraspLiftedHeightHoldRewScale", 0.0
        )
        self.dynamic_grasp_lifted_enclosure_vel_rew_scale = self.cfg["env"].get(
            "dynamicGraspLiftedEnclosureVelRewScale", 0.0
        )
        self.dynamic_grasp_stable_counter_decay = int(
            self.cfg["env"].get("dynamicGraspStableCounterDecay", 0)
        )
        self.dynamic_grasp_post_contact_rel_vel_penalty_scale = self.cfg["env"].get(
            "dynamicGraspPostContactRelVelPenaltyScale", 0.0
        )
        self.dynamic_grasp_post_contact_rel_vel_margin = self.cfg["env"].get(
            "dynamicGraspPostContactRelVelMargin",
            self.dynamic_grasp_stable_object_palm_vel,
        )
        self.dynamic_grasp_push_away_penalty_scale = self.cfg["env"].get(
            "dynamicGraspPushAwayPenaltyScale", 0.0
        )
        self.dynamic_grasp_push_away_speed_margin = self.cfg["env"].get(
            "dynamicGraspPushAwaySpeedMargin", 0.35
        )
        self.dynamic_grasp_push_away_height_margin = self.cfg["env"].get(
            "dynamicGraspPushAwayHeightMargin", 0.16
        )
        self.dynamic_grasp_pre_contact_slow_rew_scale = self.cfg["env"].get(
            "dynamicGraspPreContactSlowRewScale", 0.0
        )
        self.dynamic_grasp_pre_contact_slow_distance = self.cfg["env"].get(
            "dynamicGraspPreContactSlowDistance", 0.18
        )
        self.dynamic_grasp_pre_contact_slow_vel_scale = self.cfg["env"].get(
            "dynamicGraspPreContactSlowVelScale", 0.18
        )
        self.dynamic_grasp_controlled_contact_rew_scale = self.cfg["env"].get(
            "dynamicGraspControlledContactRewScale", 0.0
        )
        self.dynamic_grasp_controlled_contact_vel_scale = self.cfg["env"].get(
            "dynamicGraspControlledContactVelScale", 0.20
        )
        self.dynamic_grasp_impact_penalty_scale = self.cfg["env"].get(
            "dynamicGraspImpactPenaltyScale", 0.0
        )
        self.dynamic_grasp_impact_rel_vel_margin = self.cfg["env"].get(
            "dynamicGraspImpactRelVelMargin",
            self.dynamic_grasp_post_contact_rel_vel_margin,
        )
        self.dynamic_grasp_kick_penalty_scale = self.cfg["env"].get(
            "dynamicGraspKickPenaltyScale", 0.0
        )
        self.dynamic_grasp_kick_speed_margin = self.cfg["env"].get(
            "dynamicGraspKickSpeedMargin", 0.08
        )
        self.dynamic_grasp_dropped_penalty_scale = self.cfg["env"].get(
            "dynamicGraspDroppedPenaltyScale", 0.0
        )
        self.dynamic_grasp_timeout_penalty_scale = self.cfg["env"].get(
            "dynamicGraspTimeoutPenaltyScale", 0.0
        )
        self.dynamic_grasp_safe_action_delta_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspSafeActionDeltaPenaltyScale", 0.0)
        )
        self.dynamic_grasp_safe_arm_target_delta_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspSafeArmTargetDeltaPenaltyScale", 0.0)
        )
        self.dynamic_grasp_safe_arm_target_accel_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspSafeArmTargetAccelPenaltyScale", 0.0)
        )
        self.dynamic_grasp_safe_hand_target_delta_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspSafeHandTargetDeltaPenaltyScale", 0.0)
        )
        self.dynamic_grasp_pregrasp_hold_rew_scale = float(
            self.cfg["env"].get("dynamicGraspPregraspHoldRewScale", 0.0)
        )
        self.dynamic_grasp_pregrasp_hold_vel_scale = float(
            self.cfg["env"].get("dynamicGraspPregraspHoldVelScale", 0.12)
        )
        self.dynamic_grasp_pregrasp_ready_distance = float(
            self.cfg["env"].get("dynamicGraspPregraspReadyDistance", 0.08)
        )
        self.dynamic_grasp_early_contact_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspEarlyContactPenaltyScale", 0.0)
        )
        self.dynamic_grasp_pre_contact_rel_vel_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspPreContactRelVelPenaltyScale", 0.0)
        )
        self.dynamic_grasp_pre_contact_rel_vel_margin = float(
            self.cfg["env"].get("dynamicGraspPreContactRelVelMargin", 0.18)
        )
        self.dynamic_grasp_early_contact_grace_distance = float(
            self.cfg["env"].get("dynamicGraspEarlyContactGraceDistance", 0.0)
        )
        self.dynamic_grasp_ready_enclosure_rew_scale = float(
            self.cfg["env"].get("dynamicGraspReadyEnclosureRewScale", 0.0)
        )
        self.dynamic_grasp_ready_grasp_quality_rew_scale = float(
            self.cfg["env"].get("dynamicGraspReadyGraspQualityRewScale", 0.0)
        )
        self.dynamic_grasp_ready_contact_rew_scale = float(
            self.cfg["env"].get("dynamicGraspReadyContactRewScale", 0.0)
        )
        self.dynamic_grasp_ready_contact_vel_scale = float(
            self.cfg["env"].get(
                "dynamicGraspReadyContactVelScale",
                self.dynamic_grasp_controlled_contact_vel_scale,
            )
        )
        self.dynamic_grasp_surface_contact_distance = float(
            self.cfg["env"].get(
                "dynamicGraspSurfaceContactDistance",
                self.dynamic_grasp_stable_fingertip_distance,
            )
        )
        self.dynamic_grasp_min_finger_contacts = int(
            self.cfg["env"].get("dynamicGraspMinFingerContacts", 2)
        )
        self.dynamic_grasp_min_non_thumb_contacts = int(
            self.cfg["env"].get("dynamicGraspMinNonThumbContacts", 1)
        )
        self.dynamic_grasp_opposing_dot_threshold = float(
            self.cfg["env"].get("dynamicGraspOpposingDotThreshold", 0.0)
        )
        self.dynamic_grasp_require_true_grasp_for_success = bool(
            self.cfg["env"].get("dynamicGraspRequireTrueGraspForSuccess", False)
        )
        self.dynamic_grasp_gate_lift_reward_by_grasp_quality = bool(
            self.cfg["env"].get("dynamicGraspGateLiftRewardByGraspQuality", False)
        )
        self.dynamic_grasp_lift_reward_min_grasp_quality_multiplier = float(
            self.cfg["env"].get(
                "dynamicGraspLiftRewardMinGraspQualityMultiplier", 1.0
            )
        )
        self.dynamic_grasp_true_grasp_quality_rew_scale = float(
            self.cfg["env"].get("dynamicGraspTrueGraspQualityRewScale", 0.0)
        )
        self.dynamic_grasp_lifted_true_grasp_rew_scale = float(
            self.cfg["env"].get("dynamicGraspLiftedTrueGraspRewScale", 0.0)
        )
        self.dynamic_grasp_quality_lift_progress_rew_scale = float(
            self.cfg["env"].get("dynamicGraspQualityLiftProgressRewScale", 0.0)
        )
        self.dynamic_grasp_opposing_contact_rew_scale = float(
            self.cfg["env"].get("dynamicGraspOpposingContactRewScale", 0.0)
        )
        self.dynamic_grasp_scoop_lift_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspScoopLiftPenaltyScale", 0.0)
        )
        self.dynamic_grasp_palm_only_lift_penalty_scale = float(
            self.cfg["env"].get("dynamicGraspPalmOnlyLiftPenaltyScale", 0.0)
        )
        self.dynamic_grasp_palm_only_lift_dist = float(
            self.cfg["env"].get("dynamicGraspPalmOnlyLiftDist", 0.12)
        )
        self.dynamic_grasp_use_affordance_prior = bool(
            self.cfg["env"].get("dynamicGraspUseAffordancePrior", False)
        )
        self.dynamic_grasp_affordance_mode = str(
            self.cfg["env"].get("dynamicGraspAffordanceMode", "heuristic_future_side")
        )
        self.dynamic_grasp_affordance_lead_time = float(
            self.cfg["env"].get(
                "dynamicGraspAffordanceLeadTime",
                self.dynamic_grasp_intercept_lead_time,
            )
        )
        self.dynamic_grasp_affordance_surface_offset = float(
            self.cfg["env"].get("dynamicGraspAffordanceSurfaceOffset", 0.045)
        )
        self.dynamic_grasp_affordance_vertical_offset = float(
            self.cfg["env"].get("dynamicGraspAffordanceVerticalOffset", 0.0)
        )
        self.dynamic_grasp_affordance_min_speed_for_confidence = float(
            self.cfg["env"].get("dynamicGraspAffordanceMinSpeedForConfidence", 0.04)
        )
        self.dynamic_grasp_affordance_obs_noise_std = float(
            self.cfg["env"].get("dynamicGraspAffordanceObsNoiseStd", 0.0)
        )
        self.dynamic_grasp_affordance_axis_noise_std = float(
            self.cfg["env"].get("dynamicGraspAffordanceAxisNoiseStd", 0.0)
        )
        self.dynamic_grasp_affordance_dropout_prob = float(
            self.cfg["env"].get("dynamicGraspAffordanceDropoutProb", 0.0)
        )
        self.dynamic_grasp_affordance_target_rew_scale = float(
            self.cfg["env"].get("dynamicGraspAffordanceTargetRewScale", 0.0)
        )
        self.dynamic_grasp_affordance_surface_offsets_by_label = dict(
            DEFAULT_DYNAMIC_GRASP_AFFORDANCE_SURFACE_OFFSETS
        )
        self.dynamic_grasp_affordance_surface_offsets_by_label.update(
            {
                str(label): float(offset)
                for label, offset in self.cfg["env"]
                .get("dynamicGraspAffordanceSurfaceOffsetsByLabel", {})
                .items()
            }
        )
        self.object_pointcloud_num_points = int(
            self.cfg["env"].get("objectPointCloudNumPoints", 0)
        )
        self.object_pointcloud_noise_std = float(
            self.cfg["env"].get("objectPointCloudNoiseStd", 0.0)
        )
        self.object_pointcloud_dropout_prob = float(
            self.cfg["env"].get("objectPointCloudDropoutProb", 0.0)
        )
        self.object_pointcloud_use_partial_view = bool(
            self.cfg["env"].get("objectPointCloudUsePartialView", False)
        )
        self.object_pointcloud_candidate_multiplier = max(
            1, int(self.cfg["env"].get("objectPointCloudCandidateMultiplier", 1))
        )
        self.object_pointcloud_candidate_num_points = self.object_pointcloud_num_points
        if self.object_pointcloud_use_partial_view:
            self.object_pointcloud_candidate_num_points *= (
                self.object_pointcloud_candidate_multiplier
            )
        self.object_pointcloud_visible_dot_threshold = float(
            self.cfg["env"].get("objectPointCloudVisibleDotThreshold", 0.0)
        )
        self.object_pointcloud_camera_pos_cfg = self.cfg["env"].get(
            "objectPointCloudCameraPos", [0.45, -0.75, 0.75]
        )
        self.object_pointcloud_camera_pos_noise_std = float(
            self.cfg["env"].get("objectPointCloudCameraPosNoiseStd", 0.0)
        )
        self.object_pointcloud_depth_noise_std = float(
            self.cfg["env"].get("objectPointCloudDepthNoiseStd", 0.0)
        )
        self.object_pointcloud_centroid_from_points = bool(
            self.cfg["env"].get("objectPointCloudCentroidFromPoints", False)
        )
        self.object_pointcloud_outlier_prob = float(
            self.cfg["env"].get("objectPointCloudOutlierProb", 0.0)
        )
        self.object_pointcloud_outlier_scale = float(
            self.cfg["env"].get("objectPointCloudOutlierScale", 0.0)
        )
        self.object_pointcloud_randomize_order = bool(
            self.cfg["env"].get("objectPointCloudRandomizeOrder", False)
        )
        self.object_pointcloud_velocity_from_centroid = bool(
            self.cfg["env"].get("objectPointCloudVelocityFromCentroid", False)
        )
        self.object_pointcloud_velocity_noise_std = float(
            self.cfg["env"].get("objectPointCloudVelocityNoiseStd", 0.0)
        )
        self.object_pointcloud_velocity_clip = float(
            self.cfg["env"].get("objectPointCloudVelocityClip", 2.0)
        )
        self.object_pointcloud_use_mesh_surface = bool(
            self.cfg["env"].get("objectPointCloudUseMeshSurface", False)
        )
        self.domino_object_asset_root = self.cfg["env"].get(
            "dominoObjectAssetRoot", "/data1/linsixu/DOMINO/assets/objects"
        )
        self.domino_object_collision_source = self.cfg["env"].get(
            "dominoObjectCollisionSource", "collision"
        )
        self.domino_object_visual_source = self.cfg["env"].get(
            "dominoObjectVisualSource", "visual"
        )
        self.domino_object_need_vhacd = bool(
            self.cfg["env"].get("dominoObjectNeedVhacd", True)
        )
        self.domino_object_table_top_offset = float(
            self.cfg["env"].get("dominoObjectTableTopOffset", 0.15)
        )
        self.domino_object_spawn_z_margin = float(
            self.cfg["env"].get("dominoObjectSpawnZMargin", 0.004)
        )
        self.domino_object_random_yaw_default = bool(
            self.cfg["env"].get("dominoObjectRandomYaw", True)
        )
        self.domino_object_variants = self.cfg["env"].get(
            "dominoObjectVariants", DEFAULT_DOMINO_OBJECT_VARIANTS
        )
        self.log_dynamic_object_pool_metrics = bool(
            self.cfg["env"].get("logDynamicObjectPoolMetrics", True)
        )
        self.dynamic_object_pool_log_interval = max(
            1, int(self.cfg["env"].get("dynamicObjectPoolLogInterval", 64))
        )

        self.initial_tolerance = self.cfg["env"]["successTolerance"]
        self.target_tolerance = self.cfg["env"]["targetSuccessTolerance"]
        self.success_tolerance = self.initial_tolerance
        self.tolerance_curriculum_increment = self.cfg["env"][
            "toleranceCurriculumIncrement"
        ]
        self.tolerance_curriculum_interval = self.cfg["env"][
            "toleranceCurriculumInterval"
        ]

        self.save_states = self.cfg["env"]["saveStates"]
        self.save_states_filename = self.cfg["env"]["saveStatesFile"]

        self.should_load_initial_states = self.cfg["env"]["loadInitialStates"]
        self.load_states_filename = self.cfg["env"]["loadStatesFile"]
        self.initial_root_state_tensors = self.initial_dof_state_tensors = None
        self.initial_state_idx = self.num_initial_states = 0

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.randomize_object_rotation = self.cfg["env"]["randomizeObjectRotation"]
        self.reset_dof_pos_noise_fingers = self.cfg["env"][
            "resetDofPosRandomIntervalFingers"
        ]
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        self.force_only_when_lifted = self.cfg["env"].get("forceOnlyWhenLifted", False)

        self.torque_scale = self.cfg["env"].get("torqueScale", 0.0)
        self.torque_prob_range = self.cfg["env"].get("torqueProbRange", [0.001, 0.1])
        self.torque_decay = self.cfg["env"].get("torqueDecay", 0.99)
        self.torque_decay_interval = self.cfg["env"].get("torqueDecayInterval", 0.08)
        self.torque_only_when_lifted = self.cfg["env"].get(
            "torqueOnlyWhenLifted", False
        )

        self.lin_vel_impulse_prob_range = self.cfg["env"].get(
            "linVelImpulseProbRange", [0.001, 0.1]
        )
        self.lin_vel_impulse_scale = self.cfg["env"].get("linVelImpulseScale", 0.0)
        self.lin_vel_impulse_only_when_lifted = self.cfg["env"].get(
            "linVelImpulseOnlyWhenLifted", False
        )

        self.ang_vel_impulse_prob_range = self.cfg["env"].get(
            "angVelImpulseProbRange", [0.001, 0.1]
        )
        self.ang_vel_impulse_scale = self.cfg["env"].get("angVelImpulseScale", 0.0)
        self.ang_vel_impulse_only_when_lifted = self.cfg["env"].get(
            "angVelImpulseOnlyWhenLifted", False
        )

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.success_steps: int = self.cfg["env"]["successSteps"]

        # 1.0 means keypoints correspond to the corners of the object
        # larger values help the agent to prioritize rotation matching
        self.keypoint_scale = self.cfg["env"]["keypointScale"]

        # size of the object (i.e. cube) before scaling
        self.object_base_size = self.cfg["env"]["objectBaseSize"]

        # whether to sample random object dimensions
        self.with_dof_force_sensors = False
        # create fingertip force-torque sensors
        self.with_fingertip_force_sensors = False
        self.with_table_force_sensor = self.cfg["env"]["withTableForceSensor"]

        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (self.control_freq_inv * self.sim_params.dt))
            )
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.asset_files_dict = {
            "table": self.cfg["env"]["asset"]["table"],
        }

        self.keypoints_offsets = self._object_keypoint_offsets()

        self.num_keypoints = len(self.keypoints_offsets)
        (
            self.object_pointcloud_unit_points_np,
            self.object_pointcloud_unit_normals_np,
        ) = self._object_pointcloud_unit_surface(
            self.object_pointcloud_candidate_num_points
        )

        self.fingertips = [
            "index_link_3",
            "middle_link_3",
            "ring_link_3",
            "thumb_link_3",
        ]
        self.fingertip_offsets = np.array(
            [[0.05, 0.005, 0], [0.05, 0.005, 0], [0.05, 0.005, 0], [0.06, 0.005, 0]],
            dtype=np.float32,
        )
        configured_fingertips = self.cfg["env"].get("fingertipBodyNames", None)
        configured_fingertip_offsets = self.cfg["env"].get("fingertipOffsets", None)

        if self.use_sharpa:
            if self.use_right_sharpa:
                self.fingertips = [
                    "right_index_DP",
                    "right_middle_DP",
                    "right_ring_DP",
                    "right_thumb_DP",
                    "right_pinky_DP",
                ]
            elif self.use_left_sharpa:
                self.fingertips = [
                    "left_index_DP",
                    "left_middle_DP",
                    "left_ring_DP",
                    "left_thumb_DP",
                    "left_pinky_DP",
                ]
            else:
                raise ValueError(f"Unknown sharpa type: {self.use_sharpa}")
            self.fingertip_offsets = np.array(
                [
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                ],
                dtype=np.float32,
                # [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
            )
        if configured_fingertips is not None:
            self.fingertips = [str(name) for name in configured_fingertips]
            if configured_fingertip_offsets is None:
                configured_fingertip_offsets = [[0.0, 0.0, 0.0]] * len(
                    self.fingertips
                )
            self.fingertip_offsets = np.asarray(
                configured_fingertip_offsets,
                dtype=np.float32,
            )
            assert self.fingertip_offsets.shape == (len(self.fingertips), 3), (
                "fingertipOffsets must be a list of [x, y, z] values matching "
                f"fingertipBodyNames; got {self.fingertip_offsets.shape} for "
                f"{len(self.fingertips)} fingertips"
            )
        self.thumb_fingertip_idx = next(
            (idx for idx, name in enumerate(self.fingertips) if "thumb" in name),
            min(3, len(self.fingertips) - 1),
        )
        self.non_thumb_fingertip_indices = [
            idx for idx in range(len(self.fingertips)) if idx != self.thumb_fingertip_idx
        ]
        self.palm_body_name = str(
            self.cfg["env"].get(
                "palmBodyName",
                "iiwa14_link_7" if self.use_sharpa else "palm",
            )
        )
        self.palm_offset = np.asarray(
            self.cfg["env"].get("palmOffset", [-0.00, -0.02, 0.16]),
            dtype=np.float32,
        )
        assert self.palm_offset.shape == (3,), (
            f"palmOffset must be a length-3 list, got {self.palm_offset.shape}"
        )

        assert self.num_fingertips == len(self.fingertips)

        # can be only "full_state" or "asymmetric"
        self.obs_type = self.cfg["env"]["observationType"]

        if self.obs_type not in ["full_state", "asymmetric"]:
            raise Exception("Unknown type of observations!")

        print("Obs type:", self.obs_type)

        self.obs_type_size_dict = {
            "joint_pos": self.num_hand_arm_dofs,
            "joint_vel": self.num_hand_arm_dofs,
            "prev_action_targets": self.num_hand_arm_dofs,
            "palm_pos": 3,
            "palm_rot": 4,
            "palm_vel": 6,
            "object_rot": 4,
            "object_vel": 6,
            "object_vel_rel_palm": 6,
            "fingertip_pos_rel_palm": 3 * self.num_fingertips,
            "keypoints_rel_palm": 3 * self.num_keypoints,
            "keypoints_rel_goal": 3 * self.num_keypoints,
            "predicted_keypoints_rel_palm": 3 * self.num_keypoints,
            "object_pointcloud_rel_palm": 3 * self.object_pointcloud_num_points,
            "object_pointcloud_centroid_rel_palm": 3,
            "object_pointcloud_vel_rel_palm": 3,
            "object_pointcloud_tracking_confidence": 1,
            "affordance_pos_rel_palm": 3,
            "future_affordance_pos_rel_palm": 3,
            "affordance_axis_rel_palm": 3,
            "affordance_confidence": 1,
            "object_scales": 3,
            "closest_keypoint_max_dist": 1,
            "closest_fingertip_dist": self.num_fingertips,
            "lifted_object": 1,
            "progress": 1,
            "successes": 1,
            "reward": 1,
        }

        self.state_list = self.cfg["env"]["stateList"]
        self.obs_list = self.cfg["env"]["obsList"]

        # assert that all obs in state_list and obs_list are keys of self.obs_type_size_dict
        for obs_type in self.state_list + self.obs_list:
            assert obs_type in self.obs_type_size_dict, (
                f"Obs type {obs_type} not found in obs_type_size_dict"
            )

        # assert that all obs in obs_list are also in state_list
        for obs_type in self.obs_list:
            assert obs_type in self.state_list, (
                f"Obs type {obs_type} not found in state_list but is in obs_list"
            )

        self.full_state_size = sum(
            [self.obs_type_size_dict[obs_type] for obs_type in self.state_list]
        )
        self.full_obs_size = sum(
            [self.obs_type_size_dict[obs_type] for obs_type in self.obs_list]
        )

        self.up_axis = "z"

        self.fingertip_obs = True

        self.cfg["env"]["numStates"] = self.full_state_size
        self.cfg["env"]["numObservations"] = self.full_obs_size
        self.cfg["env"]["numActions"] = self.num_robot_actions

        self.cfg["device_type"] = (
            sim_device.split(":")[0] if sim_device.find(":") != -1 else sim_device
        )
        self.cfg["device_id"] = (
            int(sim_device.split(":")[1]) if sim_device.find(":") != -1 else 0
        )
        self.cfg["headless"] = headless

        # Must subscribe to keyboard events before calling super().__init__()
        self._subscribe_to_keyboard_events()

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        self.object_pointcloud_unit_points = to_torch(
            self.object_pointcloud_unit_points_np,
            dtype=torch.float,
            device=self.device,
        )
        self.object_pointcloud_unit_normals = to_torch(
            self.object_pointcloud_unit_normals_np,
            dtype=torch.float,
            device=self.device,
        )
        self.object_pointcloud_camera_pos = to_torch(
            self.object_pointcloud_camera_pos_cfg,
            dtype=torch.float,
            device=self.device,
        )
        self.prev_object_pointcloud_centroid_rel_palm = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self.object_pointcloud_velocity_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Index of environment to view in viewer and camera
        self.index_to_view = 0

        # Camera position and target for viewer
        cam_target = gymapi.Vec3(0.0, 0.0, 0.53)
        cam_pos = cam_target + gymapi.Vec3(0.0, -1.0, 0.5)
        if self.viewer is not None:
            self.gym.viewer_camera_look_at(
                self.viewer, self.envs[self.index_to_view], cam_pos, cam_target
            )

        # Init camera for wandb logging
        self._initialize_camera_sensor(cam_pos=cam_pos, cam_target=cam_target)
        self._modify_render_settings_if_headless()

        # volume to sample target position from
        target_volume_origin = np.array([0, 0.05, 0.8], dtype=np.float32)
        target_volume_extent = np.array(
            [[-0.4, 0.4], [-0.05, 0.3], [-0.12, 0.25]], dtype=np.float32
        )
        if (
            self.cfg["env"]["targetVolumeMins"] is not None
            and self.cfg["env"]["targetVolumeMaxs"] is not None
        ):
            assert (
                len(self.cfg["env"]["targetVolumeMins"])
                == len(self.cfg["env"]["targetVolumeMaxs"])
                == 3
            ), "targetVolumeMins and targetVolumeMaxs must be 3-element lists"
            mins = np.array(self.cfg["env"]["targetVolumeMins"], dtype=np.float32)
            maxs = np.array(self.cfg["env"]["targetVolumeMaxs"], dtype=np.float32)
            new_target_volume_origin = (mins + maxs) / 2
            new_target_volume_range = maxs - mins
            new_target_volume_extent = np.stack(
                [-new_target_volume_range / 2, new_target_volume_range / 2], axis=1
            )
            assert (
                target_volume_origin.shape == new_target_volume_origin.shape == (3,)
            ), (
                f"target_volume_origin.shape: {target_volume_origin.shape}, new_target_volume_origin.shape: {new_target_volume_origin.shape}"
            )
            assert (
                target_volume_extent.shape == new_target_volume_extent.shape == (3, 2)
            ), (
                f"target_volume_extent.shape: {target_volume_extent.shape}, new_target_volume_extent.shape: {new_target_volume_extent.shape}"
            )
            target_volume_origin = new_target_volume_origin
            target_volume_extent = new_target_volume_extent

        self.target_volume_origin = (
            torch.from_numpy(target_volume_origin).to(self.device).float()
        )
        self.target_volume_extent = (
            torch.from_numpy(target_volume_extent).to(self.device).float()
        )

        # Scale the target volume extent by the target volume region scale
        self.target_volume_extent = (
            self.target_volume_extent * self.target_volume_region_scale
        )

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.with_fingertip_force_sensors or self.with_table_force_sensor:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
                # self.num_envs, self.num_fingertips * 6
                self.num_envs,
                -1,
                6,
            )
            print(f"force_sensor_tensor: {self.force_sensor_tensor.shape}")
            self.gym.refresh_force_sensor_tensor(self.sim)
            if self.device == "cpu":
                raise ValueError(
                    "Force sensors not supported on CPU, they only gives 0s on CPU"
                )

        if self.with_dof_force_sensors:
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_hand_arm_dofs
            )
            self.gym.refresh_dof_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.hand_arm_default_dof_pos = torch.zeros(
            self.num_hand_arm_dofs, dtype=torch.float, device=self.device
        )

        desired_arm_pos = torch.tensor(
            [-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358]
        )  # pose v1
        if self.use_sharpa:
            desired_arm_pos = torch.tensor(
                [-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308]
            )  # same as above but 60 deg offset for the mount
        if "defaultArmDofPos" in self.cfg["env"]:
            desired_arm_pos = torch.tensor(
                self.cfg["env"]["defaultArmDofPos"],
                dtype=torch.float,
            )
        assert desired_arm_pos.shape == (self.num_arm_dofs,), (
            f"defaultArmDofPos must have length {self.num_arm_dofs}, got "
            f"{desired_arm_pos.shape}"
        )

        START_HIGHER = self.cfg["env"]["startArmHigher"]
        if START_HIGHER:
            desired_arm_pos[1] -= np.deg2rad(10)
            desired_arm_pos[3] += np.deg2rad(10)

        self.hand_arm_default_dof_pos[: self.num_arm_dofs] = desired_arm_pos.to(
            self.device
        )
        if "defaultHandDofPos" in self.cfg["env"]:
            desired_hand_pos = torch.tensor(
                self.cfg["env"]["defaultHandDofPos"],
                dtype=torch.float,
                device=self.device,
            )
            assert desired_hand_pos.shape == (self.num_hand_dofs,), (
                f"defaultHandDofPos must have length {self.num_hand_dofs}, got "
                f"{desired_hand_pos.shape}"
            )
            self.hand_arm_default_dof_pos[
                self.num_arm_dofs : self.num_hand_arm_dofs
            ] = desired_hand_pos

        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_hand_arm_dofs
        ]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_arm_hand_dof_state = self.dof_state.view(
                self.num_envs, -1, 2
            )[:, self.num_hand_arm_dofs :]
            self.blue_robot_arm_hand_dof_pos = self.blue_robot_arm_hand_dof_state[
                ..., 0
            ]
            self.blue_robot_arm_hand_dof_vel = self.blue_robot_arm_hand_dof_state[
                ..., 1
            ]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )

        self.set_actor_root_state_object_indices: List[Tensor] = []
        self.set_dof_state_object_indices: List[Tensor] = []

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.prev_actions_for_penalty = torch.zeros(
            (self.num_envs, self.num_robot_actions),
            dtype=torch.float,
            device=self.device,
        )
        self.action_delta_for_penalty = torch.zeros_like(
            self.prev_actions_for_penalty
        )
        self.prev_target_delta_for_penalty = torch.zeros(
            (self.num_envs, self.num_hand_arm_dofs),
            dtype=torch.float,
            device=self.device,
        )
        self.target_delta_for_penalty = torch.zeros_like(
            self.prev_target_delta_for_penalty
        )
        self.target_accel_for_penalty = torch.zeros_like(
            self.prev_target_delta_for_penalty
        )

        self.global_indices = torch.arange(
            self.num_envs * 3, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.prev_episode_successes = torch.zeros_like(self.successes)

        # true objective value for the whole episode, plus saving values for the previous episode
        self.true_objective = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.prev_episode_true_objective = torch.zeros_like(self.true_objective)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device
        )
        self.force_prob_range = to_torch(
            self.force_prob_range, dtype=torch.float, device=self.device
        )
        self.random_force_prob = self._sample_log_uniform(
            min_value=self.force_prob_range[0],
            max_value=self.force_prob_range[1],
            num_samples=self.num_envs,
        )
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )

        self.torque_decay = to_torch(
            self.torque_decay, dtype=torch.float, device=self.device
        )
        self.torque_prob_range = to_torch(
            self.torque_prob_range, dtype=torch.float, device=self.device
        )
        self.random_torque_prob = self._sample_log_uniform(
            min_value=self.torque_prob_range[0],
            max_value=self.torque_prob_range[1],
            num_samples=self.num_envs,
        )
        self.rb_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self.action_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )

        # Random velocity impulses applied to the object
        self.lin_vel_impulse_prob_range = to_torch(
            self.lin_vel_impulse_prob_range, dtype=torch.float, device=self.device
        )
        self.random_lin_vel_impulse_prob = self._sample_log_uniform(
            min_value=self.lin_vel_impulse_prob_range[0],
            max_value=self.lin_vel_impulse_prob_range[1],
            num_samples=self.num_envs,
        )

        self.ang_vel_impulse_prob_range = to_torch(
            self.ang_vel_impulse_prob_range, dtype=torch.float, device=self.device
        )
        self.random_ang_vel_impulse_prob = self._sample_log_uniform(
            min_value=self.ang_vel_impulse_prob_range[0],
            max_value=self.ang_vel_impulse_prob_range[1],
            num_samples=self.num_envs,
        )

        self.obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )
        self.goal_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )
        self.observed_obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )

        self.obj_keypoint_pos_fixed_size = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )
        self.goal_keypoint_pos_fixed_size = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )

        self.object_scale_noise_multiplier = torch.ones(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )

        # how many steps we were within the goal tolerance
        self.near_goal_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )

        self.lifted_object = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.dynamic_grasp_success_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.dynamic_grasp_object_xy_velocity = torch.zeros(
            (self.num_envs, 2), dtype=torch.float, device=self.device
        )
        self.dynamic_grasp_object_yaw_rate = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.object_affordance_surface_offsets = torch.full(
            (self.num_envs,),
            float(self.dynamic_grasp_affordance_surface_offset),
            dtype=torch.float,
            device=self.device,
        )
        self.prev_dynamic_grasp_contact_like = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.closest_keypoint_max_dist = -torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.closest_keypoint_max_dist_fixed_size = -torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.prev_total_episode_closest_keypoint_max_dist = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.total_episode_closest_keypoint_max_dist = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.prev_episode_closest_keypoint_max_dist = 1000 * torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.closest_fingertip_dist = -torch.ones(
            [self.num_envs, self.num_fingertips], dtype=torch.float, device=self.device
        )
        self.furthest_hand_dist = -torch.ones(
            [self.num_envs], dtype=torch.float, device=self.device
        )

        self.finger_rew_coeffs = torch.ones(
            [self.num_envs, self.num_fingertips], dtype=torch.float, device=self.device
        )

        reward_keys = [
            "raw_fingertip_delta_rew",
            "raw_hand_delta_penalty",
            "raw_lifting_rew",
            "raw_keypoint_rew",
            "raw_dynamic_velocity_match_rew",
            "raw_dynamic_intercept_rew",
            "raw_dynamic_pregrasp_alignment_rew",
            "raw_dynamic_affordance_target_rew",
            "raw_dynamic_enclosure_rew",
            "raw_dynamic_lift_progress_rew",
            "raw_dynamic_stable_hold_rew",
            "raw_dynamic_stable_grasp_progress_rew",
            "raw_dynamic_lifted_rel_vel_rew",
            "raw_dynamic_lifted_centering_rew",
            "raw_dynamic_lifted_height_hold_rew",
            "raw_dynamic_lifted_enclosure_vel_rew",
            "raw_dynamic_pre_contact_slow_rew",
            "raw_dynamic_controlled_contact_rew",
            "raw_dynamic_impact_penalty",
            "raw_dynamic_kick_penalty",
            "raw_dynamic_object_palm_rel_vel_penalty",
            "raw_dynamic_push_away_penalty",
            "raw_dynamic_dropped_penalty",
            "raw_dynamic_timeout_penalty",
            "raw_dynamic_safe_action_delta_penalty",
            "raw_dynamic_safe_arm_target_delta_penalty",
            "raw_dynamic_safe_arm_target_accel_penalty",
            "raw_dynamic_safe_hand_target_delta_penalty",
            "raw_dynamic_pregrasp_hold_rew",
            "raw_dynamic_early_contact_penalty",
            "raw_dynamic_pre_contact_rel_vel_penalty",
            "raw_dynamic_ready_enclosure_rew",
            "raw_dynamic_ready_grasp_quality_rew",
            "raw_dynamic_ready_contact_rew",
            "raw_dynamic_true_grasp_quality_rew",
            "raw_dynamic_lifted_true_grasp_rew",
            "raw_dynamic_quality_lift_progress_rew",
            "raw_dynamic_opposing_contact_rew",
            "raw_dynamic_scoop_lift_penalty",
            "raw_dynamic_palm_only_lift_penalty",
            "fingertip_delta_rew",
            "hand_delta_penalty",
            "lifting_rew",
            "lift_bonus_rew",
            "keypoint_rew",
            "dynamic_velocity_match_rew",
            "dynamic_intercept_rew",
            "dynamic_pregrasp_alignment_rew",
            "dynamic_affordance_target_rew",
            "dynamic_enclosure_rew",
            "dynamic_lifted_enclosure_rew",
            "dynamic_lift_progress_rew",
            "dynamic_stable_hold_rew",
            "dynamic_stable_grasp_progress_rew",
            "dynamic_lifted_rel_vel_rew",
            "dynamic_lifted_centering_rew",
            "dynamic_lifted_height_hold_rew",
            "dynamic_lifted_enclosure_vel_rew",
            "dynamic_pre_contact_slow_rew",
            "dynamic_controlled_contact_rew",
            "dynamic_impact_penalty",
            "dynamic_kick_penalty",
            "dynamic_object_palm_rel_vel_penalty",
            "dynamic_push_away_penalty",
            "dynamic_dropped_penalty",
            "dynamic_timeout_penalty",
            "dynamic_safe_action_delta_penalty",
            "dynamic_safe_arm_target_delta_penalty",
            "dynamic_safe_arm_target_accel_penalty",
            "dynamic_safe_hand_target_delta_penalty",
            "dynamic_pregrasp_hold_rew",
            "dynamic_early_contact_penalty",
            "dynamic_pre_contact_rel_vel_penalty",
            "dynamic_ready_enclosure_rew",
            "dynamic_ready_grasp_quality_rew",
            "dynamic_ready_contact_rew",
            "dynamic_true_grasp_quality_rew",
            "dynamic_lifted_true_grasp_rew",
            "dynamic_quality_lift_progress_rew",
            "dynamic_opposing_contact_rew",
            "dynamic_scoop_lift_penalty",
            "dynamic_palm_only_lift_penalty",
            "dynamic_success_bonus",
            "bonus_rew",
            "kuka_actions_penalty",
            "hand_actions_penalty",
            "raw_object_lin_vel_penalty",
            "raw_object_ang_vel_penalty",
            "object_lin_vel_penalty",
            "object_ang_vel_penalty",
            "total_reward",
        ]

        self.rewards_episode = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in reward_keys
        }

        self.last_curriculum_update = 0

        self.episode_root_state_tensors = [[] for _ in range(self.num_envs)]
        self.episode_dof_states = [[] for _ in range(self.num_envs)]

        self.eval_stats: bool = self.cfg["env"]["evalStats"]

        self.good_reset_boundary = self.cfg["env"].get(
            "goodResetBoundary", 0
        )  # Max number of envs that can be reset with good states

        if self.good_reset_boundary > 0:
            self.max_buffer_size = self.cfg["env"].get(
                "maxBufferSize",
                self.max_episode_length
                * self.num_envs
                * 2
                // max(self.max_consecutive_successes, 20),
            )  # Max number of states that can be stored in the buffer
            self.max_temp_buffer_size = (
                self.max_episode_length
            )  # Max number of states that can be stored in the buffer
            self.temp_root_states_buf = torch.empty(
                (
                    self.num_envs,
                    self.max_temp_buffer_size,
                    self.root_state_tensor.shape[0] // self.num_envs,
                    *self.root_state_tensor.shape[1:],
                ),
                dtype=self.root_state_tensor.dtype,
                device="cpu",
            )
            self.temp_dof_states_buf = torch.empty(
                (
                    self.num_envs,
                    self.max_temp_buffer_size,
                    self.dof_state.shape[0] // self.num_envs,
                    *self.dof_state.shape[1:],
                ),
                dtype=self.dof_state.dtype,
                device="cpu",
            )
            self.temp_buffer_index = torch.zeros(
                self.num_envs, dtype=torch.int, device="cpu"
            )
            self.root_state_resets = torch.empty(
                (
                    self.max_buffer_size,
                    self.root_state_tensor.shape[0] // self.num_envs,
                    *self.root_state_tensor.shape[1:],
                ),
                dtype=self.root_state_tensor.dtype,
                device="cpu",
            )
            self.dof_resets = torch.empty(
                (
                    self.max_buffer_size,
                    self.dof_state.shape[0] // self.num_envs,
                    *self.dof_state.shape[1:],
                ),
                dtype=self.dof_state.dtype,
                device="cpu",
            )
            self.buffer_index = 0
            self.buffer_length = 0

        if self.eval_stats:
            self.last_success_step = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
            self.success_time = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
            self.total_num_resets = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
            self.successes_count = torch.zeros(
                self.max_consecutive_successes + 1,
                dtype=torch.float,
                device=self.device,
            )
            from tensorboardX import SummaryWriter

            self.eval_summary_dir = "./eval_summaries"
            # remove the old directory if it exists
            if os.path.exists(self.eval_summary_dir):
                import shutil

                shutil.rmtree(self.eval_summary_dir)
            self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)
            self.eval_stop_after_each_env_once = bool(
                self.cfg["env"].get("evalStopAfterEachEnvOnce", False)
            )
            self.eval_first_episode_recorded = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if hasattr(self, "object_asset_labels") and hasattr(
                self, "object_asset_indices"
            ):
                self.eval_object_episode_counts = torch.zeros(
                    len(self.object_asset_labels), dtype=torch.long, device=self.device
                )
                self.eval_object_success_counts = torch.zeros(
                    len(self.object_asset_labels), dtype=torch.float, device=self.device
                )

        if self.with_table_force_sensor:
            self.table_sensor_forces_raw = torch.zeros(
                (self.num_envs, 6), dtype=torch.float, device=self.device
            )
            self.table_sensor_forces_smoothed = torch.zeros(
                (self.num_envs, 6), dtype=torch.float, device=self.device
            )
            self.max_table_sensor_force_norm_smoothed = torch.zeros(
                (self.num_envs), dtype=torch.float, device=self.device
            )

        self._init_tyler_curriculum()

        self._init_obs_action_queue()

    def _sample_log_uniform(
        self, min_value: float, max_value: float, num_samples: int
    ) -> torch.Tensor:
        """
        Sample values log-uniformly between min_value and max_value.

        Args:
            min_value: Lower bound (must be > 0).
            max_value: Upper bound (must be > 0).

        Returns:
            A tensor of shape (self.num_envs,) sampled log-uniformly.
        """
        assert min_value > 0, f"min_value must be > 0, got {min_value}"
        assert max_value > 0, f"max_value must be > 0, got {max_value}"
        assert min_value <= max_value, (
            f"min_value must be <= max_value, got {min_value} <= {max_value}"
        )

        return torch.exp(
            torch.log(min_value)
            + (torch.log(max_value) - torch.log(min_value))
            * torch.rand(num_samples, device=self.device)
        )

    ##### KEYBOARD START #####
    def _subscribe_to_keyboard_events(self) -> None:
        from dataclasses import dataclass
        from typing import Callable

        @dataclass
        class KeyboardShortcut:
            name: str
            key: int
            function: Callable

        keyboard_shortcuts = [
            KeyboardShortcut(
                name="breakpoint",
                key=gymapi.KEY_B,
                function=self._breakpoint_callback,
            ),
            KeyboardShortcut(
                name="reset",
                key=gymapi.KEY_E,
                function=self._reset_callback,
            ),
            KeyboardShortcut(
                name="toggle_do_not_move",
                key=gymapi.KEY_T,
                function=self._toggle_do_not_move_callback,
            ),
            KeyboardShortcut(
                name="toggle_debug_viz",
                key=gymapi.KEY_D,
                function=self._toggle_debug_viz_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_x_plus",
                key=gymapi.KEY_RIGHT,
                function=self._force_or_torque_x_plus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_x_minus",
                key=gymapi.KEY_LEFT,
                function=self._force_or_torque_x_minus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_y_plus",
                key=gymapi.KEY_UP,
                function=self._force_or_torque_y_plus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_y_minus",
                key=gymapi.KEY_DOWN,
                function=self._force_or_torque_y_minus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_z_plus",
                key=gymapi.KEY_PAGE_UP,
                function=self._force_or_torque_z_plus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_z_minus",
                key=gymapi.KEY_PAGE_DOWN,
                function=self._force_or_torque_z_minus_callback,
            ),
            KeyboardShortcut(
                name="toggle_force_or_torque",
                key=gymapi.KEY_F,
                function=self._toggle_force_or_torque_callback,
            ),
            KeyboardShortcut(
                name="teleport_object",
                key=gymapi.KEY_P,
                function=self._teleport_object_callback,
            ),
        ]
        self.name_to_keyboard_shortcut_dict = {
            keyboard_shortcut.name: keyboard_shortcut
            for keyboard_shortcut in keyboard_shortcuts
        }
        self._DO_NOT_MOVE = False

        # Keyboard applied force or torque or impulse
        # Press arrow keys to apply force or torque or impulse in XY plane, PAGE_UP and PAGE_DOWN for Z direction
        # Press F to toggle between force, torque, lin_vel_impulse, and ang_vel_impulse
        #
        # Control flow is currently
        # - pre_physics_step
        #   - apply_rigid_body_force_tensors
        #   - set_actor_root_state_tensor_indexed
        # - render
        #   - keyboard callbacks
        # - sim step
        #   - root_state_tensor is updated
        # - post_physics_step
        #
        # force and torque is currently stateful
        # you set it once in self.rb_forces and self.rb_torques and it doesn't automatically get reset
        # right now, they just decay over time or get overwritten by random forces/torques
        # doing this from keyboard callbacks is easy because you can set the forces/torques directly
        #
        # velocity impulses are not stateful
        # you are directly setting the root_state_tensor
        # doing this from keyboard callbacks is not easy because keyboard callbacks are called
        # after set_actor_root_state_tensor_indexed and before sim_step
        # directly modifying the root_state_tensor doesn't work because it is overwritten
        # thus, we instead set a flag and store the impulse in a buffer and apply it in pre_physics_step
        # after this, the flag and buffer are reset

        self.force_or_torque_mode = "force"
        self.need_apply_vel_impulse_from_keyboard = False
        self.vel_impulse_from_keyboard = np.array([0.0] * 6)

        self.need_teleport_object_from_keyboard = False

    def _breakpoint_callback(self) -> None:
        print("Breakpoint")
        breakpoint()

    def _reset_callback(self) -> None:
        print("Resetting...")
        # Easiest way to reset without other issues (reset being overwritten)
        # Is to set the progress buffer to the max episode length - 2 so it will reset shortly
        self.progress_buf[:] = self.max_episode_length - 2

    def _toggle_do_not_move_callback(self) -> None:
        print("Toggling do not move...")
        self._DO_NOT_MOVE = not self._DO_NOT_MOVE
        print(f"Do not move is now {self._DO_NOT_MOVE}")

    def _toggle_debug_viz_callback(self) -> None:
        print("Toggling debug viz...")
        self.debug_viz = not self.debug_viz
        print(f"Debug viz is now {self.debug_viz}")
        if not self.debug_viz:
            self.gym.clear_lines(self.viewer)

    def _force_or_torque_x_plus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 0] += (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force x plus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 0] += (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque x plus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[0] += self.lin_vel_impulse_scale
            print(f"Lin vel impulse x plus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[3] += self.ang_vel_impulse_scale
            print(f"Ang vel impulse x plus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_x_minus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 0] -= (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force x minus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 0] -= (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque x minus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[0] -= self.lin_vel_impulse_scale
            print(f"Lin vel impulse x minus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[3] -= self.ang_vel_impulse_scale
            print(f"Ang vel impulse x minus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_y_plus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 1] += (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force y plus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 1] += (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque y plus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[1] += self.lin_vel_impulse_scale
            print(f"Lin vel impulse y plus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[4] += self.ang_vel_impulse_scale
            print(f"Ang vel impulse y plus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_y_minus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 1] -= (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force y minus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 1] -= (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque y minus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[1] -= self.lin_vel_impulse_scale
            print(f"Lin vel impulse y minus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[4] -= self.ang_vel_impulse_scale
            print(f"Ang vel impulse y minus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_z_plus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 2] += (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force z plus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 2] += (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque z plus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[2] += self.lin_vel_impulse_scale
            print(f"Lin vel impulse z plus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[5] += self.ang_vel_impulse_scale
            print(f"Ang vel impulse z plus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_z_minus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 2] -= (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force z minus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 2] -= (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque z minus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[2] -= self.lin_vel_impulse_scale
            print(f"Lin vel impulse z minus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[5] -= self.ang_vel_impulse_scale
            print(f"Ang vel impulse z minus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _toggle_force_or_torque_callback(self) -> None:
        print("Toggling force or torque...")
        MODES = ["force", "torque", "lin_vel_impulse", "ang_vel_impulse"]
        assert self.force_or_torque_mode in MODES, (
            f"Invalid force or torque mode: {self.force_or_torque_mode}"
        )
        idx = MODES.index(self.force_or_torque_mode)
        idx = (idx + 1) % len(MODES)
        self.force_or_torque_mode = MODES[idx]
        print(f"Force or torque mode is now {self.force_or_torque_mode}")

    def _teleport_object_callback(self) -> None:
        print("Teleporting object...")
        self.need_teleport_object_from_keyboard = True
        print(f"Object is now teleported: {self.need_teleport_object_from_keyboard}")

    ##### KEYBOARD END #####

    # SimToolReal abstract interface - to be overriden in derived classes
    def change_on_restart(self, cfg):
        self.frame_since_restart = 0
        self.last_curriculum_update = 0

        self.cfg["env"]["distanceDeltaRewScale"] = cfg["env"]["distanceDeltaRewScale"]
        self.cfg["env"]["liftingRewScale"] = cfg["env"]["liftingRewScale"]
        self.cfg["env"]["liftingBonus"] = cfg["env"]["liftingBonus"]
        self.cfg["env"]["liftingBonusThreshold"] = cfg["env"]["liftingBonusThreshold"]
        self.cfg["env"]["keypointRewScale"] = cfg["env"]["keypointRewScale"]
        self.cfg["env"]["kukaActionsPenaltyScale"] = cfg["env"][
            "kukaActionsPenaltyScale"
        ]
        self.cfg["env"]["handActionsPenaltyScale"] = cfg["env"][
            "handActionsPenaltyScale"
        ]
        self.cfg["env"]["reachGoalBonus"] = cfg["env"]["reachGoalBonus"]
        self.cfg["env"]["fallDistance"] = cfg["env"]["fallDistance"]
        self.cfg["env"]["fallPenalty"] = cfg["env"]["fallPenalty"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        self.hand_actions_penalty_scale = self.cfg["env"]["handActionsPenaltyScale"]

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

    def _object_keypoint_offsets(self):
        return [
            [1, 1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]

    def _object_pointcloud_unit_points(self, num_points: int) -> np.ndarray:
        points, _ = SimToolReal._object_pointcloud_unit_surface(self, num_points)
        return points

    def _object_pointcloud_unit_surface(
        self, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if num_points <= 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        rng = np.random.default_rng(20260518)
        faces = rng.integers(0, 6, size=num_points)
        uv = rng.uniform(-0.5, 0.5, size=(num_points, 2)).astype(np.float32)
        points = np.zeros((num_points, 3), dtype=np.float32)
        normals = np.zeros((num_points, 3), dtype=np.float32)

        for point_idx, face in enumerate(faces):
            axis = face // 2
            sign = 0.5 if face % 2 == 0 else -0.5
            free_axes = [idx for idx in range(3) if idx != axis]
            points[point_idx, axis] = sign
            points[point_idx, free_axes[0]] = uv[point_idx, 0]
            points[point_idx, free_axes[1]] = uv[point_idx, 1]
            normals[point_idx, axis] = 1.0 if sign > 0.0 else -1.0

        points -= points.mean(axis=0, keepdims=True)
        return points, normals

    def _quat_wxyz_to_xyzw(self, quat: List[float]) -> np.ndarray:
        quat_np = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = np.linalg.norm(quat_np)
        if norm < 1e-8:
            quat_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            quat_np = quat_np / norm
        return np.array(
            [quat_np[1], quat_np[2], quat_np[3], quat_np[0]], dtype=np.float32
        )

    def _quat_xyzw_to_matrix_np(self, quat: np.ndarray) -> np.ndarray:
        x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-8:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _domino_variant_value(self, variant, names, default=None):
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name in variant:
                return variant[name]
        return default

    def _domino_asset_file(
        self, asset_root: Path, modelname: str, model_id: int, source: str
    ) -> Path:
        modeldir = asset_root / str(modelname)
        source = str(source or "collision").lower()
        preferred_dirs = (
            [modeldir / "collision", modeldir, modeldir / "visual"]
            if source == "collision"
            else [modeldir / "visual", modeldir, modeldir / "collision"]
        )
        stems = [f"base{model_id}", "base"] if model_id is not None else ["base"]
        suffixes = [".obj", ".glb", ".gltf", ".stl", ".ply"]
        for directory in preferred_dirs:
            if not directory.exists():
                continue
            for stem in stems:
                for suffix in suffixes:
                    path = directory / f"{stem}{suffix}"
                    if path.exists():
                        return path
        raise FileNotFoundError(f"No DOMINO mesh found for {modelname}/base{model_id}")

    def _domino_asset_scale(
        self, asset_root: Path, modelname: str, model_id: int
    ) -> np.ndarray:
        modeldir = asset_root / str(modelname)
        json_path = (
            modeldir / "model_data.json"
            if model_id is None
            else modeldir / f"model_data{model_id}.json"
        )
        if not json_path.exists():
            return np.ones(3, dtype=np.float64)
        with open(json_path, "r") as f:
            data = json.load(f)
        scale_np = np.asarray(data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        scale_np = scale_np.reshape(-1)
        if scale_np.size == 1:
            return np.repeat(scale_np, 3)
        return scale_np[:3]

    def _load_domino_mesh(
        self,
        mesh_path: Path,
        mesh_scale: np.ndarray,
        recenter: Optional[np.ndarray] = None,
    ):
        import trimesh

        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            parts = [
                geom
                for geom in mesh.geometry.values()
                if hasattr(geom, "vertices") and len(geom.vertices) > 0
            ]
            if not parts:
                raise RuntimeError(f"DOMINO mesh has no geometry: {mesh_path}")
            mesh = trimesh.util.concatenate(parts)
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            raise RuntimeError(f"DOMINO mesh has no vertices: {mesh_path}")
        mesh = mesh.copy()
        mesh.apply_scale(mesh_scale)
        if recenter is not None:
            mesh.apply_translation(-np.asarray(recenter, dtype=np.float64))
        return mesh

    def _sample_mesh_surface_points(
        self, mesh, num_points: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if num_points <= 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
        import trimesh

        np_state = np.random.get_state()
        np.random.seed(seed)
        try:
            points, face_ids = trimesh.sample.sample_surface(mesh, num_points)
        finally:
            np.random.set_state(np_state)
        normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
        return points.astype(np.float32), normals.astype(np.float32)

    def _write_domino_object_urdf(
        self,
        urdf_path: Path,
        visual_mesh_name: str,
        collision_mesh_name: str,
        mass: float,
        extents: np.ndarray,
    ) -> None:
        mass = max(float(mass), 1e-4)
        extents = np.maximum(np.asarray(extents, dtype=np.float64), 1e-4)
        ixx = mass * (extents[1] ** 2 + extents[2] ** 2) / 12.0
        iyy = mass * (extents[0] ** 2 + extents[2] ** 2) / 12.0
        izz = mass * (extents[0] ** 2 + extents[1] ** 2) / 12.0
        urdf_path.parent.mkdir(parents=True, exist_ok=True)
        urdf_text = f"""<?xml version="1.0"?>
<robot name="{urdf_path.stem}">
  <link name="object">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_mesh_name}" scale="1 1 1"/>
      </geometry>
      <material name="domino_gray">
        <color rgba="0.65 0.65 0.65 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{collision_mesh_name}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
  </link>
</robot>
"""
        with open(urdf_path, "w") as f:
            f.write(urdf_text)

    def _domino_object_assets(self, tmp_assets_dir: str):
        asset_root = Path(self.domino_object_asset_root).expanduser().resolve()
        if not asset_root.exists():
            raise FileNotFoundError(f"DOMINO object asset root not found: {asset_root}")

        variants = list(self.domino_object_variants or DEFAULT_DOMINO_OBJECT_VARIANTS)
        if len(variants) == 0:
            raise ValueError("dominoObjectVariants must contain at least one object")

        generated_dir = Path(tmp_assets_dir) / "domino_20"
        generated_dir.mkdir(parents=True, exist_ok=True)

        object_asset_files = []
        object_asset_scales = []
        need_vhacds = []
        pointcloud_points = []
        pointcloud_normals = []
        default_quats_xyzw = []
        random_yaw_flags = []
        table_z_offsets = []
        labels = []

        candidate_points = self.object_pointcloud_candidate_num_points
        for asset_idx, variant in enumerate(variants):
            label = str(
                self._domino_variant_value(
                    variant, ["label", "name", "asset_label"], f"object_{asset_idx}"
                )
            ).replace(" ", "_")
            modelname = str(
                self._domino_variant_value(
                    variant, ["asset_modelname", "anydex_asset_modelname"]
                )
            )
            model_id = self._domino_variant_value(
                variant, ["asset_model_id", "anydex_asset_model_id"], 0
            )
            model_id = None if model_id is None else int(model_id)
            mass = float(
                self._domino_variant_value(
                    variant, ["mass", "anydex_object_mass"], 0.025
                )
            )
            mesh_scale = self._domino_asset_scale(asset_root, modelname, model_id)
            collision_src = self._domino_asset_file(
                asset_root, modelname, model_id, self.domino_object_collision_source
            )
            visual_src = self._domino_asset_file(
                asset_root, modelname, model_id, self.domino_object_visual_source
            )

            collision_metric = self._load_domino_mesh(collision_src, mesh_scale)
            center = collision_metric.bounds.mean(axis=0)
            collision_mesh = self._load_domino_mesh(
                collision_src, mesh_scale, recenter=center
            )
            visual_mesh = self._load_domino_mesh(
                visual_src, mesh_scale, recenter=center
            )

            bounds = np.asarray(collision_mesh.bounds, dtype=np.float64)
            extents = np.maximum(bounds[1] - bounds[0], 1e-4)
            scale_for_policy = tuple((extents / self.object_base_size).tolist())

            quat_wxyz = self._domino_variant_value(
                variant,
                ["quat", "anydex_object_orientation_quat", "dynamic_object_orientation_quat"],
                [1.0, 0.0, 0.0, 0.0],
            )
            quat_xyzw = self._quat_wxyz_to_xyzw(quat_wxyz)
            rotation = self._quat_xyzw_to_matrix_np(quat_xyzw)
            corners = np.asarray(
                [
                    [x, y, z]
                    for x in (bounds[0, 0], bounds[1, 0])
                    for y in (bounds[0, 1], bounds[1, 1])
                    for z in (bounds[0, 2], bounds[1, 2])
                ],
                dtype=np.float64,
            )
            rotated_corners = (rotation @ corners.T).T
            half_height = max(0.0, float(-rotated_corners[:, 2].min()))

            object_dir = generated_dir / f"{asset_idx:02d}_{label}"
            object_dir.mkdir(parents=True, exist_ok=True)
            visual_mesh_name = "visual.obj"
            collision_mesh_name = "collision.obj"
            visual_mesh.export(object_dir / visual_mesh_name)
            collision_mesh.export(object_dir / collision_mesh_name)
            urdf_path = object_dir / f"{label}.urdf"
            self._write_domino_object_urdf(
                urdf_path=urdf_path,
                visual_mesh_name=visual_mesh_name,
                collision_mesh_name=collision_mesh_name,
                mass=mass,
                extents=extents,
            )

            points, normals = self._sample_mesh_surface_points(
                visual_mesh, candidate_points, seed=20260525 + asset_idx
            )
            object_asset_files.append(urdf_path)
            object_asset_scales.append(scale_for_policy)
            need_vhacds.append(self.domino_object_need_vhacd)
            pointcloud_points.append(points)
            pointcloud_normals.append(normals)
            default_quats_xyzw.append(quat_xyzw)
            random_yaw_flags.append(
                bool(
                    self._domino_variant_value(
                        variant,
                        ["random_yaw", "anydex_object_random_yaw"],
                        self.domino_object_random_yaw_default,
                    )
                )
            )
            table_z_offsets.append(
                self.domino_object_table_top_offset
                + half_height
                + self.domino_object_spawn_z_margin
            )
            labels.append(label)

        self.object_asset_pointcloud_points_np = np.stack(pointcloud_points, axis=0)
        self.object_asset_pointcloud_normals_np = np.stack(pointcloud_normals, axis=0)
        self.object_asset_default_quats_xyzw_np = np.stack(
            default_quats_xyzw, axis=0
        ).astype(np.float32)
        self.object_asset_random_yaw_np = np.asarray(random_yaw_flags, dtype=np.bool_)
        self.object_asset_table_z_offsets_np = np.asarray(
            table_z_offsets, dtype=np.float32
        )
        self.object_asset_labels = labels

        print(f"Loaded DOMINO object pool ({len(labels)} objects): {labels}")
        return object_asset_files, object_asset_scales, need_vhacds

    def _dynamic_affordance_surface_offsets_for_assets(
        self, object_asset_indices_np: np.ndarray
    ) -> np.ndarray:
        offsets = np.full(
            len(object_asset_indices_np),
            float(self.dynamic_grasp_affordance_surface_offset),
            dtype=np.float32,
        )
        labels = getattr(self, "object_asset_labels", None)
        if labels is None:
            return offsets
        for asset_idx, label in enumerate(labels):
            label_offset = self.dynamic_grasp_affordance_surface_offsets_by_label.get(
                str(label), self.dynamic_grasp_affordance_surface_offset
            )
            offsets[object_asset_indices_np == asset_idx] = float(label_offset)
        return offsets

    def _dynamic_affordance_targets(
        self,
        object_pos: Tensor,
        object_vel: Tensor,
        palm_pos: Tensor,
    ) -> dict:
        object_xy_vel = object_vel[:, 0:2]
        commanded_xy_vel = (
            self.dynamic_grasp_object_xy_velocity
            if hasattr(self, "dynamic_grasp_object_xy_velocity")
            else torch.zeros_like(object_xy_vel)
        )
        object_xy_speed = torch.norm(object_xy_vel, dim=-1, keepdim=True)
        commanded_xy_speed = torch.norm(commanded_xy_vel, dim=-1, keepdim=True)
        dir_source = torch.where(
            object_xy_speed > 1e-4,
            object_xy_vel,
            commanded_xy_vel,
        )
        dir_source_speed = torch.norm(dir_source, dim=-1, keepdim=True)

        palm_to_object_xy = object_pos[:, 0:2] - palm_pos[:, 0:2]
        palm_to_object_speed = torch.norm(palm_to_object_xy, dim=-1, keepdim=True)
        default_dir = torch.zeros_like(dir_source)
        default_dir[:, 0] = 1.0
        fallback_dir = torch.where(
            palm_to_object_speed > 1e-4,
            palm_to_object_xy / torch.clamp(palm_to_object_speed, min=1e-6),
            default_dir,
        )
        xy_dir = torch.where(
            dir_source_speed > 1e-4,
            dir_source / torch.clamp(dir_source_speed, min=1e-6),
            fallback_dir,
        )

        if (
            self.dynamic_grasp_use_affordance_prior
            and self.dynamic_grasp_affordance_mode != "center"
        ):
            if hasattr(self, "object_affordance_surface_offsets"):
                surface_offsets = self.object_affordance_surface_offsets
            else:
                surface_offsets = torch.full(
                    (self.num_envs,),
                    float(self.dynamic_grasp_affordance_surface_offset),
                    dtype=torch.float,
                    device=self.device,
                )
        else:
            surface_offsets = torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
            )

        affordance_pos = object_pos.clone()
        affordance_pos[:, 0:2] = (
            affordance_pos[:, 0:2] + xy_dir * surface_offsets[:, None]
        )
        affordance_pos[:, 2] = (
            affordance_pos[:, 2] + self.dynamic_grasp_affordance_vertical_offset
        )

        lead_time = max(float(self.dynamic_grasp_affordance_lead_time), 0.0)
        future_affordance_pos = affordance_pos.clone()
        future_affordance_pos[:, 0:2] = (
            future_affordance_pos[:, 0:2] + object_xy_vel * lead_time
        )
        future_affordance_pos[:, 2] = (
            future_affordance_pos[:, 2] + object_vel[:, 2] * lead_time
        )

        axis_world = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        axis_world[:, 0:2] = xy_dir
        motion_speed = torch.maximum(object_xy_speed, commanded_xy_speed).squeeze(-1)
        confidence = torch.clamp(
            motion_speed
            / max(float(self.dynamic_grasp_affordance_min_speed_for_confidence), 1e-6),
            0.0,
            1.0,
        )
        if not self.dynamic_grasp_use_affordance_prior:
            confidence = torch.zeros_like(confidence)

        return {
            "pos": affordance_pos,
            "future_pos": future_affordance_pos,
            "axis_world": axis_world,
            "confidence": confidence.unsqueeze(-1),
        }

    def _dynamic_affordance_observation(
        self,
        object_pos: Tensor,
        object_vel: Tensor,
        palm_pos: Tensor,
        palm_rot: Tensor,
        add_noise: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        targets = self._dynamic_affordance_targets(
            object_pos=object_pos,
            object_vel=object_vel,
            palm_pos=palm_pos,
        )
        affordance_pos_rel_palm = quat_rotate_inverse(
            palm_rot, targets["pos"] - palm_pos
        )
        future_affordance_pos_rel_palm = quat_rotate_inverse(
            palm_rot, targets["future_pos"] - palm_pos
        )
        affordance_axis_rel_palm = quat_rotate_inverse(
            palm_rot, targets["axis_world"]
        )
        confidence = targets["confidence"]

        if add_noise:
            if self.dynamic_grasp_affordance_obs_noise_std > 0.0:
                pos_noise = (
                    torch.randn_like(affordance_pos_rel_palm)
                    * self.dynamic_grasp_affordance_obs_noise_std
                )
                future_noise = (
                    torch.randn_like(future_affordance_pos_rel_palm)
                    * self.dynamic_grasp_affordance_obs_noise_std
                )
                affordance_pos_rel_palm = affordance_pos_rel_palm + pos_noise
                future_affordance_pos_rel_palm = (
                    future_affordance_pos_rel_palm + future_noise
                )
            if self.dynamic_grasp_affordance_axis_noise_std > 0.0:
                affordance_axis_rel_palm = affordance_axis_rel_palm + (
                    torch.randn_like(affordance_axis_rel_palm)
                    * self.dynamic_grasp_affordance_axis_noise_std
                )
                affordance_axis_rel_palm = affordance_axis_rel_palm / torch.clamp(
                    torch.norm(affordance_axis_rel_palm, dim=-1, keepdim=True),
                    min=1e-6,
                )
            if self.dynamic_grasp_affordance_dropout_prob > 0.0:
                keep = (
                    torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device)
                    > self.dynamic_grasp_affordance_dropout_prob
                )
                confidence = confidence * keep.float()

        return (
            affordance_pos_rel_palm,
            future_affordance_pos_rel_palm,
            affordance_axis_rel_palm,
            confidence,
        )

    def _object_pointcloud_observation(
        self,
        object_pos: Tensor,
        object_rot: Tensor,
        object_vel: Tensor,
        palm_pos: Tensor,
        palm_rot: Tensor,
        palm_vel: Tensor,
        add_noise: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_points = self.object_pointcloud_num_points
        if num_points <= 0:
            empty_cloud = torch.zeros((self.num_envs, 0), device=self.device)
            empty_vec = torch.zeros((self.num_envs, 3), device=self.device)
            confidence = torch.ones((self.num_envs, 1), device=self.device)
            return empty_cloud, empty_vec, empty_vec, confidence

        if self.object_pointcloud_unit_points.dim() == 3:
            candidate_num_points = self.object_pointcloud_unit_points.shape[1]
            local_points = self.object_pointcloud_unit_points
            local_normals_for_env = self.object_pointcloud_unit_normals
        else:
            candidate_num_points = self.object_pointcloud_unit_points.shape[0]
            object_sizes = (
                self.object_scales
                * self.object_base_size
                * self.object_scale_noise_multiplier
            )
            local_points = self.object_pointcloud_unit_points[None, :, :] * object_sizes[
                :, None, :
            ]
            local_normals_for_env = self.object_pointcloud_unit_normals[
                None, :, :
            ].expand(self.num_envs, -1, -1)
        flat_local_points = local_points.reshape(-1, 3)
        repeated_object_rot = object_rot[:, None, :].expand(
            -1, candidate_num_points, -1
        )
        world_points = object_pos[:, None, :] + quat_rotate(
            repeated_object_rot.reshape(-1, 4), flat_local_points
        ).reshape(self.num_envs, candidate_num_points, 3)

        tracking_confidence = torch.ones((self.num_envs, 1), device=self.device)
        if self.object_pointcloud_use_partial_view:
            flat_local_normals = local_normals_for_env.reshape(-1, 3)
            world_normals = quat_rotate(
                object_rot[:, None, :].expand(-1, candidate_num_points, -1).reshape(
                    -1, 4
                ),
                flat_local_normals,
            ).reshape(self.num_envs, candidate_num_points, 3)

            camera_pos = self.object_pointcloud_camera_pos.unsqueeze(0).expand(
                self.num_envs, -1
            )
            if add_noise and self.object_pointcloud_camera_pos_noise_std > 0.0:
                camera_pos = camera_pos + (
                    torch.randn_like(camera_pos)
                    * self.object_pointcloud_camera_pos_noise_std
                )

            point_to_camera = camera_pos[:, None, :] - world_points
            point_to_camera_unit = point_to_camera / torch.clamp(
                torch.norm(point_to_camera, dim=-1, keepdim=True), min=1e-6
            )
            visible_scores = (world_normals * point_to_camera_unit).sum(dim=-1)
            if add_noise:
                visible_scores = visible_scores + 0.01 * torch.randn_like(
                    visible_scores
                )

            _, selected_point_ids = torch.topk(
                visible_scores, k=num_points, dim=1, largest=True, sorted=True
            )
            gather_ids = selected_point_ids[:, :, None].expand(-1, -1, 3)
            world_points = torch.gather(world_points, 1, gather_ids)
            tracking_confidence = (
                visible_scores > self.object_pointcloud_visible_dot_threshold
            ).float().mean(dim=1, keepdim=True)
        else:
            world_points = world_points[:, :num_points, :]

        if add_noise and self.object_pointcloud_depth_noise_std > 0.0:
            camera_pos = self.object_pointcloud_camera_pos.unsqueeze(0).expand(
                self.num_envs, -1
            )
            camera_ray = world_points - camera_pos[:, None, :]
            camera_ray_unit = camera_ray / torch.clamp(
                torch.norm(camera_ray, dim=-1, keepdim=True), min=1e-6
            )
            world_points = world_points + (
                camera_ray_unit
                * torch.randn((self.num_envs, num_points, 1), device=self.device)
                * self.object_pointcloud_depth_noise_std
            )

        rel_world_points = world_points - palm_pos[:, None, :]

        repeated_palm_rot = palm_rot[:, None, :].expand(-1, num_points, -1)
        points_rel_palm = quat_rotate_inverse(
            repeated_palm_rot.reshape(-1, 4), rel_world_points.reshape(-1, 3)
        ).reshape(self.num_envs, num_points, 3)

        if self.object_pointcloud_centroid_from_points:
            centroid_rel_palm = points_rel_palm.mean(dim=1)
        else:
            centroid_rel_world = object_pos - palm_pos
            centroid_rel_palm = quat_rotate_inverse(palm_rot, centroid_rel_world)
        vel_rel_world = object_vel[:, 0:3] - palm_vel[:, 0:3]
        vel_rel_palm = quat_rotate_inverse(palm_rot, vel_rel_world)

        if add_noise:
            if self.object_pointcloud_noise_std > 0.0:
                points_rel_palm = points_rel_palm + (
                    torch.randn_like(points_rel_palm) * self.object_pointcloud_noise_std
                )

            if self.object_pointcloud_dropout_prob > 0.0:
                keep_mask = (
                    torch.rand(
                        (self.num_envs, num_points, 1),
                        dtype=torch.float,
                        device=self.device,
                    )
                    > self.object_pointcloud_dropout_prob
                )
                tracking_confidence = tracking_confidence * keep_mask.float().mean(
                    dim=1
                )
                points_rel_palm = torch.where(
                    keep_mask,
                    points_rel_palm,
                    centroid_rel_palm[:, None, :],
                )

            if self.object_pointcloud_outlier_prob > 0.0:
                outlier_mask = (
                    torch.rand(
                        (self.num_envs, num_points, 1),
                        dtype=torch.float,
                        device=self.device,
                    )
                    < self.object_pointcloud_outlier_prob
                )
                outlier_offsets = (
                    torch.rand(
                        (self.num_envs, num_points, 3),
                        dtype=torch.float,
                        device=self.device,
                    )
                    * 2.0
                    - 1.0
                ) * self.object_pointcloud_outlier_scale
                points_rel_palm = torch.where(
                    outlier_mask,
                    centroid_rel_palm[:, None, :] + outlier_offsets,
                    points_rel_palm,
                )
                tracking_confidence = tracking_confidence * (
                    1.0 - outlier_mask.float().mean(dim=1)
                )

            if self.object_pointcloud_randomize_order:
                random_order = torch.argsort(
                    torch.rand(
                        (self.num_envs, num_points),
                        dtype=torch.float,
                        device=self.device,
                    ),
                    dim=1,
                )
                points_rel_palm = torch.gather(
                    points_rel_palm,
                    1,
                    random_order[:, :, None].expand(-1, -1, 3),
                )

        return (
            points_rel_palm.reshape(self.num_envs, num_points * 3),
            centroid_rel_palm,
            vel_rel_palm,
            tracking_confidence,
        )

    def _object_surface_points_world(self) -> Tensor:
        num_points = int(self.object_pointcloud_num_points)
        if num_points <= 0:
            return self.object_pos[:, None, :]

        if self.object_pointcloud_unit_points.dim() == 3:
            local_points = self.object_pointcloud_unit_points[:, :num_points, :]
            num_points = local_points.shape[1]
        else:
            num_points = min(num_points, self.object_pointcloud_unit_points.shape[0])
            object_sizes = (
                self.object_scales
                * self.object_base_size
                * self.object_scale_noise_multiplier
            )
            local_points = (
                self.object_pointcloud_unit_points[None, :num_points, :]
                * object_sizes[:, None, :]
            )

        repeated_object_rot = self.object_rot[:, None, :].expand(
            -1, num_points, -1
        )
        return self.object_pos[:, None, :] + quat_rotate(
            repeated_object_rot.reshape(-1, 4), local_points.reshape(-1, 3)
        ).reshape(self.num_envs, num_points, 3)

    def _dynamic_true_grasp_metrics(
        self, object_palm_dist: Tensor, lifted_object: Tensor
    ) -> dict:
        surface_points = self._object_surface_points_world()
        fingertip_surface_dist = torch.norm(
            self.fingertip_pos_offset[:, :, None, :]
            - surface_points[:, None, :, :],
            dim=-1,
        ).min(dim=-1).values

        contact_scale = max(float(self.dynamic_grasp_surface_contact_distance), 1e-6)
        contact_scores = torch.exp(-fingertip_surface_dist / contact_scale)
        finger_contact = fingertip_surface_dist < contact_scale
        finger_contact_count = finger_contact.float().sum(dim=-1)

        thumb_idx = self.thumb_fingertip_idx
        non_thumb_indices = self.non_thumb_fingertip_indices
        thumb_score = contact_scores[:, thumb_idx]
        thumb_contact = finger_contact[:, thumb_idx]

        if len(non_thumb_indices) > 0:
            non_thumb_scores = contact_scores[:, non_thumb_indices]
            non_thumb_contact = finger_contact[:, non_thumb_indices]
            non_thumb_contact_count = non_thumb_contact.float().sum(dim=-1)

            fingertip_vectors = self.fingertip_pos_offset - self.object_pos[:, None, :]
            fingertip_dirs = fingertip_vectors / torch.clamp(
                torch.norm(fingertip_vectors, dim=-1, keepdim=True), min=1e-6
            )
            thumb_dir = fingertip_dirs[:, thumb_idx]
            non_thumb_dirs = fingertip_dirs[:, non_thumb_indices]
            thumb_to_finger_dot = (thumb_dir[:, None, :] * non_thumb_dirs).sum(dim=-1)

            threshold = float(self.dynamic_grasp_opposing_dot_threshold)
            opposition_den = max(threshold + 1.0, 1e-6)
            opposition_progress = torch.clamp(
                (threshold - thumb_to_finger_dot) / opposition_den,
                0.0,
                1.0,
            )
            opposing_score = (
                opposition_progress * non_thumb_scores * thumb_score[:, None]
            ).max(dim=1).values
            opposing_contact = (
                (thumb_to_finger_dot < threshold)
                & non_thumb_contact
                & thumb_contact[:, None]
            ).any(dim=1)
            non_thumb_quality = torch.clamp(
                non_thumb_scores.sum(dim=-1)
                / max(float(self.dynamic_grasp_min_non_thumb_contacts), 1.0),
                0.0,
                1.0,
            )
        else:
            non_thumb_contact_count = torch.zeros_like(finger_contact_count)
            opposing_score = torch.zeros_like(finger_contact_count)
            opposing_contact = torch.zeros_like(thumb_contact)
            non_thumb_quality = torch.zeros_like(finger_contact_count)

        finger_count_quality = torch.clamp(
            contact_scores.sum(dim=-1)
            / max(float(self.dynamic_grasp_min_finger_contacts), 1.0),
            0.0,
            1.0,
        )
        grasp_quality = torch.clamp(
            0.30 * finger_count_quality
            + 0.25 * non_thumb_quality
            + 0.25 * thumb_score
            + 0.20 * opposing_score,
            0.0,
            1.0,
        )
        true_grasp = (
            thumb_contact
            & (non_thumb_contact_count >= self.dynamic_grasp_min_non_thumb_contacts)
            & (finger_contact_count >= self.dynamic_grasp_min_finger_contacts)
            & opposing_contact
        )
        scoop_lift = lifted_object & (~true_grasp)
        palm_only_lift = (
            lifted_object
            & (object_palm_dist < self.dynamic_grasp_palm_only_lift_dist)
            & (
                (finger_contact_count < self.dynamic_grasp_min_finger_contacts)
                | (non_thumb_contact_count < self.dynamic_grasp_min_non_thumb_contacts)
                | (~thumb_contact)
            )
        )

        return {
            "fingertip_surface_dist": fingertip_surface_dist,
            "finger_contact_count": finger_contact_count,
            "non_thumb_contact_count": non_thumb_contact_count,
            "thumb_contact": thumb_contact,
            "opposing_contact": opposing_contact,
            "opposing_score": opposing_score,
            "grasp_quality": grasp_quality,
            "true_grasp": true_grasp,
            "scoop_lift": scoop_lift,
            "palm_only_lift": palm_only_lift,
        }

    def _object_pointcloud_centroid_velocity_observation(
        self, centroid_rel_palm: Tensor, add_noise: bool
    ) -> Tensor:
        dt = max(float(self.control_dt), 1e-6)
        velocity_rel_palm = (
            centroid_rel_palm - self.prev_object_pointcloud_centroid_rel_palm
        ) / dt

        initialized = self.object_pointcloud_velocity_initialized
        initialized = initialized & (self.progress_buf > 1)
        velocity_rel_palm = torch.where(
            initialized[:, None],
            velocity_rel_palm,
            torch.zeros_like(velocity_rel_palm),
        )

        if add_noise and self.object_pointcloud_velocity_noise_std > 0.0:
            velocity_rel_palm = velocity_rel_palm + (
                torch.randn_like(velocity_rel_palm)
                * self.object_pointcloud_velocity_noise_std
            )

        if self.object_pointcloud_velocity_clip > 0.0:
            velocity_rel_palm = torch.clamp(
                velocity_rel_palm,
                -self.object_pointcloud_velocity_clip,
                self.object_pointcloud_velocity_clip,
            )

        self.prev_object_pointcloud_centroid_rel_palm[:] = centroid_rel_palm.detach()
        self.object_pointcloud_velocity_initialized[:] = True
        return velocity_rel_palm

    def _object_start_pose(self, robot_pose, table_pose_dy, table_pose_dz):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = robot_pose.p.x

        pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.25

        object_start_pose.p.y = robot_pose.p.y + pose_dy
        object_start_pose.p.z = robot_pose.p.z + pose_dz

        # HACK: Overwrite
        if self.cfg["env"]["objectStartPose"] is not None:
            assert len(self.cfg["env"]["objectStartPose"]) == 7, (
                f"objectStartPose must be a 7-element list, got {len(self.cfg['env']['objectStartPose'])}"
            )
            # Assumes [x, y, z, qx, qy, qz, qw]
            object_start_pose.p.x = self.cfg["env"]["objectStartPose"][0]
            object_start_pose.p.y = self.cfg["env"]["objectStartPose"][1]
            object_start_pose.p.z = self.cfg["env"]["objectStartPose"][2]
            object_start_pose.r.x = self.cfg["env"]["objectStartPose"][3]
            object_start_pose.r.y = self.cfg["env"]["objectStartPose"][4]
            object_start_pose.r.z = self.cfg["env"]["objectStartPose"][5]
            object_start_pose.r.w = self.cfg["env"]["objectStartPose"][6]

        return object_start_pose

    def _main_object_assets_and_scales(self, object_asset_root, tmp_assets_dir):
        object_name = self.cfg["env"]["objectName"]
        known_object_names = set(NAME_TO_OBJECT.keys())
        if object_name in known_object_names:
            # One of known objects
            obj = NAME_TO_OBJECT[object_name]
            object_asset_files = [obj.urdf_path]
            object_asset_scales = [obj.scale]
            need_vhacds = [obj.need_vhacd]

        elif object_name == "handle_head_primitives":
            object_asset_files, object_asset_scales, need_vhacds = (
                self._handle_head_primitives(
                    str(Path(tmp_assets_dir) / "handle_head_primitives"),
                )
            )

        elif object_name in {"domino_20", "domino_object_pool"}:
            object_asset_files, object_asset_scales, need_vhacds = (
                self._domino_object_assets(tmp_assets_dir)
            )

        else:
            raise ValueError(f"Unknown object name: {object_name}")

        USE_FIXED_GOAL_STATES = self.cfg["env"]["useFixedGoalStates"]
        if USE_FIXED_GOAL_STATES:
            FIXED_GOAL_STATES = self.cfg["env"]["fixedGoalStates"]
            FIXED_GOAL_STATES_JSON_PATH = self.cfg["env"]["fixedGoalStatesJsonPath"]
            assert (FIXED_GOAL_STATES is None) != (
                FIXED_GOAL_STATES_JSON_PATH is None
            ), (
                "Exactly one of fixedGoalStates or fixedGoalStatesJsonPath must be set when useFixedGoalStates is True"
            )

            if FIXED_GOAL_STATES is not None:
                self.trajectory_states = torch.tensor(
                    FIXED_GOAL_STATES, device=self.device
                )
            if FIXED_GOAL_STATES_JSON_PATH is not None:
                with open(FIXED_GOAL_STATES_JSON_PATH, "r") as f:
                    self.trajectory_states = torch.tensor(
                        json.load(f)["goals"], device=self.device
                    )

            # Set max consecutive successes to the length of the trajectory so we don't run out of goal states
            self.max_consecutive_successes = len(self.trajectory_states)

        return object_asset_files, object_asset_scales, need_vhacds

    def _load_main_object_asset(self):
        """Load manipulated object and goal assets."""
        object_assets = []
        for object_asset_file, need_vhacd in zip(
            self.object_asset_files, self.object_need_vhacds
        ):
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.vhacd_enabled = need_vhacd

            # WARNING: This should not be done if trying to set different densities for different parts of the object, unless handled appropriately in the URDF
            object_asset_options.collapse_fixed_joints = True

            # This should speed up things and make physics better
            # But self-collision may be an issue
            object_asset_options.replace_cylinder_with_capsule = True
            # object_asset_options.replace_cylinder_with_capsule = False

            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)

            object_asset_ = self.gym.load_asset(
                self.sim, object_asset_dir, object_asset_fname, object_asset_options
            )
            object_assets.append(object_asset_)
        object_rb_count = max(
            self.gym.get_asset_rigid_body_count(object_asset)
            for object_asset in object_assets
        )
        object_shapes_count = max(
            self.gym.get_asset_rigid_shape_count(object_asset)
            for object_asset in object_assets
        )
        return object_assets, object_rb_count, object_shapes_count

    def _load_additional_assets(self, object_asset_root, arm_pose):
        if not self.create_goal_object:
            self.goal_assets = []
            return 0, 0

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True

        # WARNING: This should not be done if trying to set different densities for different parts of the object, unless handled appropriately in the URDF
        object_asset_options.collapse_fixed_joints = True

        # This should speed up things and make physics better
        # But self-collision may be an issue
        object_asset_options.replace_cylinder_with_capsule = True
        # object_asset_options.replace_cylinder_with_capsule = False

        self.goal_assets = []
        for object_asset_file in self.object_asset_files:
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)

            goal_asset_ = self.gym.load_asset(
                self.sim, object_asset_dir, object_asset_fname, object_asset_options
            )
            self.goal_assets.append(goal_asset_)
        goal_rb_count = max(
            self.gym.get_asset_rigid_body_count(goal_asset)
            for goal_asset in self.goal_assets
        )
        goal_shapes_count = max(
            self.gym.get_asset_rigid_shape_count(goal_asset)
            for goal_asset in self.goal_assets
        )

        return goal_rb_count, goal_shapes_count

    def _create_additional_objects(
        self, env_ptr, env_idx, object_asset_idx, object_start_pose=None
    ):
        if not self.create_goal_object:
            return

        self.goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )
        if object_start_pose is None:
            object_start_pose = self.object_start_pose
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.p.z -= 0.04
        goal_start_pose.r = object_start_pose.r

        goal_asset = self.goal_assets[object_asset_idx]
        goal_handle = self.gym.create_actor(
            env_ptr,
            goal_asset,
            goal_start_pose,
            "goal_object",
            env_idx + self.num_envs,
            0,
            0,
        )
        goal_object_idx = self.gym.get_actor_index(
            env_ptr, goal_handle, gymapi.DOMAIN_SIM
        )
        self.goal_object_indices.append(goal_object_idx)
        for name in self.gym.get_actor_rigid_body_names(env_ptr, goal_handle):
            self.rigid_body_name_to_idx["goal/" + name] = (
                self.gym.find_actor_rigid_body_index(
                    env_ptr, goal_handle, name, gymapi.DOMAIN_ENV
                )
            )

        GREEN = (0.0, 1.0, 0.0)
        self._set_actor_color(env_ptr, goal_handle, GREEN)

    def _after_envs_created(self):
        self.goal_object_indices = to_torch(
            self.goal_object_indices, dtype=torch.long, device=self.device
        )

    def _extra_reset_rules(self, resets):
        if self.dynamic_tabletop_grasp:
            resets = resets | self._dynamic_tabletop_workspace_resets()
        return resets

    def _sample_delta_goal(
        self, goal_states, delta_goal_distance, delta_rotation_degrees
    ):
        # get the target volume origin and extent
        target_volume_origin = self.target_volume_origin
        target_volume_extent = self.target_volume_extent
        target_volume_min_coord = target_volume_origin + target_volume_extent[:, 0]
        target_volume_max_coord = target_volume_origin + target_volume_extent[:, 1]

        last_goal = goal_states.clone()
        last_goal_pos = last_goal[:, :3]
        last_goal_quat_xyzw = last_goal[:, 3:7]

        new_goal_pos = last_goal_pos + torch_rand_float(
            -delta_goal_distance,
            delta_goal_distance,
            (goal_states.shape[0], 3),
            device=self.device,
        )
        new_goal_pos = torch.clamp(
            new_goal_pos, target_volume_min_coord, target_volume_max_coord
        )

        new_goal_quat_xyzw = self.sample_delta_quat_xyzw(
            input_quat_xyzw=last_goal_quat_xyzw,
            delta_rotation_degrees=delta_rotation_degrees,
        )
        goal_states[:, 0:3] = new_goal_pos
        goal_states[:, 3:7] = new_goal_quat_xyzw
        return goal_states

    def _clip_goal_z(self, env_ids: Tensor):
        min_z = (
            self.object_init_state[env_ids, 2:3] - 0.05 + self.lifting_bonus_threshold
        )
        self.goal_states[env_ids, 2:3] = torch.max(
            min_z, self.goal_states[env_ids, 2:3]
        )

    def _reset_target(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        tensor_reset=True,
        is_first_goal=True,
    ) -> None:
        has_goal_object = bool(self.create_goal_object) and len(self.goal_object_indices) > 0
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            USE_FIXED_GOAL_STATES = self.cfg["env"]["useFixedGoalStates"]
            if USE_FIXED_GOAL_STATES:
                trajectory_state = self.trajectory_states.repeat(len(env_ids), 1, 1)
                current_subgoal_idx = (
                    self.successes[env_ids] % len(self.trajectory_states)
                ).long()
                batch_indices = torch.arange(len(env_ids), device=self.device)
                self.goal_states[env_ids, 0:7] = trajectory_state[
                    batch_indices, current_subgoal_idx, 0:7
                ]
            elif not is_first_goal and self.goal_sampling_type == "delta":
                self.goal_states[env_ids, 0:7] = self._sample_delta_goal(
                    self.goal_states[env_ids, 0:7],
                    self.delta_goal_distance,
                    self.delta_rotation_degrees,
                )
                # Actually don't clip goal z for delta poses to allow it to go below the lifted z
                # self._clip_goal_z(env_ids)
            elif not is_first_goal and self.goal_sampling_type == "coin_flip":
                # flip a coin. 50% of envs only get delta translation, 0 rotation and 50% get delta rotation, 0 translation
                coin_flips = torch_rand_float(
                    0.0, 1.0, (len(env_ids), 1), device=self.device
                )
                translation_only_goal_states = self._sample_delta_goal(
                    self.goal_states[env_ids, 0:7], self.delta_goal_distance, 0.0
                )
                rotation_only_goal_states = self._sample_delta_goal(
                    self.goal_states[env_ids, 0:7], 0.0, self.delta_rotation_degrees
                )
                self.goal_states[env_ids, 0:7] = torch.where(
                    coin_flips < 0.5,
                    translation_only_goal_states,
                    rotation_only_goal_states,
                )
                # Actually don't clip goal z for delta poses to allow it to go below the lifted z
                # self._clip_goal_z(env_ids)
            else:
                # Randomly sample a target pose
                target_volume_origin = self.target_volume_origin
                target_volume_extent = self.target_volume_extent

                target_volume_min_coord = (
                    target_volume_origin + target_volume_extent[:, 0]
                )
                target_volume_max_coord = (
                    target_volume_origin + target_volume_extent[:, 1]
                )
                target_volume_size = target_volume_max_coord - target_volume_min_coord

                rand_pos_floats = torch_rand_float(
                    0.0, 1.0, (len(env_ids), 3), device=self.device
                )
                target_coords = (
                    target_volume_min_coord + rand_pos_floats * target_volume_size
                )

                self.goal_states[env_ids, 0:3] = target_coords
                new_rot = self.get_random_quat(env_ids)
                self.goal_states[env_ids, 3:7] = new_rot
                self._clip_goal_z(env_ids)
            if has_goal_object:
                self.root_state_tensor[self.goal_object_indices[env_ids], 0:7] = (
                    self.goal_states[env_ids, 0:7]
                )
                self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = (
                    torch.zeros_like(
                        self.root_state_tensor[
                            self.goal_object_indices[env_ids], 7:13
                        ]
                    )
                )
        if (
            has_goal_object
            and len(env_ids) > 0
            and reset_buf_idxs is not None
            and tensor_reset
        ):
            # TODO: Check if last 6 indices are 0
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[self.goal_object_indices[env_ids], :] = (
                self.root_state_resets[
                    reset_buf_idxs[env_ids].cpu(),
                    self.goal_object_indices[env_ids].cpu() % rs_ofs,
                    :,
                ].to(self.device)
            )
            self.goal_states[env_ids, 0:7] = self.root_state_tensor[
                self.goal_object_indices[env_ids], 0:7
            ]

        if has_goal_object:
            self.deferred_set_actor_root_state_tensor_indexed(
                [self.goal_object_indices[env_ids]]
            )

        # HACK: Force the goal object pose to be the specified value
        if self.cfg["env"]["goalObjectPose"] is not None:
            desired_goal_object_pose = self.cfg["env"]["goalObjectPose"]
            assert len(desired_goal_object_pose) == 7, (
                f"desired_goal_object_pose must be a 7-element list, got {len(desired_goal_object_pose)}"
            )
            for i in range(7):
                self.goal_states[env_ids, i] = desired_goal_object_pose[i]
            if has_goal_object:
                self.root_state_tensor[self.goal_object_indices[env_ids], 0:7] = (
                    self.goal_states[env_ids, 0:7]
                )
                self.deferred_set_actor_root_state_tensor_indexed(
                    [self.goal_object_indices]
                )

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        if not self.create_goal_object or len(self.goal_object_indices) == 0:
            return []
        return [self.goal_object_indices[env_ids]]

    def _extra_curriculum(self):
        self.success_tolerance, self.last_curriculum_update = tolerance_curriculum(
            self.last_curriculum_update,
            self.frame_since_restart,
            self.tolerance_curriculum_interval,
            self.prev_episode_successes,
            self.success_tolerance,
            self.initial_tolerance,
            self.target_tolerance,
            self.tolerance_curriculum_increment,
        )

        eval_success_tolerance = self.cfg["env"].get("evalSuccessTolerance", None)
        if eval_success_tolerance is not None:
            self.success_tolerance = eval_success_tolerance

    # SimToolReal implementation
    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return dict(
            success_tolerance=self.success_tolerance,
            prev_episode_successes=self.prev_episode_successes,
            prev_episode_true_objective=self.prev_episode_true_objective,
            dof_state=self.dof_state,
            root_state_tensor=self.root_state_tensor,
            rigid_body_states=self.rigid_body_states,
            successes=self.successes,
            true_objective=self.true_objective,
            near_goal_steps=self.near_goal_steps,
            lifted_object=self.lifted_object,
            dynamic_grasp_success_steps=self.dynamic_grasp_success_steps,
            dynamic_grasp_object_xy_velocity=self.dynamic_grasp_object_xy_velocity,
            dynamic_grasp_object_yaw_rate=self.dynamic_grasp_object_yaw_rate,
            closest_keypoint_max_dist=self.closest_keypoint_max_dist,
            closest_keypoint_max_dist_fixed_size=self.closest_keypoint_max_dist_fixed_size,
            closest_fingertip_dist=self.closest_fingertip_dist,
            furthest_hand_dist=self.furthest_hand_dist,
            prev_targets=self.prev_targets,
            cur_targets=self.cur_targets,
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            reset_goal_buf=self.reset_goal_buf,
            obj_keypoint_pos=self.obj_keypoint_pos,
            goal_keypoint_pos=self.goal_keypoint_pos,
            obj_keypoint_pos_fixed_size=self.obj_keypoint_pos_fixed_size,
            goal_keypoint_pos_fixed_size=self.goal_keypoint_pos_fixed_size,
            rewards_episode=self.rewards_episode,
            last_curriculum_update=self.last_curriculum_update,
            rb_forces=self.rb_forces,
            rb_torques=self.rb_torques,
            random_force_prob=self.random_force_prob,
            random_torque_prob=self.random_torque_prob,
            random_lin_vel_impulse_prob=self.random_lin_vel_impulse_prob,
            random_ang_vel_impulse_prob=self.random_ang_vel_impulse_prob,
            goal_states=self.goal_states,
            goal_init_state=self.goal_init_state,
            object_init_state=self.object_init_state,
            prev_total_episode_closest_keypoint_max_dist=self.prev_total_episode_closest_keypoint_max_dist,
            total_episode_closest_keypoint_max_dist=self.total_episode_closest_keypoint_max_dist,
            prev_episode_closest_keypoint_max_dist=self.prev_episode_closest_keypoint_max_dist,
            frame_since_restart=self.frame_since_restart,
        )

    def set_env_state(self, env_state):
        if env_state is None:
            return

        had_shape_mismatch = False
        rewards_episode = env_state.get("rewards_episode", None)
        if rewards_episode is not None:
            for key in rewards_episode.keys():
                if (
                    key in self.rewards_episode
                    and self.rewards_episode[key].shape == rewards_episode[key].shape
                ):
                    self.rewards_episode[key].copy_(rewards_episode[key])
                else:
                    had_shape_mismatch = True
                    print(
                        "Skipping loading rewards_episode value",
                        key,
                        "because of shape mismatch",
                    )
        if "rewards_episode" in env_state:
            del env_state["rewards_episode"]

        for key in self.get_env_state().keys():
            value = env_state.get(key, None)
            if value is None:
                continue

            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
            if (
                isinstance(value, torch.Tensor)
                and self.__dict__[key].shape != value.shape
            ):
                had_shape_mismatch = True
                print(
                    "Skipping loading env state value", key, "because of shape mismatch"
                )
                continue

            if isinstance(value, torch.Tensor):
                self.__dict__[key].copy_(value)
            else:
                self.__dict__[key] = value
            print(f"Loaded env state value {key}:{value}")

        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_hand_arm_dofs
        ]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(all_env_ids, tensor_reset=had_shape_mismatch)
        self.set_actor_root_state_tensor_indexed()
        if had_shape_mismatch:
            self.set_dof_state_tensor_indexed()
            self.populate_sim_buffers()
            self.populate_obs_and_states_buffers()
            self.clamp_obs()
            print(
                "Checkpoint env state shape mismatched current env; reset all envs "
                "instead of reusing partial saved state."
            )
        print(
            f"Success tolerance value after loading from checkpoint: {self.success_tolerance}"
        )

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.control_dt = self.dt * self.control_freq_inv
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # If randomizing, apply once immediately on startup before the first sim step
        # Necessary for setup_only=True properties
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _handle_head_primitives(self, generated_assets_dir):
        if not os.path.exists(generated_assets_dir):
            os.makedirs(generated_assets_dir)

        try:
            filenames = os.listdir(generated_assets_dir)
            for fname in filenames:
                if fname.endswith(".urdf"):
                    os.remove(join(generated_assets_dir, fname))
        except Exception as exc:
            print(
                f"Exception {exc} while removing older procedurally-generated urdf assets"
            )

        # We are generating "handle_head" objects, which consist of a handle and a head
        # The handle and head are either a cuboid or a cylinder
        # The origin of the object is at the center of the handle
        # To have different densities, we would need to have 2 links each with a different density, but this breaks the current code
        # An alternative could be to manually compute the mass and inertia rather than using the density field alone
        # We implement this variable density approach now

        # The x-direction is along the handle
        # The head is at +x from the handle
        # There is no relative rotation between the handle and head

        NUM_OBJECTS_PER_TYPE = 100
        np.random.seed(42)

        from isaacgymenvs.tasks.simtoolreal.generate_objects import (
            generate_handle_head_urdf,
        )
        from isaacgymenvs.tasks.simtoolreal.object_size_distributions import (
            OBJECT_SIZE_DISTRIBUTIONS,
        )

        handle_head_types = set(self.cfg["env"]["handleHeadTypes"])
        object_size_distributions = [
            obj for obj in OBJECT_SIZE_DISTRIBUTIONS if obj.type in handle_head_types
        ]

        files_list = []
        scales_list = []
        for object_size_distribution in object_size_distributions:
            handle_head_type = object_size_distribution.type

            # Sample densities
            handle_densities = object_size_distribution.sample_handle_densities(
                NUM_OBJECTS_PER_TYPE
            )
            head_densities = object_size_distribution.sample_head_densities(
                NUM_OBJECTS_PER_TYPE
            )

            # Sample scales
            # Currently different for each object
            handle_scales = object_size_distribution.sample_handle_scales(
                NUM_OBJECTS_PER_TYPE
            )
            head_scales = object_size_distribution.sample_head_scales(
                NUM_OBJECTS_PER_TYPE
            )
            assert handle_scales.shape in [
                (NUM_OBJECTS_PER_TYPE, 2),
                (NUM_OBJECTS_PER_TYPE, 3),
            ], (
                f"handle_scales shape: {handle_scales.shape}, expected ({NUM_OBJECTS_PER_TYPE}, 2) or ({NUM_OBJECTS_PER_TYPE}, 3)"
            )
            files_list.append(
                [
                    generate_handle_head_urdf(
                        filepath=Path(generated_assets_dir)
                        / (
                            f"{idx:03d}_{handle_head_type}_handle_head_{handle_scales[idx]}_{head_scales[idx] if head_scales is not None else 'None'}_{handle_densities[idx]}_{head_densities[idx] if head_densities is not None else 'None'}".replace(
                                ".", "-"
                            )
                            + ".urdf"
                        ),
                        handle_scale=handle_scales[idx],
                        head_scale=head_scales[idx]
                        if head_scales is not None
                        else None,
                        handle_density=handle_densities[idx],
                        head_density=head_densities[idx]
                        if head_scales is not None
                        else None,
                    )
                    for idx in range(NUM_OBJECTS_PER_TYPE)
                ]
            )
            scales_list.append(handle_scales)

        all_files = [file for sublist in files_list for file in sublist]
        all_scales = [scale for sublist in scales_list for scale in sublist]

        def convert_scale_to_three_elements(scale):
            # Object scales must have 3 elements
            # Cylinders currently have 2 elements, so we make the third element the same as the second element
            if len(scale) == 3:
                return scale
            elif len(scale) == 2:
                return (scale[0], scale[1], scale[1])
            else:
                raise ValueError(f"Invalid scale: {scale}")

        all_scales = [convert_scale_to_three_elements(scale) for scale in all_scales]
        need_vhacds = [False] * len(all_files)

        # Note, we need to make sure all_scales is rescaled by the base size
        all_scales = [
            (
                x / self.object_base_size,
                y / self.object_base_size,
                z / self.object_base_size,
            )
            for (x, y, z) in all_scales
        ]

        # Randomize order
        RANDOMIZE_ORDER = True
        if RANDOMIZE_ORDER:
            indices = list(range(len(all_files)))
            np.random.shuffle(indices)
            all_files = [all_files[i] for i in indices]
            all_scales = [all_scales[i] for i in indices]
            need_vhacds = [need_vhacds[i] for i in indices]

        DEBUG_PRINT = False
        if DEBUG_PRINT:
            print(f"all_files[0]: {all_files[0]}")
            print(f"all_scales[0]: {all_scales[0]}")
            print(f"need_vhacds[0]: {need_vhacds[0]}")
            # print(f"All files: {all_files}")
            # print(f"All scales: {all_scales}")

        return all_files, all_scales, need_vhacds

    def _create_envs(self, num_envs, spacing, num_per_row):
        if self.should_load_initial_states:
            self.load_initial_states()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../../assets"
        )
        robot_asset_root = asset_root
        robot_asset_file = self.robot_asset_file
        if os.path.isabs(robot_asset_file):
            robot_asset_root = os.path.dirname(robot_asset_file)
            robot_asset_file = os.path.basename(robot_asset_file)
        elif self.robot_asset_root_cfg is not None:
            robot_asset_root = os.path.expanduser(str(self.robot_asset_root_cfg))
        robot_asset_path = os.path.join(robot_asset_root, robot_asset_file)
        if not os.path.exists(robot_asset_path):
            raise FileNotFoundError(
                f"Robot asset not found: {robot_asset_path}. "
                "Set env.robotAssetRoot/env.asset.robot or generate the asset first."
            )

        object_asset_root = asset_root
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales, self.object_need_vhacds = (
            self._main_object_assets_and_scales(object_asset_root, tmp_assets_dir.name)
        )

        asset_options = gymapi.AssetOptions()
        # asset_options.vhacd_enabled = True  # Should be False so the robot is not complicated to model, but can test
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = bool(
            self.cfg["env"].get("assetCollapseFixedJoints", True)
        )
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {robot_asset_file} from {robot_asset_root}")
        robot_asset = self.gym.load_asset(
            self.sim, robot_asset_root, robot_asset_file, asset_options
        )
        print(f"Loaded asset {robot_asset}")

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(robot_asset)
        assert self.num_hand_arm_dofs == num_hand_arm_dofs, (
            f"Number of DOFs in asset {robot_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"
        )

        max_agg_bodies = self.num_hand_arm_bodies
        max_agg_shapes = self.num_hand_arm_shapes

        robot_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(robot_asset, i)
            for i in range(self.num_hand_arm_bodies)
        ]
        print(f"Robot num rigid bodies: {self.num_hand_arm_bodies}")
        print(f"Robot rigid bodies: {robot_rigid_body_names}")

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []

        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(robot_dof_props["upper"][i])

        self.arm_hand_dof_lower_limits = to_torch(
            self.arm_hand_dof_lower_limits, device=self.device
        )
        self.arm_hand_dof_upper_limits = to_torch(
            self.arm_hand_dof_upper_limits, device=self.device
        )

        robot_start_pose_cfg = self.cfg["env"].get("robotStartPose", None)
        if robot_start_pose_cfg is None:
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(
                *get_axis_params(0.0, self.up_axis_idx)
            ) + gymapi.Vec3(0.0, 0.8, 0)
            robot_pose.r = gymapi.Quat(0, 0, 0, 1)
        else:
            robot_pose = cfg_pose_to_transform(robot_start_pose_cfg, "robotStartPose")

        object_assets, object_rb_count, object_shapes_count = (
            self._load_main_object_asset()
        )
        max_agg_bodies += object_rb_count
        max_agg_shapes += object_shapes_count

        # load auxiliary objects
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(
            self.sim, asset_root, self.asset_files_dict["table"], table_asset_options
        )

        if self.with_table_force_sensor:
            table_sensor_pose = gymapi.Transform()
            table_sensor_props = gymapi.ForceSensorProperties()
            # If both enable_constraint_solver_forces=False and enable_forward_dynamics_forces=False, always will have a force of 0.0
            table_sensor_props.enable_constraint_solver_forces = True  # Defaults True, keep True to get constraint forces to get contact forces
            table_sensor_props.enable_forward_dynamics_forces = False  # Defaults True, but set to False to avoid gravity being part of this force. Can be True if have disable_gravity=True for the table
            table_sensor_props.use_world_frame = True
            self.table_sensor_idx = self.gym.create_asset_force_sensor(
                asset=table_asset,
                body_idx=0,
                local_pose=table_sensor_pose,
                props=table_sensor_props,
            )
            if self.table_sensor_idx == -1:
                raise ValueError("Failed to create table force sensor")

        table_start_pose_cfg = self.cfg["env"].get("tableStartPose", None)
        if table_start_pose_cfg is None:
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3()
            table_pose.p.x = robot_pose.p.x
            table_pose_dy, table_pose_dz = -0.8, self.cfg["env"]["tableResetZ"]
            table_pose.p.y = robot_pose.p.y + table_pose_dy
            table_pose.p.z = robot_pose.p.z + table_pose_dz
        else:
            table_pose = cfg_pose_to_transform(table_start_pose_cfg, "tableStartPose")
            table_pose_dy = table_pose.p.y - robot_pose.p.y
            table_pose_dz = table_pose.p.z - robot_pose.p.z

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        additional_rb, additional_shapes = self._load_additional_assets(
            object_asset_root, robot_pose
        )
        max_agg_bodies += additional_rb
        max_agg_shapes += additional_shapes

        # set up object and goal positions
        self.object_start_pose = self._object_start_pose(
            robot_pose, table_pose_dy, table_pose_dz
        )

        self.robots = []
        self.envs = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robots = []
        self.objects = []

        object_init_state = []
        table_init_state = []

        self.rigid_body_name_to_idx = {}

        self.robot_indices = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = []
        object_indices = []
        table_indices = []
        object_scales = []
        object_asset_indices = []
        object_keypoint_offsets = []
        object_keypoint_offsets_fixed_size = []

        # Sanity checks
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        for name in self.fingertips:
            assert name in body_names, (
                f"Fingertip body {name} not found in asset {robot_asset}; "
                f"available bodies: {body_names}"
            )
        assert self.palm_body_name in body_names, (
            f"Palm body {self.palm_body_name} not found in asset {robot_asset}; "
            f"available bodies: {body_names}"
        )

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.fingertips
        ]

        if self.with_fingertip_force_sensors:
            finger_sensor_pose = gymapi.Transform()
            self.finger_sensor_idxs = [
                self.gym.create_asset_force_sensor(
                    asset=robot_asset, body_idx=ft_handle, local_pose=finger_sensor_pose
                )
                for ft_handle in self.fingertip_handles
            ]

        self.robot_name = self.arm_embodiment
        self.palm_handle = self.gym.find_asset_rigid_body_index(
            robot_asset, self.palm_body_name
        )

        # this rely on the fact that objects are added right after the arms in terms of create_actor()
        self.object_rb_handles = list(
            range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count)
        )
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            # Account for the blue robot's additional rigid bodies
            self.object_rb_handles = list(
                range(
                    2 * self.num_hand_arm_bodies,
                    2 * self.num_hand_arm_bodies + object_rb_count,
                )
            )

        # Set asset rigid shape properties (friction)
        MODIFY_ASSET_FRICTIONS = self.cfg["env"]["modifyAssetFrictions"]

        if MODIFY_ASSET_FRICTIONS:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=self.cfg["env"]["robotFriction"],
                fingertip_friction=self.cfg["env"]["fingerTipFriction"],
            )
            self.set_table_asset_rigid_shape_properties(
                table_asset=table_asset,
                friction=self.cfg["env"]["tableFriction"],
            )
            for object_asset_idx_to_modify in range(len(object_assets)):
                self.set_object_asset_rigid_shape_properties(
                    object_asset=object_assets[object_asset_idx_to_modify],
                    friction=self.cfg["env"]["objectFriction"],
                )
        else:
            # Still run this because it sets the collision filters to avoid self-collisions between adjacent links
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=None,
                fingertip_friction=None,
            )

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            collision_group = i  # Each env has a different collision group so they don't collide with each other
            collision_filter = -1  # -1 = use asset collision filters set in mjcf loader (for URDF, enable all self-collisions)
            # 0 = enable all self-collisions
            # >0 = disable all self-collisions
            segmentation_id = 0
            robot_actor = self.gym.create_actor(
                env_ptr,
                robot_asset,
                robot_pose,
                "robot",
                collision_group,
                collision_filter,
                segmentation_id,
            )
            if self.robot_dof_property_preset == "legacy_kuka_sharpa":
                populate_dof_properties(
                    robot_dof_props, self.num_arm_dofs, self.num_hand_dofs
                )
            elif self.robot_dof_property_preset == "generic_position":
                populate_generic_dof_properties(
                    robot_dof_props,
                    self.num_arm_dofs,
                    self.num_hand_dofs,
                    arm_stiffness=self.cfg["env"].get("robotArmStiffness", 600.0),
                    arm_damping=self.cfg["env"].get("robotArmDamping", 60.0),
                    arm_effort=self.cfg["env"].get("robotArmEffort", 87.0),
                    arm_armature=self.cfg["env"].get("robotArmArmature", 0.0),
                    arm_friction=self.cfg["env"].get("robotArmJointFriction", 0.0),
                    hand_stiffness=self.cfg["env"].get("robotHandStiffness", 40.0),
                    hand_damping=self.cfg["env"].get("robotHandDamping", 4.0),
                    hand_effort=self.cfg["env"].get("robotHandEffort", 20.0),
                    hand_armature=self.cfg["env"].get("robotHandArmature", 0.0),
                    hand_friction=self.cfg["env"].get("robotHandJointFriction", 0.01),
                )
            else:
                raise ValueError(
                    f"Unknown robotDofPropertyPreset={self.robot_dof_property_preset}"
                )

            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(
                env_ptr, robot_actor, gymapi.DOMAIN_SIM
            )
            self.robot_indices.append(robot_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, robot_actor):
                self.rigid_body_name_to_idx["robot/" + name] = (
                    self.gym.find_actor_rigid_body_index(
                        env_ptr, robot_actor, name, gymapi.DOMAIN_ENV
                    )
                )

            if self.with_dof_force_sensors:
                self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)

            if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                blue_robot_actor = self.gym.create_actor(
                    env_ptr,
                    robot_asset,
                    robot_pose,
                    "blue_robot",
                    i + self.num_envs * 2,
                    -1,
                    0,
                )
                self.gym.set_actor_dof_properties(
                    env_ptr,
                    blue_robot_actor,
                    robot_dof_props,
                )
                self.blue_robots.append(blue_robot_actor)
                BLUE = (0, 0, 1)
                self._set_actor_color(env_ptr, blue_robot_actor, BLUE)

                blue_robot_idx = self.gym.get_actor_index(
                    env_ptr, blue_robot_actor, gymapi.DOMAIN_SIM
                )
                self.blue_robot_indices.append(blue_robot_idx)

            # add object
            object_asset_idx = i % len(object_assets)
            object_asset = object_assets[object_asset_idx]
            object_asset_indices.append(object_asset_idx)

            object_start_pose_for_env = gymapi.Transform()
            object_start_pose_for_env.p = gymapi.Vec3(
                self.object_start_pose.p.x,
                self.object_start_pose.p.y,
                self.object_start_pose.p.z,
            )
            object_start_pose_for_env.r = gymapi.Quat(
                self.object_start_pose.r.x,
                self.object_start_pose.r.y,
                self.object_start_pose.r.z,
                self.object_start_pose.r.w,
            )
            if hasattr(self, "object_asset_table_z_offsets_np"):
                object_start_pose_for_env.p.z = table_pose.p.z + float(
                    self.object_asset_table_z_offsets_np[object_asset_idx]
                )
            if hasattr(self, "object_asset_default_quats_xyzw_np"):
                object_quat_xyzw = self.object_asset_default_quats_xyzw_np[
                    object_asset_idx
                ]
                object_start_pose_for_env.r = gymapi.Quat(
                    float(object_quat_xyzw[0]),
                    float(object_quat_xyzw[1]),
                    float(object_quat_xyzw[2]),
                    float(object_quat_xyzw[3]),
                )

            object_handle = self.gym.create_actor(
                env_ptr,
                object_asset,
                object_start_pose_for_env,
                "object",
                i,
                0,
                0,
            )
            object_init_state.append(
                [
                    object_start_pose_for_env.p.x,
                    object_start_pose_for_env.p.y,
                    object_start_pose_for_env.p.z,
                    object_start_pose_for_env.r.x,
                    object_start_pose_for_env.r.y,
                    object_start_pose_for_env.r.z,
                    object_start_pose_for_env.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            object_indices.append(object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                self.rigid_body_name_to_idx["object/" + name] = (
                    self.gym.find_actor_rigid_body_index(
                        env_ptr, object_handle, name, gymapi.DOMAIN_ENV
                    )
                )

            object_scale = self.object_asset_scales[object_asset_idx]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= (
                        object_scale[coord_idx]
                        * self.object_base_size
                        * self.keypoint_scale
                        / 2
                    )
                object_offsets.append(keypoint)
            object_keypoint_offsets.append(object_offsets)

            # We make a version of keypoint offsets that are a fixed size for all objects
            object_scale_fixed_size = self.cfg["env"]["fixedSize"]
            assert len(object_scale_fixed_size) == 3, (
                f"object_scale_fixed_size must be a 3-element list, got {len(object_scale_fixed_size)}"
            )
            object_offsets_fixed_size = []
            for keypoint in self.keypoints_offsets:
                keypoint_fixed_size = copy(keypoint)
                for coord_idx in range(3):
                    keypoint_fixed_size[coord_idx] *= (
                        object_scale_fixed_size[coord_idx] * self.keypoint_scale / 2
                    )  # Don't multiply by object_base_size here since it's already metric scale
                object_offsets_fixed_size.append(keypoint_fixed_size)
            object_keypoint_offsets_fixed_size.append(object_offsets_fixed_size)

            # table object
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table_object", i, 0, 0
            )
            table_init_state.append(
                [
                    table_pose.p.x,
                    table_pose.p.y,
                    table_pose.p.z,
                    table_pose.r.x,
                    table_pose.r.y,
                    table_pose.r.z,
                    table_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            table_object_idx = self.gym.get_actor_index(
                env_ptr, table_handle, gymapi.DOMAIN_SIM
            )
            table_indices.append(table_object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                self.rigid_body_name_to_idx["table/" + name] = (
                    self.gym.find_actor_rigid_body_index(
                        env_ptr, table_handle, name, gymapi.DOMAIN_ENV
                    )
                )

            # task-specific objects (i.e. goal object for reorientation task)
            self._create_additional_objects(
                env_ptr,
                env_idx=i,
                object_asset_idx=object_asset_idx,
                object_start_pose=object_start_pose_for_env,
            )

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(object_handle)

        # Default false because this is slow
        DEBUG_PRINT_OBJECT_MASS_AND_INERTIA_AND_COM = False
        # Get mass and inertia of object
        if DEBUG_PRINT_OBJECT_MASS_AND_INERTIA_AND_COM:
            original_masses, original_inertias, original_coms = (
                self._get_original_object_masses_and_inertias_and_coms()
            )
            print(f"Original masses: {original_masses[0]}")
            print(f"Original inertias: {original_inertias[0]}")
            print(f"Original coms: {original_coms[0]}")
            breakpoint()

        # Set mass and inertia of object
        MODIFY_OBJECT_MASS_AND_INERTIA = False
        if MODIFY_OBJECT_MASS_AND_INERTIA:
            # Get mass and inertia of object
            original_masses, original_inertias, original_coms = (
                self._get_original_object_masses_and_inertias_and_coms()
            )
            print(f"Original masses: {original_masses[0]}")
            print(f"Original inertias: {original_inertias[0]}")

            self.set_object_masses_and_inertias(
                envs=self.envs,
                objects=self.objects,
                masses=[0.2] * len(self.objects),
                inertias=[(0.001, 0.001, 0.001)] * len(self.objects),
            )

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(
            self.envs[0], object_handle
        )
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.table_init_state = to_torch(
            table_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        object_asset_indices_np = np.asarray(object_asset_indices, dtype=np.int64)
        if (
            self.object_pointcloud_use_mesh_surface
            and hasattr(self, "object_asset_pointcloud_points_np")
        ):
            self.object_pointcloud_unit_points_np = self.object_asset_pointcloud_points_np[
                object_asset_indices_np
            ].astype(np.float32)
            self.object_pointcloud_unit_normals_np = self.object_asset_pointcloud_normals_np[
                object_asset_indices_np
            ].astype(np.float32)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.object_rb_handles = to_torch(
            self.object_rb_handles, dtype=torch.long, device=self.device
        )
        self.object_rb_masses = to_torch(
            self.object_rb_masses, dtype=torch.float, device=self.device
        )

        self.robot_indices = to_torch(
            self.robot_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            object_indices, dtype=torch.long, device=self.device
        )
        self.table_indices = to_torch(
            table_indices, dtype=torch.long, device=self.device
        )
        self.object_asset_indices = to_torch(
            object_asset_indices, dtype=torch.long, device=self.device
        )
        self.object_affordance_surface_offsets = to_torch(
            self._dynamic_affordance_surface_offsets_for_assets(object_asset_indices_np),
            dtype=torch.float,
            device=self.device,
        )
        if hasattr(self, "object_asset_default_quats_xyzw_np"):
            self.object_default_rot = to_torch(
                self.object_asset_default_quats_xyzw_np[object_asset_indices_np],
                dtype=torch.float,
                device=self.device,
            )
            self.object_random_yaw = to_torch(
                self.object_asset_random_yaw_np[object_asset_indices_np],
                dtype=torch.bool,
                device=self.device,
            )
            self.object_table_z_offsets = to_torch(
                self.object_asset_table_z_offsets_np[object_asset_indices_np],
                dtype=torch.float,
                device=self.device,
            )
            self.object_asset_labels_for_env = [
                self.object_asset_labels[asset_idx] for asset_idx in object_asset_indices
            ]
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = to_torch(
                self.blue_robot_indices, dtype=torch.long, device=self.device
            )

        self.object_scales = to_torch(
            object_scales, dtype=torch.float, device=self.device
        )
        self.object_keypoint_offsets = to_torch(
            object_keypoint_offsets, dtype=torch.float, device=self.device
        )
        self.object_keypoint_offsets_fixed_size = to_torch(
            object_keypoint_offsets_fixed_size, dtype=torch.float, device=self.device
        )

        self.joint_names = self.gym.get_actor_joint_names(env_ptr, robot_actor)
        props = self.gym.get_actor_dof_properties(env_ptr, robot_actor)
        self.joint_lower_limits = props["lower"]
        self.joint_upper_limits = props["upper"]

        print(f"Robot joint names: {self.joint_names}")

        self._after_envs_created()

        try:
            # by this point we don't need the temporary folder for procedurally generated assets
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    def _get_original_object_masses_and_inertias_and_coms(
        self,
    ) -> Tuple[
        List[float], List[Tuple[float, float, float]], List[Tuple[float, float, float]]
    ]:
        original_masses, original_inertias, original_coms = [], [], []
        for env, object in zip(self.envs, self.objects):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            assert len(object_rb_props) == 1, (
                f"Expected 1 rigid body, got {len(object_rb_props)}"
            )
            object_rb_prop = object_rb_props[0]
            original_mass = object_rb_prop.mass
            original_inertia = (
                object_rb_prop.inertia.x.x,
                object_rb_prop.inertia.y.y,
                object_rb_prop.inertia.z.z,
            )
            original_com = (
                object_rb_prop.com.x,
                object_rb_prop.com.y,
                object_rb_prop.com.z,
            )
            original_masses.append(original_mass)
            original_inertias.append(original_inertia)
            original_coms.append(original_com)
        return original_masses, original_inertias, original_coms

    def _set_actor_color(self, env, actor, color: Tuple[float, float, float]) -> None:
        for rigid_body_idx in range(self.gym.get_actor_rigid_body_count(env, actor)):
            self.gym.set_rigid_body_color(
                env,
                actor,
                rigid_body_idx,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*color),
            )

    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""
        # this is positive if we got closer, negative if we're further away than the closest we've gotten
        fingertip_deltas_closest = (
            self.closest_fingertip_dist - self.curr_fingertip_distances
        )
        # update the values if finger tips got closer to the object
        self.closest_fingertip_dist = torch.minimum(
            self.closest_fingertip_dist, self.curr_fingertip_distances
        )

        # again, positive is closer, negative is further away
        # here we use index of the 1st finger, when the distance is large it doesn't matter which one we use
        hand_deltas_furthest = (
            self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        )
        # update the values if finger tips got further away from the object
        self.furthest_hand_dist = torch.maximum(
            self.furthest_hand_dist, self.curr_fingertip_distances[:, 0]
        )

        # clip between zero and +inf to turn deltas into rewards
        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_deltas *= self.finger_rew_coeffs
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        fingertip_delta_rew *= ~lifted_object

        # clip between zero and -inf to turn deltas into penalties
        hand_delta_penalty = torch.clip(hand_deltas_furthest, -10, 0)
        hand_delta_penalty *= ~lifted_object
        # multiply by the number of fingers so two rewards are on the same scale
        hand_delta_penalty *= self.num_fingertips

        return fingertip_delta_rew, hand_delta_penalty

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""

        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        # this flag tells us if we lifted an object above a certain height compared to the initial position
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self.lifted_object

        # Since we stop rewarding the agent for height after the object is lifted, we should give it large positive reward
        # to compensate for "lost" opportunity to get more lifting reward for sitting just below the threshold.
        # This bonus depends on the max lifting reward (lifting reward coeff * threshold) and the discount factor
        # (i.e. the effective future horizon for the agent)
        # For threshold 0.15, lifting reward coeff = 3 and gamma 0.995 (effective horizon ~500 steps)
        # a value of 300 for the bonus reward seems reasonable
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold

        # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the
        # keypoint reward
        lifting_rew *= ~lifted_object

        # update the flag that describes whether we lifted an object above the table or not
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object

    def _keypoint_reward(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        # this is positive if we got closer, negative if we're further away
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist
        max_keypoint_deltas_fixed_size = (
            self.closest_keypoint_max_dist_fixed_size
            - self.keypoints_max_dist_fixed_size
        )

        # update the values if we got closer to the target
        self.closest_keypoint_max_dist = torch.minimum(
            self.closest_keypoint_max_dist, self.keypoints_max_dist
        )
        self.closest_keypoint_max_dist_fixed_size = torch.minimum(
            self.closest_keypoint_max_dist_fixed_size,
            self.keypoints_max_dist_fixed_size,
        )

        # clip between zero and +inf to turn deltas into rewards
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)
        max_keypoint_deltas_fixed_size = torch.clip(
            max_keypoint_deltas_fixed_size, 0, 100
        )

        # administer reward only when we already lifted an object from the table
        # to prevent the situation where the agent just rolls it around the table
        keypoint_rew = max_keypoint_deltas * lifted_object
        keypoint_rew_fixed_size = max_keypoint_deltas_fixed_size * lifted_object

        return keypoint_rew, keypoint_rew_fixed_size

    def _action_penalties(self) -> Tuple[Tensor, Tensor]:
        kuka_actions_penalty = (
            torch.sum(
                torch.abs(self.arm_hand_dof_vel[..., 0 : self.num_arm_dofs]),
                dim=-1,
            )
            * self.kuka_actions_penalty_scale
        )
        hand_actions_penalty = (
            torch.sum(
                torch.abs(
                    self.arm_hand_dof_vel[
                        ..., self.num_arm_dofs : self.num_hand_arm_dofs
                    ]
                ),
                dim=-1,
            )
            * self.hand_actions_penalty_scale
        )

        return -1 * kuka_actions_penalty, -1 * hand_actions_penalty

    def _dynamic_grasp_speed_curriculum_alpha(self) -> float:
        if (
            not self.dynamic_grasp_speed_curriculum
            or self.dynamic_grasp_speed_curriculum_steps <= 0
        ):
            return 1.0

        return min(
            float(self.frame_since_restart)
            / float(self.dynamic_grasp_speed_curriculum_steps),
            1.0,
        )

    def _dynamic_grasp_current_motion_ranges(self):
        alpha = self._dynamic_grasp_speed_curriculum_alpha()

        def lerp_range(start_range, target_range):
            start_min, start_max = start_range
            target_min, target_max = target_range
            return (
                start_min + (target_min - start_min) * alpha,
                start_max + (target_max - start_max) * alpha,
            )

        speed_range = lerp_range(
            self.dynamic_grasp_start_speed_range,
            self.dynamic_grasp_initial_speed_range,
        )
        yaw_rate_range = lerp_range(
            self.dynamic_grasp_start_yaw_rate_range,
            self.dynamic_grasp_initial_yaw_rate_range,
        )
        return speed_range, yaw_rate_range, alpha

    def _set_dynamic_object_initial_velocity(
        self, env_ids: Tensor, obj_indices: Tensor
    ) -> None:
        """Give reset objects a planar tabletop velocity for dynamic grasping."""
        if len(env_ids) == 0:
            return

        speed_range, yaw_rate_range, _ = self._dynamic_grasp_current_motion_ranges()
        min_speed, max_speed = speed_range
        min_yaw_rate, max_yaw_rate = yaw_rate_range
        speeds = torch_rand_float(
            min_speed, max_speed, (len(env_ids), 1), device=self.device
        )
        headings = torch_rand_float(
            -math.pi, math.pi, (len(env_ids), 1), device=self.device
        )
        planar_vel = torch.cat(
            [torch.cos(headings) * speeds, torch.sin(headings) * speeds], dim=-1
        )
        yaw_rates = torch_rand_float(
            min_yaw_rate, max_yaw_rate, (len(env_ids), 1), device=self.device
        )

        self.root_state_tensor[obj_indices, 7:13] = 0.0
        self.root_state_tensor[obj_indices, 7:9] = planar_vel
        self.root_state_tensor[obj_indices, 12:13] = yaw_rates
        self.dynamic_grasp_object_xy_velocity[env_ids] = planar_vel
        self.dynamic_grasp_object_yaw_rate[env_ids] = yaw_rates.squeeze(-1)

    def _apply_dynamic_tabletop_motion(self) -> None:
        """Keep unlifted dynamic-grasp objects sliding despite tabletop friction."""
        if not self.dynamic_tabletop_grasp or not self.dynamic_grasp_persistent_motion:
            return

        active_env_ids = (~self.lifted_object).nonzero(as_tuple=False).squeeze(-1)
        if len(active_env_ids) == 0:
            return

        obj_indices = self.object_indices[active_env_ids]
        desired_xy_vel = self.dynamic_grasp_object_xy_velocity[active_env_ids].clone()

        if self.dynamic_grasp_bounce_at_workspace:
            x_min, x_max = self.dynamic_grasp_workspace_x
            y_min, y_max = self.dynamic_grasp_workspace_y
            object_xy = self.root_state_tensor[obj_indices, 0:2]
            hit_x = (
                ((object_xy[:, 0] <= x_min) & (desired_xy_vel[:, 0] < 0.0))
                | ((object_xy[:, 0] >= x_max) & (desired_xy_vel[:, 0] > 0.0))
            )
            hit_y = (
                ((object_xy[:, 1] <= y_min) & (desired_xy_vel[:, 1] < 0.0))
                | ((object_xy[:, 1] >= y_max) & (desired_xy_vel[:, 1] > 0.0))
            )
            desired_xy_vel[hit_x, 0] *= -1.0
            desired_xy_vel[hit_y, 1] *= -1.0
            self.dynamic_grasp_object_xy_velocity[active_env_ids] = desired_xy_vel

        self.root_state_tensor[obj_indices, 7:9] = desired_xy_vel
        self.root_state_tensor[obj_indices, 12] = self.dynamic_grasp_object_yaw_rate[
            active_env_ids
        ]
        self.deferred_set_actor_root_state_tensor_indexed([obj_indices])

    def _dynamic_tabletop_workspace_resets(self) -> Tensor:
        """Reset dynamic-grasp episodes when the moving object leaves the table area."""
        x_min, x_max = self.dynamic_grasp_workspace_x
        y_min, y_max = self.dynamic_grasp_workspace_y
        object_out_of_workspace = (
            (self.object_pos[:, 0] < x_min)
            | (self.object_pos[:, 0] > x_max)
            | (self.object_pos[:, 1] < y_min)
            | (self.object_pos[:, 1] > y_max)
        )
        self.extras["reset/dynamic_object_out_of_workspace"] = (
            object_out_of_workspace.float().mean().item()
        )
        return object_out_of_workspace.to(dtype=self.reset_buf.dtype)

    def _log_dynamic_object_pool_metrics(
        self,
        is_success: Tensor,
        lifted_object: Tensor,
        stable_grasp: Tensor,
        lift_progress: Tensor,
        object_palm_rel_speed: Tensor,
        true_grasp: Optional[Tensor] = None,
        grasp_quality: Optional[Tensor] = None,
        scoop_lift: Optional[Tensor] = None,
        palm_only_lift: Optional[Tensor] = None,
    ) -> None:
        if (
            not self.log_dynamic_object_pool_metrics
            or not hasattr(self, "object_asset_indices")
            or not hasattr(self, "object_asset_labels")
        ):
            return
        if self.frame_since_restart % self.dynamic_object_pool_log_interval != 0:
            return

        mask_values = self.object_asset_indices
        metrics = {
            "success": is_success.float(),
            "lifted": lifted_object.float(),
            "stable_grasp": stable_grasp.float(),
            "lift_progress": lift_progress.float(),
            "object_palm_rel_speed": object_palm_rel_speed.float(),
        }
        if true_grasp is not None:
            metrics["true_grasp"] = true_grasp.float()
        if grasp_quality is not None:
            metrics["grasp_quality"] = grasp_quality.float()
        if scoop_lift is not None:
            metrics["scoop_lift"] = scoop_lift.float()
        if palm_only_lift is not None:
            metrics["palm_only_lift"] = palm_only_lift.float()
        for asset_idx, label in enumerate(self.object_asset_labels):
            env_mask = mask_values == asset_idx
            env_mask_float = env_mask.float()
            count = env_mask_float.sum().clamp(min=1.0)
            for metric_name, metric_value in metrics.items():
                self.extras[f"dynamic_object/{label}/{metric_name}"] = (
                    (metric_value * env_mask_float).sum() / count
                ).item()

    def _compute_dynamic_tabletop_grasp_reward(self) -> Tuple[Tensor, Tensor]:
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(
            lifted_object
        )

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        closest_fingertip_dist = self.curr_fingertip_distances.min(dim=-1).values
        mean_fingertip_dist = self.curr_fingertip_distances.mean(dim=-1)
        near_object_weight = torch.exp(
            -closest_fingertip_dist / self.dynamic_grasp_near_scale
        )

        object_xy_vel = self.object_linvel[:, 0:2]
        palm_xy_vel = self._palm_state[:, 7:9]
        rel_xy_speed = torch.norm(palm_xy_vel - object_xy_vel, dim=-1)
        dynamic_velocity_match_rew = (
            torch.exp(-rel_xy_speed / self.dynamic_grasp_velocity_match_scale)
            * near_object_weight
            * (~lifted_object)
        )

        object_xy_speed = torch.norm(object_xy_vel, dim=-1, keepdim=True)
        object_xy_dir = object_xy_vel / torch.clamp(object_xy_speed, min=1e-4)
        affordance_targets = self._dynamic_affordance_targets(
            object_pos=self.object_pos,
            object_vel=self.object_state[:, 7:13],
            palm_pos=self.palm_center_pos,
        )
        if self.dynamic_grasp_use_affordance_prior:
            predicted_object_xy = affordance_targets["future_pos"][:, 0:2]
            object_xy_dir = affordance_targets["axis_world"][:, 0:2]
        else:
            predicted_object_xy = (
                self.object_pos[:, 0:2]
                + object_xy_vel * self.dynamic_grasp_intercept_lead_time
            )
        palm_intercept_xy_dist = torch.norm(
            self.palm_center_pos[:, 0:2] - predicted_object_xy, dim=-1
        )
        dynamic_intercept_rew = (
            torch.exp(
                -palm_intercept_xy_dist
                / self.dynamic_grasp_intercept_distance_scale
            )
            * (~lifted_object)
        )

        dynamic_affordance_target_rew = (
            torch.exp(
                -palm_intercept_xy_dist
                / max(float(self.dynamic_grasp_intercept_distance_scale), 1e-6)
            )
            * (~lifted_object).float()
            * affordance_targets["confidence"].squeeze(-1)
        )
        pregrasp_target_xy = (
            predicted_object_xy
            + object_xy_dir * self.dynamic_grasp_pregrasp_ahead_distance
        )
        palm_pregrasp_xy_dist = torch.norm(
            self.palm_center_pos[:, 0:2] - pregrasp_target_xy, dim=-1
        )
        dynamic_pregrasp_alignment_rew = (
            torch.exp(
                -palm_pregrasp_xy_dist
                / self.dynamic_grasp_pregrasp_distance_scale
            )
            * (~lifted_object)
        )

        dynamic_enclosure_base_rew = torch.exp(
            -mean_fingertip_dist / self.dynamic_grasp_enclosure_distance_scale
        )
        dynamic_enclosure_rew = dynamic_enclosure_base_rew * (~lifted_object)
        dynamic_lifted_enclosure_rew = dynamic_enclosure_base_rew * lifted_object

        lift_target_height = max(self.lifting_bonus_threshold - 0.05, 1e-3)
        lift_progress = torch.clamp(
            (self.object_pos[:, 2] - self.object_init_state[:, 2])
            / lift_target_height,
            0.0,
            1.0,
        )
        dynamic_lift_progress_rew = (
            lift_progress * dynamic_enclosure_base_rew * (~lifted_object)
        )
        dynamic_safe_action_delta_penalty = torch.sum(
            torch.square(self.action_delta_for_penalty), dim=-1
        )
        dynamic_safe_arm_target_delta_penalty = torch.sum(
            torch.square(self.target_delta_for_penalty[:, : self.num_arm_dofs]),
            dim=-1,
        )
        dynamic_safe_arm_target_accel_penalty = torch.sum(
            torch.square(self.target_accel_for_penalty[:, : self.num_arm_dofs]),
            dim=-1,
        )
        dynamic_safe_hand_target_delta_penalty = torch.sum(
            torch.square(
                self.target_delta_for_penalty[
                    :, self.num_arm_dofs : self.num_hand_arm_dofs
                ]
            ),
            dim=-1,
        )

        object_palm_rel_speed = torch.norm(
            self.object_linvel - self._palm_state[:, 7:10], dim=-1
        )
        lifted_object_float = lifted_object.float()
        object_palm_dist = torch.norm(
            self.object_pos - self.palm_center_pos, dim=-1
        )
        true_grasp_metrics = self._dynamic_true_grasp_metrics(
            object_palm_dist, lifted_object
        )
        grasp_quality = true_grasp_metrics["grasp_quality"]
        true_grasp = true_grasp_metrics["true_grasp"]
        dynamic_true_grasp_quality_rew = grasp_quality * (~lifted_object).float()
        dynamic_lifted_true_grasp_rew = true_grasp.float() * lifted_object_float
        dynamic_quality_lift_progress_rew = (
            grasp_quality
            * lift_progress
            * dynamic_enclosure_base_rew
            * (~lifted_object).float()
        )
        dynamic_opposing_contact_rew = (
            true_grasp_metrics["opposing_score"] * (~lifted_object).float()
        )
        dynamic_scoop_lift_penalty = (
            true_grasp_metrics["scoop_lift"].float() * (1.0 - grasp_quality)
        )
        dynamic_palm_only_lift_penalty = true_grasp_metrics[
            "palm_only_lift"
        ].float()

        if self.dynamic_grasp_gate_lift_reward_by_grasp_quality:
            min_multiplier = float(
                self.dynamic_grasp_lift_reward_min_grasp_quality_multiplier
            )
            lift_reward_gate = min_multiplier + (1.0 - min_multiplier) * grasp_quality
            lifting_rew *= lift_reward_gate
            lift_bonus_rew *= lift_reward_gate
            dynamic_lift_progress_rew *= lift_reward_gate

        stable_grasp_base = (
            lifted_object
            & (mean_fingertip_dist < self.dynamic_grasp_stable_fingertip_distance)
            & (object_palm_rel_speed < self.dynamic_grasp_stable_object_palm_vel)
        )
        if self.dynamic_grasp_require_true_grasp_for_success:
            stable_grasp = stable_grasp_base & true_grasp
        else:
            stable_grasp = stable_grasp_base
        if self.dynamic_grasp_stable_counter_decay > 0:
            stable_grasp_int = stable_grasp.to(
                dtype=self.dynamic_grasp_success_steps.dtype
            )
            counter_decay = torch.full_like(
                self.dynamic_grasp_success_steps,
                self.dynamic_grasp_stable_counter_decay,
            )
            counter_delta = torch.where(
                stable_grasp,
                stable_grasp_int,
                -counter_decay,
            )
            self.dynamic_grasp_success_steps = torch.clamp(
                self.dynamic_grasp_success_steps + counter_delta,
                min=0,
                max=self.dynamic_grasp_success_steps_required,
            )
        else:
            stable_grasp_int = stable_grasp.to(
                dtype=self.dynamic_grasp_success_steps.dtype
            )
            self.dynamic_grasp_success_steps = (
                self.dynamic_grasp_success_steps + stable_grasp_int
            ) * stable_grasp_int

        stable_grasp_progress = torch.clamp(
            self.dynamic_grasp_success_steps.float()
            / max(float(self.dynamic_grasp_success_steps_required), 1.0),
            0.0,
            1.0,
        )
        dynamic_lifted_rel_vel_rew = (
            torch.exp(
                -object_palm_rel_speed
                / max(float(self.dynamic_grasp_lifted_rel_vel_scale), 1e-6)
            )
            * lifted_object_float
        )
        dynamic_lifted_centering_rew = (
            torch.exp(
                -object_palm_dist
                / max(
                    float(self.dynamic_grasp_lifted_centering_distance_scale),
                    1e-6,
                )
            )
            * lifted_object_float
        )
        dynamic_lifted_height_hold_rew = lift_progress * lifted_object_float
        dynamic_lifted_enclosure_vel_rew = (
            dynamic_enclosure_base_rew * dynamic_lifted_rel_vel_rew
        )
        contact_like = (
            (closest_fingertip_dist < self.dynamic_grasp_stable_fingertip_distance)
            | lifted_object
        )
        pregrasp_ready = (
            palm_pregrasp_xy_dist < self.dynamic_grasp_pregrasp_ready_distance
        )
        palm_xy_speed = torch.norm(palm_xy_vel, dim=-1)
        dynamic_pregrasp_hold_rew = (
            torch.exp(
                -palm_pregrasp_xy_dist
                / max(float(self.dynamic_grasp_pregrasp_distance_scale), 1e-6)
            )
            * torch.exp(
                -palm_xy_speed
                / max(float(self.dynamic_grasp_pregrasp_hold_vel_scale), 1e-6)
            )
            * (~contact_like).float()
            * (~lifted_object).float()
        )
        if self.dynamic_grasp_early_contact_grace_distance > 0.0:
            early_contact_not_ready_weight = torch.clamp(
                (
                    palm_pregrasp_xy_dist
                    - self.dynamic_grasp_pregrasp_ready_distance
                )
                / self.dynamic_grasp_early_contact_grace_distance,
                min=0.0,
                max=1.0,
            )
        else:
            early_contact_not_ready_weight = (~pregrasp_ready).float()
        dynamic_early_contact_penalty = (
            contact_like.float()
            * early_contact_not_ready_weight
            * (~lifted_object).float()
        )
        pre_contact_rel_vel_mask = (
            (object_palm_dist < self.dynamic_grasp_pre_contact_slow_distance)
            & (~contact_like)
            & (~lifted_object)
        )
        dynamic_pre_contact_rel_vel_penalty = (
            torch.square(
                torch.clamp(
                    object_palm_rel_speed
                    - self.dynamic_grasp_pre_contact_rel_vel_margin,
                    min=0.0,
                )
            )
            * pre_contact_rel_vel_mask.float()
        )
        contact_onset = contact_like & (~self.prev_dynamic_grasp_contact_like)
        pre_contact_slow_mask = (
            (object_palm_dist < self.dynamic_grasp_pre_contact_slow_distance)
            & (~contact_like)
            & (~lifted_object)
        )
        dynamic_pre_contact_slow_rew = (
            torch.exp(
                -object_palm_rel_speed
                / max(float(self.dynamic_grasp_pre_contact_slow_vel_scale), 1e-6)
            )
            * pre_contact_slow_mask.float()
        )
        dynamic_controlled_contact_rew = (
            contact_like.float()
            * (~lifted_object).float()
            * dynamic_enclosure_base_rew
            * torch.exp(
                -object_palm_rel_speed
                / max(
                    float(self.dynamic_grasp_controlled_contact_vel_scale), 1e-6
                )
            )
        )
        pregrasp_ready_not_lifted = pregrasp_ready.float() * (~lifted_object).float()
        dynamic_ready_enclosure_rew = (
            dynamic_enclosure_base_rew * pregrasp_ready_not_lifted
        )
        dynamic_ready_grasp_quality_rew = (
            grasp_quality * pregrasp_ready_not_lifted
        )
        dynamic_ready_contact_rew = (
            contact_like.float()
            * pregrasp_ready_not_lifted
            * dynamic_enclosure_base_rew
            * torch.exp(
                -object_palm_rel_speed
                / max(float(self.dynamic_grasp_ready_contact_vel_scale), 1e-6)
            )
        )
        dynamic_impact_penalty = (
            torch.square(
                torch.clamp(
                    object_palm_rel_speed - self.dynamic_grasp_impact_rel_vel_margin,
                    min=0.0,
                )
            )
            * contact_onset.float()
            * (~lifted_object).float()
        )
        dynamic_object_palm_rel_vel_penalty = (
            torch.clamp(
                object_palm_rel_speed
                - self.dynamic_grasp_post_contact_rel_vel_margin,
                min=0.0,
            )
            * contact_like.float()
        )

        commanded_xy_speed = torch.norm(
            self.dynamic_grasp_object_xy_velocity, dim=-1
        )
        current_object_xy_speed = torch.norm(object_xy_vel, dim=-1)
        dynamic_kick_penalty = (
            torch.square(
                torch.clamp(
                    current_object_xy_speed
                    - commanded_xy_speed
                    - self.dynamic_grasp_kick_speed_margin,
                    min=0.0,
                )
            )
            * contact_like.float()
            * (~lifted_object).float()
            * (~stable_grasp).float()
        )
        excess_xy_speed = torch.clamp(
            current_object_xy_speed
            - commanded_xy_speed
            - self.dynamic_grasp_push_away_speed_margin,
            min=0.0,
        )
        excess_uncontrolled_height = (
            torch.clamp(
                self.object_pos[:, 2]
                - self.object_init_state[:, 2]
                - self.dynamic_grasp_push_away_height_margin,
                min=0.0,
            )
            * (object_palm_rel_speed > self.dynamic_grasp_post_contact_rel_vel_margin)
        )
        dynamic_push_away_penalty = (
            (excess_xy_speed + 2.0 * excess_uncontrolled_height)
            * contact_like.float()
            * (~stable_grasp).float()
        )
        dynamic_dropped_penalty = (
            (self.object_pos[:, 2] < self.object_init_state[:, 2])
            & lifted_object
            & (~stable_grasp)
        ).float()

        is_success = (
            self.dynamic_grasp_success_steps
            >= self.dynamic_grasp_success_steps_required
        )
        dynamic_timeout_penalty = (
            (self.progress_buf >= self.max_episode_length - 1) & (~is_success)
        ).float()
        self.successes += is_success
        self.reset_goal_buf[:] = 0

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode[
            "raw_dynamic_velocity_match_rew"
        ] += dynamic_velocity_match_rew
        self.rewards_episode["raw_dynamic_intercept_rew"] += dynamic_intercept_rew
        self.rewards_episode[
            "raw_dynamic_pregrasp_alignment_rew"
        ] += dynamic_pregrasp_alignment_rew
        self.rewards_episode[
            "raw_dynamic_affordance_target_rew"
        ] += dynamic_affordance_target_rew
        self.rewards_episode[
            "raw_dynamic_enclosure_rew"
        ] += dynamic_enclosure_base_rew
        self.rewards_episode[
            "raw_dynamic_lift_progress_rew"
        ] += dynamic_lift_progress_rew
        self.rewards_episode[
            "raw_dynamic_stable_hold_rew"
        ] += stable_grasp.float()
        self.rewards_episode[
            "raw_dynamic_stable_grasp_progress_rew"
        ] += stable_grasp_progress
        self.rewards_episode[
            "raw_dynamic_lifted_rel_vel_rew"
        ] += dynamic_lifted_rel_vel_rew
        self.rewards_episode[
            "raw_dynamic_lifted_centering_rew"
        ] += dynamic_lifted_centering_rew
        self.rewards_episode[
            "raw_dynamic_lifted_height_hold_rew"
        ] += dynamic_lifted_height_hold_rew
        self.rewards_episode[
            "raw_dynamic_lifted_enclosure_vel_rew"
        ] += dynamic_lifted_enclosure_vel_rew
        self.rewards_episode[
            "raw_dynamic_pre_contact_slow_rew"
        ] += dynamic_pre_contact_slow_rew
        self.rewards_episode[
            "raw_dynamic_controlled_contact_rew"
        ] += dynamic_controlled_contact_rew
        self.rewards_episode["raw_dynamic_impact_penalty"] += dynamic_impact_penalty
        self.rewards_episode["raw_dynamic_kick_penalty"] += dynamic_kick_penalty
        self.rewards_episode[
            "raw_dynamic_object_palm_rel_vel_penalty"
        ] += dynamic_object_palm_rel_vel_penalty
        self.rewards_episode[
            "raw_dynamic_push_away_penalty"
        ] += dynamic_push_away_penalty
        self.rewards_episode[
            "raw_dynamic_dropped_penalty"
        ] += dynamic_dropped_penalty
        self.rewards_episode[
            "raw_dynamic_timeout_penalty"
        ] += dynamic_timeout_penalty
        self.rewards_episode[
            "raw_dynamic_safe_action_delta_penalty"
        ] += dynamic_safe_action_delta_penalty
        self.rewards_episode[
            "raw_dynamic_safe_arm_target_delta_penalty"
        ] += dynamic_safe_arm_target_delta_penalty
        self.rewards_episode[
            "raw_dynamic_safe_arm_target_accel_penalty"
        ] += dynamic_safe_arm_target_accel_penalty
        self.rewards_episode[
            "raw_dynamic_safe_hand_target_delta_penalty"
        ] += dynamic_safe_hand_target_delta_penalty
        self.rewards_episode[
            "raw_dynamic_pregrasp_hold_rew"
        ] += dynamic_pregrasp_hold_rew
        self.rewards_episode[
            "raw_dynamic_early_contact_penalty"
        ] += dynamic_early_contact_penalty
        self.rewards_episode[
            "raw_dynamic_pre_contact_rel_vel_penalty"
        ] += dynamic_pre_contact_rel_vel_penalty
        self.rewards_episode[
            "raw_dynamic_ready_enclosure_rew"
        ] += dynamic_ready_enclosure_rew
        self.rewards_episode[
            "raw_dynamic_ready_grasp_quality_rew"
        ] += dynamic_ready_grasp_quality_rew
        self.rewards_episode[
            "raw_dynamic_ready_contact_rew"
        ] += dynamic_ready_contact_rew
        self.rewards_episode[
            "raw_dynamic_true_grasp_quality_rew"
        ] += dynamic_true_grasp_quality_rew
        self.rewards_episode[
            "raw_dynamic_lifted_true_grasp_rew"
        ] += dynamic_lifted_true_grasp_rew
        self.rewards_episode[
            "raw_dynamic_quality_lift_progress_rew"
        ] += dynamic_quality_lift_progress_rew
        self.rewards_episode[
            "raw_dynamic_opposing_contact_rew"
        ] += dynamic_opposing_contact_rew
        self.rewards_episode[
            "raw_dynamic_scoop_lift_penalty"
        ] += dynamic_scoop_lift_penalty
        self.rewards_episode[
            "raw_dynamic_palm_only_lift_penalty"
        ] += dynamic_palm_only_lift_penalty

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        lifting_rew *= self.lifting_rew_scale
        dynamic_velocity_match_rew *= self.dynamic_grasp_velocity_match_rew_scale
        dynamic_intercept_rew *= self.dynamic_grasp_intercept_rew_scale
        dynamic_pregrasp_alignment_rew *= (
            self.dynamic_grasp_pregrasp_alignment_rew_scale
        )
        dynamic_affordance_target_rew *= (
            self.dynamic_grasp_affordance_target_rew_scale
        )
        dynamic_enclosure_rew *= self.dynamic_grasp_enclosure_rew_scale
        dynamic_lifted_enclosure_rew *= (
            self.dynamic_grasp_lifted_enclosure_rew_scale
        )
        dynamic_lift_progress_rew *= self.dynamic_grasp_lift_progress_rew_scale
        dynamic_stable_hold_rew = (
            stable_grasp.float() * self.dynamic_grasp_stable_hold_rew_scale
        )
        dynamic_stable_grasp_progress_rew = (
            stable_grasp_progress * self.dynamic_grasp_stable_progress_rew_scale
        )
        dynamic_lifted_rel_vel_rew *= self.dynamic_grasp_lifted_rel_vel_rew_scale
        dynamic_lifted_centering_rew *= self.dynamic_grasp_lifted_centering_rew_scale
        dynamic_lifted_height_hold_rew *= (
            self.dynamic_grasp_lifted_height_hold_rew_scale
        )
        dynamic_lifted_enclosure_vel_rew *= (
            self.dynamic_grasp_lifted_enclosure_vel_rew_scale
        )
        dynamic_pre_contact_slow_rew *= self.dynamic_grasp_pre_contact_slow_rew_scale
        dynamic_controlled_contact_rew *= (
            self.dynamic_grasp_controlled_contact_rew_scale
        )
        dynamic_impact_penalty = (
            -dynamic_impact_penalty * self.dynamic_grasp_impact_penalty_scale
        )
        dynamic_kick_penalty = (
            -dynamic_kick_penalty * self.dynamic_grasp_kick_penalty_scale
        )
        dynamic_object_palm_rel_vel_penalty = (
            -dynamic_object_palm_rel_vel_penalty
            * self.dynamic_grasp_post_contact_rel_vel_penalty_scale
        )
        dynamic_push_away_penalty = (
            -dynamic_push_away_penalty * self.dynamic_grasp_push_away_penalty_scale
        )
        dynamic_dropped_penalty = (
            -dynamic_dropped_penalty * self.dynamic_grasp_dropped_penalty_scale
        )
        dynamic_timeout_penalty = (
            -dynamic_timeout_penalty * self.dynamic_grasp_timeout_penalty_scale
        )
        dynamic_safe_action_delta_penalty = (
            -dynamic_safe_action_delta_penalty
            * self.dynamic_grasp_safe_action_delta_penalty_scale
        )
        dynamic_safe_arm_target_delta_penalty = (
            -dynamic_safe_arm_target_delta_penalty
            * self.dynamic_grasp_safe_arm_target_delta_penalty_scale
        )
        dynamic_safe_arm_target_accel_penalty = (
            -dynamic_safe_arm_target_accel_penalty
            * self.dynamic_grasp_safe_arm_target_accel_penalty_scale
        )
        dynamic_safe_hand_target_delta_penalty = (
            -dynamic_safe_hand_target_delta_penalty
            * self.dynamic_grasp_safe_hand_target_delta_penalty_scale
        )
        dynamic_pregrasp_hold_rew *= self.dynamic_grasp_pregrasp_hold_rew_scale
        dynamic_early_contact_penalty = (
            -dynamic_early_contact_penalty
            * self.dynamic_grasp_early_contact_penalty_scale
        )
        dynamic_pre_contact_rel_vel_penalty = (
            -dynamic_pre_contact_rel_vel_penalty
            * self.dynamic_grasp_pre_contact_rel_vel_penalty_scale
        )
        dynamic_ready_enclosure_rew *= self.dynamic_grasp_ready_enclosure_rew_scale
        dynamic_ready_grasp_quality_rew *= (
            self.dynamic_grasp_ready_grasp_quality_rew_scale
        )
        dynamic_ready_contact_rew *= self.dynamic_grasp_ready_contact_rew_scale
        dynamic_true_grasp_quality_rew *= (
            self.dynamic_grasp_true_grasp_quality_rew_scale
        )
        dynamic_lifted_true_grasp_rew *= (
            self.dynamic_grasp_lifted_true_grasp_rew_scale
        )
        dynamic_quality_lift_progress_rew *= (
            self.dynamic_grasp_quality_lift_progress_rew_scale
        )
        dynamic_opposing_contact_rew *= (
            self.dynamic_grasp_opposing_contact_rew_scale
        )
        dynamic_scoop_lift_penalty = (
            -dynamic_scoop_lift_penalty * self.dynamic_grasp_scoop_lift_penalty_scale
        )
        dynamic_palm_only_lift_penalty = (
            -dynamic_palm_only_lift_penalty
            * self.dynamic_grasp_palm_only_lift_penalty_scale
        )
        dynamic_success_bonus = is_success.float() * self.dynamic_grasp_success_bonus

        reward = (
            fingertip_delta_rew
            + hand_delta_penalty
            + lifting_rew
            + lift_bonus_rew
            + dynamic_velocity_match_rew
            + dynamic_intercept_rew
            + dynamic_pregrasp_alignment_rew
            + dynamic_affordance_target_rew
            + dynamic_enclosure_rew
            + dynamic_lifted_enclosure_rew
            + dynamic_lift_progress_rew
            + dynamic_stable_hold_rew
            + dynamic_stable_grasp_progress_rew
            + dynamic_lifted_rel_vel_rew
            + dynamic_lifted_centering_rew
            + dynamic_lifted_height_hold_rew
            + dynamic_lifted_enclosure_vel_rew
            + dynamic_pre_contact_slow_rew
            + dynamic_controlled_contact_rew
            + dynamic_impact_penalty
            + dynamic_kick_penalty
            + dynamic_object_palm_rel_vel_penalty
            + dynamic_push_away_penalty
            + dynamic_dropped_penalty
            + dynamic_timeout_penalty
            + dynamic_safe_action_delta_penalty
            + dynamic_safe_arm_target_delta_penalty
            + dynamic_safe_arm_target_accel_penalty
            + dynamic_safe_hand_target_delta_penalty
            + dynamic_pregrasp_hold_rew
            + dynamic_early_contact_penalty
            + dynamic_pre_contact_rel_vel_penalty
            + dynamic_ready_enclosure_rew
            + dynamic_ready_grasp_quality_rew
            + dynamic_ready_contact_rew
            + dynamic_true_grasp_quality_rew
            + dynamic_lifted_true_grasp_rew
            + dynamic_quality_lift_progress_rew
            + dynamic_opposing_contact_rew
            + dynamic_scoop_lift_penalty
            + dynamic_palm_only_lift_penalty
            + dynamic_success_bonus
            + kuka_actions_penalty
            + hand_actions_penalty
        )
        self.rew_buf[:] = reward

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        info_successes = self._successes_for_info()
        self.extras["successes"] = info_successes
        self.extras["success_ratio"] = (
            info_successes.mean().item()
            / max(self.max_consecutive_successes, 1)
        )
        self.extras["dynamic_grasp_success_steps"] = (
            self.dynamic_grasp_success_steps.float().mean().item()
        )
        self.extras["dynamic_object_xy_speed"] = (
            torch.norm(self.object_linvel[:, 0:2], dim=-1).mean().item()
        )
        self.extras["dynamic_commanded_xy_speed"] = (
            torch.norm(self.dynamic_grasp_object_xy_velocity, dim=-1).mean().item()
        )
        self.extras["dynamic_object_palm_rel_speed"] = (
            object_palm_rel_speed.mean().item()
        )
        self.extras["dynamic_object_palm_dist"] = object_palm_dist.mean().item()
        self.extras["dynamic_stable_grasp_progress"] = (
            stable_grasp_progress.mean().item()
        )
        self.extras["dynamic_contact_fraction"] = contact_like.float().mean().item()
        self.extras["dynamic_lifted_fraction"] = lifted_object.float().mean().item()
        self.extras["dynamic_stable_grasp_fraction"] = (
            stable_grasp.float().mean().item()
        )
        self.extras["dynamic_stable_grasp_base_fraction"] = (
            stable_grasp_base.float().mean().item()
        )
        self.extras["dynamic_true_grasp_fraction"] = (
            true_grasp.float().mean().item()
        )
        self.extras["dynamic_grasp_quality"] = grasp_quality.mean().item()
        self.extras["dynamic_finger_contact_count"] = (
            true_grasp_metrics["finger_contact_count"].mean().item()
        )
        self.extras["dynamic_non_thumb_contact_count"] = (
            true_grasp_metrics["non_thumb_contact_count"].mean().item()
        )
        self.extras["dynamic_thumb_contact_fraction"] = (
            true_grasp_metrics["thumb_contact"].float().mean().item()
        )
        self.extras["dynamic_opposing_contact_fraction"] = (
            true_grasp_metrics["opposing_contact"].float().mean().item()
        )
        self.extras["dynamic_opposing_score"] = (
            true_grasp_metrics["opposing_score"].mean().item()
        )
        self.extras["dynamic_scoop_lift_fraction"] = (
            true_grasp_metrics["scoop_lift"].float().mean().item()
        )
        self.extras["dynamic_palm_only_lift_fraction"] = (
            true_grasp_metrics["palm_only_lift"].float().mean().item()
        )
        self.extras["dynamic_dropped_fraction"] = (
            (dynamic_dropped_penalty < 0.0).float().mean().item()
            if self.dynamic_grasp_dropped_penalty_scale > 0.0
            else torch.zeros_like(dynamic_dropped_penalty).mean().item()
        )
        self.extras["dynamic_timeout_fraction"] = (
            (dynamic_timeout_penalty < 0.0).float().mean().item()
            if self.dynamic_grasp_timeout_penalty_scale > 0.0
            else torch.zeros_like(dynamic_timeout_penalty).mean().item()
        )
        self.extras["dynamic_closest_fingertip_dist"] = (
            closest_fingertip_dist.mean().item()
        )
        self.extras["dynamic_mean_fingertip_dist"] = mean_fingertip_dist.mean().item()
        self.extras["dynamic_lift_progress"] = lift_progress.mean().item()
        contact_count = contact_like.float().sum().clamp(min=1.0)
        self.extras["dynamic_object_palm_rel_speed_contact"] = (
            (object_palm_rel_speed * contact_like.float()).sum() / contact_count
        ).item()
        self.extras["dynamic_post_contact_rel_vel_penalty"] = (
            dynamic_object_palm_rel_vel_penalty.mean().item()
        )
        self.extras["dynamic_push_away_penalty"] = (
            dynamic_push_away_penalty.mean().item()
        )
        self.extras["dynamic_pregrasp_xy_dist"] = (
            palm_pregrasp_xy_dist.mean().item()
        )
        self.extras["dynamic_affordance_enabled"] = float(
            self.dynamic_grasp_use_affordance_prior
        )
        self.extras["dynamic_affordance_confidence"] = (
            affordance_targets["confidence"].mean().item()
        )
        self.extras["dynamic_affordance_target_xy_dist"] = (
            palm_intercept_xy_dist.mean().item()
        )
        self.extras["dynamic_affordance_target_rew"] = (
            dynamic_affordance_target_rew.mean().item()
        )
        self.extras["dynamic_pregrasp_ready_fraction"] = (
            pregrasp_ready.float().mean().item()
        )
        self.extras["dynamic_palm_xy_speed"] = palm_xy_speed.mean().item()
        lifted_count = lifted_object_float.sum().clamp(min=1.0)
        self.extras["dynamic_lifted_object_palm_rel_speed"] = (
            (object_palm_rel_speed * lifted_object_float).sum() / lifted_count
        ).item()
        self.extras["dynamic_lifted_object_palm_dist"] = (
            (object_palm_dist * lifted_object_float).sum() / lifted_count
        ).item()
        self.extras["dynamic_contact_onset_fraction"] = (
            contact_onset.float().mean().item()
        )
        self.extras["dynamic_pre_contact_slow_fraction"] = (
            pre_contact_slow_mask.float().mean().item()
        )
        self.extras["dynamic_pre_contact_slow_rew"] = (
            dynamic_pre_contact_slow_rew.mean().item()
        )
        self.extras["dynamic_controlled_contact_rew"] = (
            dynamic_controlled_contact_rew.mean().item()
        )
        self.extras["dynamic_impact_penalty"] = (
            dynamic_impact_penalty.mean().item()
        )
        self.extras["dynamic_kick_penalty"] = dynamic_kick_penalty.mean().item()
        self.extras["dynamic_safe_action_delta_penalty"] = (
            dynamic_safe_action_delta_penalty.mean().item()
        )
        self.extras["dynamic_safe_arm_target_delta_penalty"] = (
            dynamic_safe_arm_target_delta_penalty.mean().item()
        )
        self.extras["dynamic_safe_arm_target_accel_penalty"] = (
            dynamic_safe_arm_target_accel_penalty.mean().item()
        )
        self.extras["dynamic_safe_hand_target_delta_penalty"] = (
            dynamic_safe_hand_target_delta_penalty.mean().item()
        )
        self.extras["dynamic_pregrasp_hold_rew"] = (
            dynamic_pregrasp_hold_rew.mean().item()
        )
        self.extras["dynamic_early_contact_penalty"] = (
            dynamic_early_contact_penalty.mean().item()
        )
        self.extras["dynamic_early_contact_not_ready_weight"] = (
            early_contact_not_ready_weight.mean().item()
        )
        self.extras["dynamic_pre_contact_rel_vel_penalty"] = (
            dynamic_pre_contact_rel_vel_penalty.mean().item()
        )
        self.extras["dynamic_ready_enclosure_rew"] = (
            dynamic_ready_enclosure_rew.mean().item()
        )
        self.extras["dynamic_ready_grasp_quality_rew"] = (
            dynamic_ready_grasp_quality_rew.mean().item()
        )
        self.extras["dynamic_ready_contact_rew"] = (
            dynamic_ready_contact_rew.mean().item()
        )
        self.extras["dynamic_true_grasp_quality_rew"] = (
            dynamic_true_grasp_quality_rew.mean().item()
        )
        self.extras["dynamic_lifted_true_grasp_rew"] = (
            dynamic_lifted_true_grasp_rew.mean().item()
        )
        self.extras["dynamic_quality_lift_progress_rew"] = (
            dynamic_quality_lift_progress_rew.mean().item()
        )
        self.extras["dynamic_opposing_contact_rew"] = (
            dynamic_opposing_contact_rew.mean().item()
        )
        self.extras["dynamic_scoop_lift_penalty"] = (
            dynamic_scoop_lift_penalty.mean().item()
        )
        self.extras["dynamic_palm_only_lift_penalty"] = (
            dynamic_palm_only_lift_penalty.mean().item()
        )
        self.extras[
            "dynamic_grasp_speed_curriculum_alpha"
        ] = self._dynamic_grasp_speed_curriculum_alpha()
        self._log_dynamic_object_pool_metrics(
            is_success=is_success,
            lifted_object=lifted_object,
            stable_grasp=stable_grasp,
            lift_progress=lift_progress,
            object_palm_rel_speed=object_palm_rel_speed,
            true_grasp=true_grasp,
            grasp_quality=grasp_quality,
            scoop_lift=true_grasp_metrics["scoop_lift"],
            palm_only_lift=true_grasp_metrics["palm_only_lift"],
        )
        self.prev_dynamic_grasp_contact_like[:] = contact_like
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (dynamic_velocity_match_rew, "dynamic_velocity_match_rew"),
            (dynamic_intercept_rew, "dynamic_intercept_rew"),
            (
                dynamic_pregrasp_alignment_rew,
                "dynamic_pregrasp_alignment_rew",
            ),
            (
                dynamic_affordance_target_rew,
                "dynamic_affordance_target_rew",
            ),
            (dynamic_enclosure_rew, "dynamic_enclosure_rew"),
            (dynamic_lifted_enclosure_rew, "dynamic_lifted_enclosure_rew"),
            (dynamic_lift_progress_rew, "dynamic_lift_progress_rew"),
            (dynamic_stable_hold_rew, "dynamic_stable_hold_rew"),
            (
                dynamic_stable_grasp_progress_rew,
                "dynamic_stable_grasp_progress_rew",
            ),
            (dynamic_lifted_rel_vel_rew, "dynamic_lifted_rel_vel_rew"),
            (dynamic_lifted_centering_rew, "dynamic_lifted_centering_rew"),
            (
                dynamic_lifted_height_hold_rew,
                "dynamic_lifted_height_hold_rew",
            ),
            (
                dynamic_lifted_enclosure_vel_rew,
                "dynamic_lifted_enclosure_vel_rew",
            ),
            (dynamic_pre_contact_slow_rew, "dynamic_pre_contact_slow_rew"),
            (dynamic_controlled_contact_rew, "dynamic_controlled_contact_rew"),
            (dynamic_impact_penalty, "dynamic_impact_penalty"),
            (dynamic_kick_penalty, "dynamic_kick_penalty"),
            (
                dynamic_object_palm_rel_vel_penalty,
                "dynamic_object_palm_rel_vel_penalty",
            ),
            (dynamic_push_away_penalty, "dynamic_push_away_penalty"),
            (dynamic_dropped_penalty, "dynamic_dropped_penalty"),
            (dynamic_timeout_penalty, "dynamic_timeout_penalty"),
            (
                dynamic_safe_action_delta_penalty,
                "dynamic_safe_action_delta_penalty",
            ),
            (
                dynamic_safe_arm_target_delta_penalty,
                "dynamic_safe_arm_target_delta_penalty",
            ),
            (
                dynamic_safe_arm_target_accel_penalty,
                "dynamic_safe_arm_target_accel_penalty",
            ),
            (
                dynamic_safe_hand_target_delta_penalty,
                "dynamic_safe_hand_target_delta_penalty",
            ),
            (dynamic_pregrasp_hold_rew, "dynamic_pregrasp_hold_rew"),
            (dynamic_early_contact_penalty, "dynamic_early_contact_penalty"),
            (
                dynamic_pre_contact_rel_vel_penalty,
                "dynamic_pre_contact_rel_vel_penalty",
            ),
            (dynamic_ready_enclosure_rew, "dynamic_ready_enclosure_rew"),
            (
                dynamic_ready_grasp_quality_rew,
                "dynamic_ready_grasp_quality_rew",
            ),
            (dynamic_ready_contact_rew, "dynamic_ready_contact_rew"),
            (dynamic_true_grasp_quality_rew, "dynamic_true_grasp_quality_rew"),
            (dynamic_lifted_true_grasp_rew, "dynamic_lifted_true_grasp_rew"),
            (
                dynamic_quality_lift_progress_rew,
                "dynamic_quality_lift_progress_rew",
            ),
            (dynamic_opposing_contact_rew, "dynamic_opposing_contact_rew"),
            (dynamic_scoop_lift_penalty, "dynamic_scoop_lift_penalty"),
            (
                dynamic_palm_only_lift_penalty,
                "dynamic_palm_only_lift_penalty",
            ),
            (dynamic_success_bonus, "dynamic_success_bonus"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (hand_actions_penalty, "hand_actions_penalty"),
            (reward, "total_reward"),
        ]
        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

    def _successes_for_info(self) -> Tensor:
        done_mask = self.reset_buf.to(dtype=torch.bool)
        return torch.where(done_mask, self.successes, self.prev_episode_successes)

    def _compute_resets(self, is_success):
        ones = torch.ones_like(self.reset_buf)
        zeros = torch.zeros_like(self.reset_buf)

        object_z_low = torch.where(self.object_pos[:, 2] < 0.1, ones, zeros)  # fall
        if self.max_consecutive_successes > 0:
            # Reset progress buffer if max_consecutive_successes > 0
            self.progress_buf = torch.where(
                is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf
            )
            max_consecutive_successes_reached = torch.where(
                self.successes >= self.max_consecutive_successes, ones, zeros
            )
        else:
            max_consecutive_successes_reached = zeros

        max_episode_length_reached = torch.where(
            self.progress_buf >= self.max_episode_length - 1, ones, zeros
        )

        if self.with_table_force_sensor:
            TABLE_FORCE_THRESHOLD = 100.0
            table_force_too_high = torch.where(
                self.max_table_sensor_force_norm_smoothed > TABLE_FORCE_THRESHOLD,
                ones,
                zeros,
            )
        else:
            table_force_too_high = zeros

        # hand far from the object
        hand_far_from_object = torch.where(
            self.curr_fingertip_distances.max(dim=-1).values > 1.5, ones, zeros
        )

        # Reset when dropped
        # Dropped means the object was lifted and then dropped back down to the table
        if self.cfg["env"]["resetWhenDropped"]:
            # As of right now:
            # - table center is at 0.38m and is 0.3m tall, so its surface is at 0.38m + 0.15m = 0.53m
            # - object init state is at 0.63m (table height 0.38m + object relative to table height 0.25m) with +/-0.02m variation
            # - The goal states are sampled in 0.8m + [-0.12m, 0.25m] = [0.68m, 1.05m]
            # - Right now, liftedBonusThreshold is 0.15m, so the object is lifted if it is at least 0.15m - 0.05m above init state, which is 0.73m
            # - We don't want it to flicker between first lifted then suddenly dropped if it slightly goes down, so we need hysteresis here
            # - Thus, we choose a threshold of object init state at 0.63m, so there is 0.1m gap below lifted threshold and 0.05m gap below goal states
            # - And there is 0.1 gap above the table surface, so dropping the object on the table will be detected as dropped
            dropped_z = self.object_init_state[:, 2]
            dropped = (
                torch.where(self.object_pos[:, 2] < dropped_z, ones, zeros)
                * self.lifted_object
            )
        else:
            dropped = zeros

        resets = (
            self.reset_buf
            | object_z_low
            | max_consecutive_successes_reached
            | max_episode_length_reached
            | table_force_too_high
            | hand_far_from_object
            | dropped
        )
        resets = self._extra_reset_rules(resets)

        # Print resets when there is only one environment
        PRINT_RESET_REASONS = False
        if self.num_envs == 1 and resets.item() and PRINT_RESET_REASONS:
            print("=" * 100)
            print("REASON FOR RESET:")
            print(f"object_z_low: {object_z_low.item()}")
            print(
                f"max_consecutive_successes_reached: {max_consecutive_successes_reached.item()}"
            )
            print(f"max_episode_length_reached: {max_episode_length_reached.item()}")
            print(f"table_force_too_high: {table_force_too_high.item()}")
            print(f"hand_far_from_object: {hand_far_from_object.item()}")
            print(f"dropped: {dropped.item()}")
            print(f"resets: {resets.item()}")
            print("=" * 100)
            print(f"self.successes: {self.successes.item()}")
            print()

        # Keep track of reasons for reset
        from collections import Counter, deque

        if not hasattr(self, "recent_reset_reason_history"):
            MAX_HISTORY_LENGTH = 4096
            self.recent_reset_reason_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.cumulative_reset_reason_counts = {
                "object_z_low": 0,
                "max_consecutive_successes_reached": 0,
                "max_episode_length_reached": 0,
                "table_force_too_high": 0,
                "hand_far_from_object": 0,
                "dropped": 0,
            }
        # Current means this recent step (across all environments)
        # Recent means the last MAX_HISTORY_LENGTH resets
        # Cumulative means across all time
        # Use a deque to keep track of recent (FIFO) and use Counter on it to get the counts
        # Use a dict to keep track of cumulative counts so that memory is not a problem indefinitely

        # Get current counts
        current_reset_reason_counts = {
            "object_z_low": object_z_low.sum().item(),
            "max_consecutive_successes_reached": max_consecutive_successes_reached.sum().item(),
            "max_episode_length_reached": max_episode_length_reached.sum().item(),
            "table_force_too_high": table_force_too_high.sum().item(),
            "hand_far_from_object": hand_far_from_object.sum().item(),
            "dropped": dropped.sum().item(),
        }

        # Update counts
        for reason, count in current_reset_reason_counts.items():
            self.cumulative_reset_reason_counts[reason] += count
        for reason, count in current_reset_reason_counts.items():
            self.recent_reset_reason_history.extend([reason] * count)
        recent_reset_reason_counts = Counter(self.recent_reset_reason_history)

        # We log the recent counts (fractions of all recent resets)
        # We don't want cumulative because it is affected too much by the past (we want to see how well the policy is doing now)
        recent_total = sum(recent_reset_reason_counts.values())
        for reason, count in recent_reset_reason_counts.items():
            self.extras[f"reset/{reason}"] = (
                count / recent_total if recent_total > 0 else 0
            )

        PRINT = False
        if PRINT:
            current_total = sum(current_reset_reason_counts.values())
            if current_total > 0:
                print(f"{current_total} resets in the last step!")
                print()

                cumulative_total = sum(self.cumulative_reset_reason_counts.values())
                print("Across all time:")
                print("-" * 100)
                for reason, count in self.cumulative_reset_reason_counts.items():
                    if count > 0:
                        print(
                            f"Reset reason: {reason} {count}/{cumulative_total} ({count / cumulative_total:.1%})"
                        )
                print()

                recent_total = sum(recent_reset_reason_counts.values())
                print("Recent:")
                print("-" * 100)
                for reason, count in recent_reset_reason_counts.items():
                    if count > 0:
                        print(
                            f"Reset reason: {reason} {count}/{recent_total} ({count / recent_total:.1%})"
                        )
                print()
                print()

        return resets

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance,
            self.initial_tolerance,
            self.target_tolerance,
            self.successes,
        )
        return true_objective

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        if self.dynamic_tabletop_grasp:
            return self._compute_dynamic_tabletop_grasp_reward()

        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(
            lifted_object
        )
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            keypoint_rew = keypoint_rew_fixed_size

        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        # noinspection PyTypeChecker
        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
        near_goal_fixed_size: Tensor = (
            self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance
        )
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            near_goal = near_goal_fixed_size

        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            # If near_goal is True (1): (steps + 1) * 1 = Increment
            # If near_goal is False (0): (steps + 0) * 0 = Reset to 0
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = is_success
        self.successes += is_success

        # Print successes when there is only one environment
        # if self.num_envs == 1 and is_success.item():
        #     print("~" * 100)
        #     print("IS SUCCESS:")
        #     print(f"self.successes: {self.successes.item()}")
        #     print("~" * 100)

        self.reset_goal_buf[:] = goal_resets

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew
        self.rewards_episode["raw_object_lin_vel_penalty"] += object_lin_vel_penalty
        self.rewards_episode["raw_object_ang_vel_penalty"] += object_ang_vel_penalty

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        object_lin_vel_penalty *= self.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        # We spread out the reward over "success_steps"
        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_rew = is_success * self.reach_goal_bonus

        reward = (
            fingertip_delta_rew
            + hand_delta_penalty  # + sign here because hand_delta_penalty is negative
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )

        self.rew_buf[:] = reward

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        # HACK: Force no reset for isaac_env_ros type testing
        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        info_successes = self._successes_for_info()
        self.extras["successes"] = info_successes
        self.extras["success_ratio"] = (
            info_successes.mean().item() / self.max_consecutive_successes
        )
        self.extras["closest_keypoint_max_dist"] = (
            self.prev_episode_closest_keypoint_max_dist
        )
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        # scalars for logging
        # self.extras["true_objective_mean"] = self.true_objective.mean()
        # self.extras["true_objective_min"] = self.true_objective.min()
        # self.extras["true_objective_max"] = self.true_objective.max()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (hand_actions_penalty, "hand_actions_penalty"),
            (bonus_rew, "bonus_rew"),
            (object_lin_vel_penalty, "object_lin_vel_penalty"),
            (object_ang_vel_penalty, "object_ang_vel_penalty"),
            (reward, "total_reward"),
        ]

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

    def _eval_stats(self, is_success: Tensor) -> None:
        if self.eval_stats:
            frame: int = self.frame_since_restart
            n_frames = torch.empty_like(self.last_success_step).fill_(frame)
            self.success_time = torch.where(
                is_success, n_frames - self.last_success_step, self.success_time
            )
            self.last_success_step = torch.where(
                is_success, n_frames, self.last_success_step
            )
            mask_ = self.success_time > 0
            if any(mask_):
                avg_time_mean = (
                    (self.success_time * mask_).sum(dim=0) / mask_.sum(dim=0)
                ).item()
            else:
                avg_time_mean = math.nan

            self.total_resets = self.total_resets + self.reset_buf.sum()
            self.total_successes = (
                self.total_successes + (self.successes * self.reset_buf).sum()
            )
            self.total_num_resets += self.reset_buf

            reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                last_successes = self.successes[reset_ids].long()
                self.successes_count[last_successes] += 1
                self._record_eval_object_first_episodes(reset_ids, last_successes)

            if frame % 100 == 0:
                # The direct average shows the overall result more quickly, but slightly undershoots long term
                # policy performance.
                total_resets = self.total_resets.clamp(min=1.0)
                total_success_rate = self.total_successes / total_resets
                print(f"Max num successes: {self.successes.max().item()}")
                print(f"Total success rate: {total_success_rate.item():.2f}")
                print(
                    f"Average consecutive successes: {self.prev_episode_successes.mean().item():.2f}"
                )
                print(
                    f"Total num resets: {self.total_num_resets.sum().item()} --> {self.total_num_resets}"
                )
                print(
                    f"Reset percentage: {(self.total_num_resets > 0).sum() / self.num_envs:.2%}"
                )
                print(
                    f"Last ep successes: {self.prev_episode_successes.mean().item():.2f}"
                )
                print(
                    f"Last ep true objective: {self.prev_episode_true_objective.mean().item():.2f}"
                )

                self.eval_summaries.add_scalar(
                    "last_ep_successes",
                    self.prev_episode_successes.mean().item(),
                    frame,
                )
                self.eval_summaries.add_scalar(
                    "total_success_rate",
                    total_success_rate.item(),
                    frame,
                )
                self.eval_summaries.add_scalar(
                    "last_ep_true_objective",
                    self.prev_episode_true_objective.mean().item(),
                    frame,
                )
                self.eval_summaries.add_scalar(
                    "reset_stats/reset_percentage",
                    (self.total_num_resets > 0).sum() / self.num_envs,
                    frame,
                )
                self.eval_summaries.add_scalar(
                    "reset_stats/min_num_resets",
                    self.total_num_resets.min().item(),
                    frame,
                )

                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_time_frames", avg_time_mean, frame
                )
                frame_time = self.control_freq_inv * self.dt
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_time_seconds",
                    avg_time_mean * frame_time,
                    frame,
                )
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_per_minute",
                    60.0 / (avg_time_mean * frame_time),
                    frame,
                )
                print(
                    f"Policy speed (successes per minute): {60.0 / (avg_time_mean * frame_time):.2f}"
                )

                # create a matplotlib bar chart of the self.successes_count
                import matplotlib.pyplot as plt

                plt.bar(
                    list(range(self.max_consecutive_successes + 1)),
                    self.successes_count.cpu().numpy(),
                )
                plt.title("Successes histogram")
                plt.xlabel("Successes")
                plt.ylabel("Frequency")
                plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
                plt.clf()

            if (
                getattr(self, "eval_stop_after_each_env_once", False)
                and self.eval_first_episode_recorded.all().item()
            ):
                self._write_eval_object_stats(frame=frame, final=True)
                raise SystemExit(0)

    def _record_eval_object_first_episodes(
        self, reset_ids: Tensor, last_successes: Tensor
    ) -> None:
        if not (
            hasattr(self, "eval_object_episode_counts")
            and hasattr(self, "eval_object_success_counts")
        ):
            return

        new_mask = ~self.eval_first_episode_recorded[reset_ids]
        if not new_mask.any().item():
            return

        new_reset_ids = reset_ids[new_mask]
        new_successes = (last_successes[new_mask] > 0).float()
        object_asset_indices = self.object_asset_indices[new_reset_ids]
        self.eval_first_episode_recorded[new_reset_ids] = True
        self.eval_object_episode_counts.scatter_add_(
            0,
            object_asset_indices,
            torch.ones_like(object_asset_indices, dtype=torch.long),
        )
        self.eval_object_success_counts.scatter_add_(
            0, object_asset_indices, new_successes
        )

    def _eval_object_stats_dict(self) -> dict:
        if not (
            hasattr(self, "eval_object_episode_counts")
            and hasattr(self, "eval_object_success_counts")
        ):
            return {}

        counts = self.eval_object_episode_counts.detach().cpu().numpy()
        successes = self.eval_object_success_counts.detach().cpu().numpy()
        per_object = {}
        for idx, label in enumerate(self.object_asset_labels):
            count = int(counts[idx])
            success = float(successes[idx])
            per_object[str(label)] = {
                "episodes": count,
                "successes": success,
                "success_rate": success / count if count > 0 else float("nan"),
            }

        total_episodes = int(counts.sum())
        total_successes = float(successes.sum())
        return {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_success_rate": (
                total_successes / total_episodes if total_episodes > 0 else float("nan")
            ),
            "per_object": per_object,
        }

    def _write_eval_object_stats(self, frame: int, final: bool = False) -> None:
        stats = self._eval_object_stats_dict()
        if not stats:
            return

        stats["frame"] = int(frame)
        stats["completed_envs"] = int(self.eval_first_episode_recorded.sum().item())
        stats["num_envs"] = int(self.num_envs)
        stats["final"] = bool(final)
        os.makedirs(self.eval_summary_dir, exist_ok=True)
        stats_path = os.path.join(self.eval_summary_dir, "per_object_eval_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print("=" * 80)
        print("Per-object deterministic eval summary")
        print(
            f"Total: {stats['total_successes']:.0f}/{stats['total_episodes']} "
            f"({stats['total_success_rate']:.3f})"
        )
        for label, object_stats in stats["per_object"].items():
            print(
                f"  {label}: {object_stats['successes']:.0f}/"
                f"{object_stats['episodes']} "
                f"({object_stats['success_rate']:.3f})"
            )
        print(f"Wrote {stats_path}")
        print("=" * 80)

    def populate_sim_buffers(self) -> Tuple[Tensor, int]:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.with_fingertip_force_sensors or self.with_table_force_sensor:
            self.gym.refresh_force_sensor_tensor(self.sim)
        if self.with_dof_force_sensors:
            self.gym.refresh_dof_force_tensor(self.sim)

        if self.with_table_force_sensor:
            self.table_sensor_forces_raw = self.force_sensor_tensor[
                :, self.table_sensor_idx, :
            ]

            # Smooth the force because the signal can be spikey, and we don't want to make decisions based on spikey signals
            TABLE_SENSOR_FORCE_SMOOTHING_ALPHA = 0.1  # 1 = no smoothing, 0 = no updates
            self.table_sensor_forces_smoothed = self.interpolate(
                init=self.table_sensor_forces_smoothed,
                final=self.table_sensor_forces_raw,
                alpha=TABLE_SENSOR_FORCE_SMOOTHING_ALPHA,
            )
            table_sensor_force_norm_smoothed = self.table_sensor_forces_smoothed[
                :, :3
            ].norm(dim=-1)
            self.max_table_sensor_force_norm_smoothed = torch.where(
                table_sensor_force_norm_smoothed
                > self.max_table_sensor_force_norm_smoothed,
                table_sensor_force_norm_smoothed,
                self.max_table_sensor_force_norm_smoothed,
            )

        if self.with_fingertip_force_sensors:
            raise NotImplementedError(
                "Fingertip force sensors are not implemented yet, be careful about indexing"
            )
            self.finger_sensor_forces = self.force_sensor_tensor[
                :, self.finger_sensor_idxs, :
            ]

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]

        # Ultra hack: Move the perceived object position up by 0.1 meters
        # from copy import deepcopy
        # D = [0.00, 0.02, -0.02]
        # D = [0.0, 0.0, 0.015]
        # D = [0.0, 0.0, -0.015]
        # D = [0.0, 0.01, 0.0]
        # DX = D[0]
        # DY = D[1]
        # DZ = D[2]
        # self.object_state = deepcopy(self.object_state)
        # self.object_state[:, 0] += DX
        # self.object_state[:, 1] += DY
        # self.object_state[:, 2] += DZ
        # self.object_pose = deepcopy(self.object_pose)
        # self.object_pose[:, 0] += DX
        # self.object_pose[:, 1] += DY
        # self.object_pose[:, 2] += DZ
        # self.object_pos = deepcopy(self.object_pos)
        # self.object_pos[:, 0] += DX
        # self.object_pos[:, 1] += DY
        # self.object_pos[:, 2] += DZ

        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # Update object state queue
        self.object_state_queue = self.update_queue(
            queue=self.object_state_queue, current_values=self.object_state
        )

        # Observed object state
        self.observed_object_state = self.object_state.clone()
        use_object_state_delay_noise = self.cfg["env"]["useObjectStateDelayNoise"]
        if use_object_state_delay_noise:
            # Sample a delay index from the queue
            delay_index = torch.randint(
                0,
                self.object_state_queue.shape[1],
                (self.num_envs,),
                device=self.device,
            )
            self.observed_object_state[:] = self.object_state_queue[
                torch.arange(self.num_envs), delay_index
            ].clone()

            # Add noise to the observed object state
            xyz_noise_std = self.cfg["env"]["objectStateXyzNoiseStd"]
            rotation_noise_degrees = self.cfg["env"]["objectStateRotationNoiseDegrees"]
            self.observed_object_state[:, 0:3] += (
                torch.randn_like(self.observed_object_state[:, 0:3]) * xyz_noise_std
            )
            self.observed_object_state[:, 3:7] = self.sample_delta_quat_xyzw(
                input_quat_xyzw=self.observed_object_state[:, 3:7],
                delta_rotation_degrees=rotation_noise_degrees,
            )

        self.observed_object_pose = self.observed_object_state[:, 0:7]
        self.observed_object_pos = self.observed_object_state[:, 0:3]
        self.observed_object_rot = self.observed_object_state[:, 3:7]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # HACK: Move offsets down by X along x axis to grab hammer on bottom of handle
        use_hack_object_pos_offset = self.cfg["env"]["use_hack_object_pos_offset"]
        hack_object_pos_offset = self.cfg["env"]["hack_object_pos_offset"]
        if use_hack_object_pos_offset:
            self.object_pos_offset = torch.tensor(
                [-hack_object_pos_offset, 0.0, 0.0], device=self.device
            )[None].repeat_interleave(self.num_envs, dim=0)
            self.object_pos = self.object_pos + quat_rotate(
                self.object_rot, self.object_pos_offset
            )
            self.goal_pos = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_pos_offset
            )

        self.palm_center_offset = (
            torch.from_numpy(self.palm_offset)
            .to(self.device)
            .repeat((self.num_envs, 1))
        )
        self._palm_state = self.rigid_body_states[:, self.palm_handle][:, 0:13]
        self._palm_pos = self.rigid_body_states[:, self.palm_handle][:, 0:3]
        self._palm_rot = self.rigid_body_states[:, self.palm_handle][:, 3:7]
        self.palm_center_pos = self._palm_pos + quat_rotate(
            self._palm_rot, self.palm_center_offset
        )

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 0:13
        ]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 0:3
        ]
        self.fingertip_rot = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 3:7
        ]

        if not isinstance(self.fingertip_offsets, torch.Tensor):
            self.fingertip_offsets = (
                torch.from_numpy(self.fingertip_offsets)
                .to(self.device)
                .repeat((self.num_envs, 1, 1))
            )

        if hasattr(self, "fingertip_pos_rel_object"):
            self.fingertip_pos_rel_object_prev[:, :, :] = self.fingertip_pos_rel_object
        else:
            self.fingertip_pos_rel_object_prev = None

        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos).to(self.device)
        for i in range(self.num_fingertips):
            self.fingertip_pos_offset[:, i] = self.fingertip_pos[:, i] + quat_rotate(
                self.fingertip_rot[:, i], self.fingertip_offsets[:, i]
            )

        obj_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_fingertips, 1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(
            self.fingertip_pos_rel_object, dim=-1
        )

        # when episode ends or target changes we reset this to -1, this will initialize it to the actual distance on the 1st frame of the episode
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0,
            self.curr_fingertip_distances,
            self.closest_fingertip_dist,
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0,
            self.curr_fingertip_distances[:, 0],
            self.furthest_hand_dist,
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(
            1, self.num_fingertips, 1
        )
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_center_repeat

        if self.fingertip_pos_rel_object_prev is None:
            self.fingertip_pos_rel_object_prev = self.fingertip_pos_rel_object.clone()

        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + quat_rotate(
                self.object_rot,
                self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier,
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot,
                self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier,
            )
            self.observed_obj_keypoint_pos[:, i] = (
                self.observed_object_pos
                + quat_rotate(
                    self.observed_object_rot,
                    self.object_keypoint_offsets[:, i]
                    * self.object_scale_noise_multiplier,
                )
            )

            self.obj_keypoint_pos_fixed_size[:, i] = self.object_pos + quat_rotate(
                self.object_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )
            self.goal_keypoint_pos_fixed_size[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        self.observed_keypoints_rel_goal = (
            self.observed_obj_keypoint_pos - self.goal_keypoint_pos
        )
        self.keypoints_rel_goal_fixed_size = (
            self.obj_keypoint_pos_fixed_size - self.goal_keypoint_pos_fixed_size
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(
            1, self.num_keypoints, 1
        )
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_center_repeat
        self.observed_keypoints_rel_palm = (
            self.observed_obj_keypoint_pos - palm_center_repeat
        )

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        self.keypoint_distances_l2_fixed_size = torch.norm(
            self.keypoints_rel_goal_fixed_size, dim=-1
        )

        # furthest keypoint from the goal
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values
        self.keypoints_max_dist_fixed_size = self.keypoint_distances_l2_fixed_size.max(
            dim=-1
        ).values

        # this is the closest the keypoint had been to the target in the current episode (for the furthest keypoint of all)
        # make sure we initialize this value before using it for obs or rewards
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0,
            self.keypoints_max_dist,
            self.closest_keypoint_max_dist,
        )
        self.closest_keypoint_max_dist_fixed_size = torch.where(
            self.closest_keypoint_max_dist_fixed_size < 0.0,
            self.keypoints_max_dist_fixed_size,
            self.closest_keypoint_max_dist_fixed_size,
        )

    def populate_obs_and_states_buffers(self) -> None:
        num_dofs = self.num_hand_arm_dofs
        obs_dict = {}

        # We first fill in the obs_dict with the values that should be given to the critic, which are clean
        # Then after creating state_buf, we add the noisy delayed object state observations and other observation changes to the policy's obs_buf

        ## POLICY OBSERVATIONS ##
        # dof positions
        obs_dict["joint_pos"] = unscale(
            self.arm_hand_dof_pos[:, :num_dofs],
            self.arm_hand_dof_lower_limits[:num_dofs],
            self.arm_hand_dof_upper_limits[:num_dofs],
        )
        # dof velocities
        obs_dict["joint_vel"] = self.arm_hand_dof_vel[:, :num_dofs]
        # prev action targets
        obs_dict["prev_action_targets"] = self.prev_targets.clone()
        # palm pos
        obs_dict["palm_pos"] = self.palm_center_pos
        # palm rot
        obs_dict["palm_rot"] = self._palm_state[:, 3:7]
        # object rot
        obs_dict["object_rot"] = self.object_state[:, 3:7]
        # keypoint distances relative to the palm of the hand
        keypoint_rel_pos_size = 3 * self.num_keypoints
        obs_dict["keypoints_rel_palm"] = self.keypoints_rel_palm.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        # keypoint distances relative to the goal
        obs_dict["keypoints_rel_goal"] = self.keypoints_rel_goal.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        # fingertip pos relative to the palm of the hand
        fingertip_rel_pos_size = 3 * self.num_fingertips
        obs_dict["fingertip_pos_rel_palm"] = self.fingertip_pos_rel_palm.reshape(
            self.num_envs, fingertip_rel_pos_size
        )
        # object scales
        obs_dict["object_scales"] = (
            self.object_scales * self.object_scale_noise_multiplier
        )

        ## CRITIC OBSERVATIONS ##
        # palm linvel, ang vel
        obs_dict["palm_vel"] = self._palm_state[:, 7:13]
        # object vel
        obs_dict["object_vel"] = self.object_state[:, 7:13]
        # object velocity relative to palm, plus a short-horizon object-centric
        # prediction that helps dynamic-grasp policies learn interception.
        obs_dict["object_vel_rel_palm"] = obs_dict["object_vel"] - obs_dict["palm_vel"]
        obs_dict["predicted_keypoints_rel_palm"] = (
            self.keypoints_rel_palm
            + obs_dict["object_vel_rel_palm"][:, None, 0:3]
            * self.dynamic_grasp_intercept_lead_time
        ).reshape(self.num_envs, keypoint_rel_pos_size)
        (
            obs_dict["object_pointcloud_rel_palm"],
            obs_dict["object_pointcloud_centroid_rel_palm"],
            obs_dict["object_pointcloud_vel_rel_palm"],
            obs_dict["object_pointcloud_tracking_confidence"],
        ) = self._object_pointcloud_observation(
            object_pos=self.object_pos,
            object_rot=self.object_rot,
            object_vel=obs_dict["object_vel"],
            palm_pos=self.palm_center_pos,
            palm_rot=obs_dict["palm_rot"],
            palm_vel=obs_dict["palm_vel"],
            add_noise=False,
        )
        (
            obs_dict["affordance_pos_rel_palm"],
            obs_dict["future_affordance_pos_rel_palm"],
            obs_dict["affordance_axis_rel_palm"],
            obs_dict["affordance_confidence"],
        ) = self._dynamic_affordance_observation(
            object_pos=self.object_pos,
            object_vel=obs_dict["object_vel"],
            palm_pos=self.palm_center_pos,
            palm_rot=obs_dict["palm_rot"],
            add_noise=False,
        )
        # closest distance to the furthest keypoint, achieved so far in this episode
        obs_dict["closest_keypoint_max_dist"] = (
            self.closest_keypoint_max_dist.unsqueeze(-1)
        )
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            obs_dict["closest_keypoint_max_dist"] = (
                self.closest_keypoint_max_dist_fixed_size.unsqueeze(-1)
            )

        # closest distance between a fingertip and an object achieved since last target reset
        # this should help the critic predict the anticipated fingertip reward
        obs_dict["closest_fingertip_dist"] = self.closest_fingertip_dist.unsqueeze(-1)
        # indicates whether we already lifted the object from the table or not, should help the critic be more accurate
        obs_dict["lifted_object"] = self.lifted_object.unsqueeze(-1)
        # this should help the critic predict the future rewards better and anticipate the episode termination
        obs_dict["progress"] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
        obs_dict["successes"] = torch.log(self.successes + 1).unsqueeze(-1)
        # this is where we will add the reward observation
        reward_obs_scale = 0.01
        obs_dict["reward"] = reward_obs_scale * self.rew_buf

        # ##############################################################################################################
        # Create state_buf
        # ##############################################################################################################
        self.states_buf = torch.cat(
            [obs_dict[k].reshape(self.num_envs, -1) for k in self.state_list], dim=-1
        )

        # Policy observations
        # Add noisy delayed object state observations
        use_object_state_delay_noise = self.cfg["env"]["useObjectStateDelayNoise"]
        if use_object_state_delay_noise:
            # Add noise
            obs_dict["object_rot"] = self.observed_object_state[:, 3:7]
            obs_dict["object_vel"] = (
                self.observed_object_state[:, 7:13] * self.turn_off_object_vel_obs_scale
            )
            keypoint_rel_pos_size = 3 * self.num_keypoints
            obs_dict["keypoints_rel_palm"] = self.observed_keypoints_rel_palm.reshape(
                self.num_envs, keypoint_rel_pos_size
            )
            obs_dict["keypoints_rel_goal"] = self.observed_keypoints_rel_goal.reshape(
                self.num_envs, keypoint_rel_pos_size
            )

        # Add noise to joint velocities
        obs_dict["joint_vel"] += (
            torch.randn_like(obs_dict["joint_vel"])
            * self.cfg["env"]["jointVelocityObsNoiseStd"]
        )
        # palm linvel, ang vel
        obs_dict["palm_vel"] = (
            self._palm_state[:, 7:13] * self.turn_off_palm_vel_obs_scale
        )
        # object vel
        obs_dict["object_vel"] = (
            self.object_state[:, 7:13] * self.turn_off_object_vel_obs_scale
        )
        obs_dict["object_vel_rel_palm"] = (
            obs_dict["object_vel"] - obs_dict["palm_vel"]
        )
        keypoints_rel_palm_for_policy = obs_dict["keypoints_rel_palm"].reshape(
            self.num_envs, self.num_keypoints, 3
        )
        obs_dict["predicted_keypoints_rel_palm"] = (
            keypoints_rel_palm_for_policy
            + obs_dict["object_vel_rel_palm"][:, None, 0:3]
            * self.dynamic_grasp_intercept_lead_time
        ).reshape(self.num_envs, keypoint_rel_pos_size)
        observed_object_vel = (
            self.observed_object_state[:, 7:13] * self.turn_off_object_vel_obs_scale
        )
        (
            obs_dict["object_pointcloud_rel_palm"],
            obs_dict["object_pointcloud_centroid_rel_palm"],
            obs_dict["object_pointcloud_vel_rel_palm"],
            obs_dict["object_pointcloud_tracking_confidence"],
        ) = self._object_pointcloud_observation(
            object_pos=self.observed_object_pos,
            object_rot=self.observed_object_rot,
            object_vel=observed_object_vel,
            palm_pos=self.palm_center_pos,
            palm_rot=obs_dict["palm_rot"],
            palm_vel=obs_dict["palm_vel"],
            add_noise=True,
        )
        (
            obs_dict["affordance_pos_rel_palm"],
            obs_dict["future_affordance_pos_rel_palm"],
            obs_dict["affordance_axis_rel_palm"],
            obs_dict["affordance_confidence"],
        ) = self._dynamic_affordance_observation(
            object_pos=self.observed_object_pos,
            object_vel=observed_object_vel,
            palm_pos=self.palm_center_pos,
            palm_rot=obs_dict["palm_rot"],
            add_noise=True,
        )
        if self.object_pointcloud_velocity_from_centroid:
            obs_dict["object_pointcloud_vel_rel_palm"] = (
                self._object_pointcloud_centroid_velocity_observation(
                    obs_dict["object_pointcloud_centroid_rel_palm"],
                    add_noise=True,
                )
            )
        # closest distance to the furthest keypoint, achieved so far in this episode
        obs_dict["closest_keypoint_max_dist"] = (
            self.closest_keypoint_max_dist.unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # closest distance between a fingertip and an object achieved since last target reset
        # this should help the critic predict the anticipated fingertip reward
        obs_dict["closest_fingertip_dist"] = (
            self.closest_fingertip_dist.unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # indicates whether we already lifted the object from the table or not, should help the critic be more accurate
        obs_dict["lifted_object"] = (
            self.lifted_object.unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # this should help the critic predict the future rewards better and anticipate the episode termination
        obs_dict["progress"] = (
            torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
            * self.turn_off_extra_obs_scale
        )
        obs_dict["successes"] = (
            torch.log(self.successes + 1).unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # this is where we will add the reward observation
        reward_obs_scale = 0.01
        obs_dict["reward"] = (
            reward_obs_scale * self.rew_buf * self.turn_off_extra_obs_scale
        )

        # ##############################################################################################################
        # Create obs_buf
        # ##############################################################################################################
        self.obs_buf = torch.cat(
            [obs_dict[k].reshape(self.num_envs, -1) for k in self.obs_list], dim=-1
        )

        # Update obs queue
        self.obs_queue = self.update_queue(
            queue=self.obs_queue, current_values=self.obs_buf
        )

        # Modify obs to be delayed
        use_obs_delay = self.cfg["env"]["useObsDelay"]
        if use_obs_delay:
            # Sample a delay index from the queue
            delay_index = torch.randint(
                0, self.obs_queue.shape[1], (self.num_envs,), device=self.device
            )
            self.obs_buf[:] = self.obs_queue[
                torch.arange(self.num_envs), delay_index
            ].clone()

        # HACK: For testing delay, force a fixed full delay
        FORCE_FULL_DELAY = False
        if FORCE_FULL_DELAY:
            self.obs_buf[:] = self.obs_queue[:, -1].clone()

        # Default CHECK_WITH_COMPUTED_OBS = False
        # Set to True to check if the observations are computed correctly
        CHECK_WITH_COMPUTED_OBS = False
        if CHECK_WITH_COMPUTED_OBS:
            # Create urdf object
            if not hasattr(self, "urdf_object"):
                self.urdf_object = create_urdf_object(
                    robot_name="iiwa14_left_sharpa_adjusted_restricted"
                )

            computed_obs = compute_observation(
                q=self.arm_hand_dof_pos.cpu().numpy(),
                qd=self.arm_hand_dof_vel.cpu().numpy(),
                prev_action_targets=self.prev_targets.cpu().numpy(),
                object_pose=self.object_pose.cpu().numpy(),
                goal_object_pose=self.goal_pose.cpu().numpy(),
                object_scales=self.object_scales.cpu().numpy(),
                urdf=self.urdf_object,
                obs_list=self.obs_list,
            )
            computed_obs = torch.from_numpy(computed_obs).float().to(self.device)

            # Validate
            assert computed_obs.shape == (self.num_envs, len(OBS_NAMES)), (
                f"computed_obs.shape: {computed_obs.shape}, expected: ({self.num_envs}, {len(OBS_NAMES)})"
            )
            assert self.obs_buf.shape == computed_obs.shape, (
                f"self.obs_buf.shape: {self.obs_buf.shape}, expected: {computed_obs.shape}"
            )
            num_errors = 0
            for i, name in enumerate(OBS_NAMES):
                val_orig = self.obs_buf[0, i].item()
                val_computed = computed_obs[0, i].item()
                print(
                    f"{name}: original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}"
                )
                # Note that there are some reasonably large 2e-3 differences in the palm vel computation
                # Maybe from Jacobian computation being different from the maximal sim computation
                if abs(val_orig - val_computed) > 1e-2:
                    num_errors += 1
                    print("--------------------------------")
                    print(
                        f"Error: {name}: original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}"
                    )
                    print("--------------------------------")
            print("=" * 100)
            print(f"num_errors: {num_errors}")
            print("=" * 100)
            breakpoint()

    @property
    def turn_off_palm_vel_obs_scale(self) -> float:
        # 1 means not turned off
        # 0.5 means half turned off
        # 0 means turned off
        if self.cfg["env"]["turn_off_palm_vel_obs"]:
            scale = 0.0
        elif self.cfg["env"]["turn_off_palm_vel_obs_slowly"]:
            if self.cfg["env"]["use_obs_dropout"]:
                prob_of_turn_off = self._tyler_curriculum_scale
                scale = 0.0 if random.random() < prob_of_turn_off else 1.0
            else:
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0
        self.extras["turn_off_palm_vel_obs_scale"] = scale
        return scale

    @property
    def turn_off_object_vel_obs_scale(self) -> float:
        # 1 means not turned off
        # 0.5 means half turned off
        # 0 means turned off
        if self.cfg["env"]["turn_off_object_vel_obs"]:
            scale = 0.0
        elif self.cfg["env"]["turn_off_object_vel_obs_slowly"]:
            if self.cfg["env"]["use_obs_dropout"]:
                prob_of_turn_off = self._tyler_curriculum_scale
                scale = 0.0 if random.random() < prob_of_turn_off else 1.0
            else:
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0
        self.extras["turn_off_object_vel_obs_scale"] = scale
        return scale

    @property
    def turn_off_extra_obs_scale(self) -> float:
        # 1 means not turned off
        # 0.5 means half turned off
        # 0 means turned off
        if self.cfg["env"]["turn_off_extra_obs"]:
            scale = 0.0
        elif self.cfg["env"]["turn_off_extra_obs_slowly"]:
            if self.cfg["env"]["use_obs_dropout"]:
                # When curriculum_scale is 0.0, turn_off_extra_obs_scale is 1.0, which means no extra obs are turned off
                # When curriculum_scale is 1.0, turn_off_extra_obs_scale is 0.0, which means all extra obs are turned off
                # Smoothly transition between 1.0 and 0.0
                prob_of_turn_off = self._tyler_curriculum_scale
                scale = 0.0 if random.random() < prob_of_turn_off else 1.0
            else:
                # When curriculum_scale is 0.0, turn_off_extra_obs_scale is 1.0, which means no extra obs are turned off
                # When curriculum_scale is 1.0, turn_off_extra_obs_scale is 0.0, which means all extra obs are turned off
                # Smoothly transition between 1.0 and 0.0
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0

        self.extras["turn_off_extra_obs_scale"] = scale
        return scale

    def clamp_obs(self) -> None:
        if self.clamp_abs_observations > 0:
            self.obs_buf.clamp_(
                -self.clamp_abs_observations, self.clamp_abs_observations
            )
            self.states_buf.clamp_(
                -self.clamp_abs_observations, self.clamp_abs_observations
            )

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat(
            (
                q_x.unsqueeze(-1),
                q_y.unsqueeze(-1),
                q_z.unsqueeze(-1),
                q_w.unsqueeze(-1),
            ),
            dim=-1,
        )

        return new_rot

    def _quat_mul_xyzw_torch(self, quat_l: Tensor, quat_r: Tensor) -> Tensor:
        x_l, y_l, z_l, w_l = quat_l.unbind(dim=-1)
        x_r, y_r, z_r, w_r = quat_r.unbind(dim=-1)
        quat = torch.stack(
            (
                w_l * x_r + x_l * w_r + y_l * z_r - z_l * y_r,
                w_l * y_r - x_l * z_r + y_l * w_r + z_l * x_r,
                w_l * z_r + x_l * y_r - y_l * x_r + z_l * w_r,
                w_l * w_r - x_l * x_r - y_l * y_r - z_l * z_r,
            ),
            dim=-1,
        )
        return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-8)

    def _sample_object_asset_reset_quat(self, env_ids: Tensor) -> Tensor:
        base_quat = self.object_default_rot[env_ids]
        yaw_enabled = self.object_random_yaw[env_ids]
        yaw = torch_rand_float(
            -math.pi, math.pi, (len(env_ids), 1), device=self.device
        ).squeeze(-1)
        yaw = torch.where(yaw_enabled, yaw, torch.zeros_like(yaw))
        zeros = torch.zeros_like(yaw)
        yaw_quat = torch.stack(
            (zeros, zeros, torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)),
            dim=-1,
        )
        return self._quat_mul_xyzw_torch(yaw_quat, base_quat)

    def reset_target_pose(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        tensor_reset=True,
        is_first_goal=True,
    ) -> None:
        self._reset_target(
            env_ids,
            reset_buf_idxs,
            tensor_reset=tensor_reset,
            is_first_goal=is_first_goal,
        )

        if tensor_reset:
            self.reset_goal_buf[env_ids] = 0
            self.near_goal_steps[env_ids] = 0
            self.prev_total_episode_closest_keypoint_max_dist[env_ids] = (
                self.total_episode_closest_keypoint_max_dist[env_ids]
            )
            self.total_episode_closest_keypoint_max_dist[env_ids] += torch.where(
                self.closest_keypoint_max_dist[env_ids] > 0,
                self.closest_keypoint_max_dist[env_ids],
                torch.zeros_like(self.closest_keypoint_max_dist[env_ids]),
            )
            self.closest_keypoint_max_dist[env_ids] = -1
            self.closest_keypoint_max_dist_fixed_size[env_ids] = -1

    def reset_object_pose(
        self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True
    ):
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            obj_indices = self.object_indices[env_ids]
            table_indices = self.table_indices[env_ids]

            # decide table reset z
            table_reset_z = (
                torch_rand_float(
                    -self.cfg["env"]["tableResetZRange"],
                    self.cfg["env"]["tableResetZRange"],
                    (len(env_ids), 1),
                    device=self.device,
                )
                + self.cfg["env"]["tableResetZ"]
            )
            self.table_init_state[env_ids, 2:3] = table_reset_z
            if hasattr(self, "object_table_z_offsets"):
                self.object_init_state[env_ids, 2:3] = (
                    table_reset_z + self.object_table_z_offsets[env_ids, None]
                )
            else:
                self.object_init_state[env_ids, 2:3] = (
                    table_reset_z + self.cfg["env"]["tableObjectZOffset"]
                )

            # reset object
            rand_pos_floats = torch_rand_float(
                -1.0, 1.0, (len(env_ids), 3), device=self.device
            )
            USE_FIXED_INIT_OBJECT_POSE = self.cfg["env"]["useFixedInitObjectPose"]
            if USE_FIXED_INIT_OBJECT_POSE:
                rand_pos_floats[:] = 0.0
            self.root_state_tensor[obj_indices] = self.object_init_state[
                env_ids
            ].clone()
            self.root_state_tensor[table_indices] = self.table_init_state[
                env_ids
            ].clone()

            # indices 0..2 correspond to the object position
            self.root_state_tensor[obj_indices, 0:1] = (
                self.object_init_state[env_ids, 0:1]
                + self.reset_position_noise_x * rand_pos_floats[:, 0:1]
            )
            self.root_state_tensor[obj_indices, 1:2] = (
                self.object_init_state[env_ids, 1:2]
                + self.reset_position_noise_y * rand_pos_floats[:, 1:2]
            )
            self.root_state_tensor[obj_indices, 2:3] = (
                self.object_init_state[env_ids, 2:3]
                + self.reset_position_noise_z * rand_pos_floats[:, 2:3]
            )

            if self.randomize_object_rotation:
                if hasattr(self, "object_default_rot"):
                    new_object_rot = self._sample_object_asset_reset_quat(env_ids)
                    if USE_FIXED_INIT_OBJECT_POSE:
                        new_object_rot = self.object_default_rot[env_ids].clone()
                else:
                    new_object_rot = self.get_random_quat(env_ids)
                    if USE_FIXED_INIT_OBJECT_POSE:
                        new_object_rot[:] = 0.0
                        new_object_rot[:, -1] = 1.0  # xyzw

                        # HACK: Rotate the object by 180 degrees around the z-axis to go from right handed to left handed robot
                        from scipy.spatial.transform import Rotation as R

                        new_object_rot[:] = (
                            torch.from_numpy(
                                R.from_euler("z", 180, degrees=True).as_quat()
                            )
                            .float()
                            .to(self.device)[None]
                        )

                # indices 3,4,5,6 correspond to the rotation quaternion
                self.root_state_tensor[obj_indices, 3:7] = new_object_rot

            self.root_state_tensor[obj_indices, 7:13] = torch.zeros_like(
                self.root_state_tensor[obj_indices, 7:13]
            )
            if self.dynamic_tabletop_grasp:
                self._set_dynamic_object_initial_velocity(env_ids, obj_indices)

            noise_min, noise_max = self.cfg["env"]["objectScaleNoiseMultiplierRange"]
            self.object_scale_noise_multiplier[env_ids] = torch_rand_float(
                noise_min, noise_max, (len(env_ids), 3), device=self.device
            )

        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            obj_indices = self.object_indices[env_ids]
            # TODO: Check if last 6 indices are 0
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[obj_indices, :] = self.root_state_resets[
                reset_buf_idxs[env_ids].cpu(), obj_indices.cpu() % rs_ofs, :
            ].to(self.device)

        # since we reset the object, we also should update distances between fingers and the object
        if tensor_reset:
            self.closest_fingertip_dist[env_ids] = -1
            self.furthest_hand_dist[env_ids] = -1
            self.lifted_object[env_ids] = False
            self.dynamic_grasp_success_steps[env_ids] = 0
            self.prev_dynamic_grasp_contact_like[env_ids] = False
            if self.dynamic_tabletop_grasp and reset_buf_idxs is not None:
                self.dynamic_grasp_object_xy_velocity[env_ids] = self.root_state_tensor[
                    self.object_indices[env_ids], 7:9
                ]
                self.dynamic_grasp_object_yaw_rate[env_ids] = self.root_state_tensor[
                    self.object_indices[env_ids], 12
                ]
        self.deferred_set_actor_root_state_tensor_indexed(
            [self.object_indices[env_ids]]
        )
        self.deferred_set_actor_root_state_tensor_indexed([self.table_indices[env_ids]])

    def deferred_set_actor_root_state_tensor_indexed(
        self, obj_indices: List[Tensor]
    ) -> None:
        self.set_actor_root_state_object_indices.extend(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[Tensor] = self.set_actor_root_state_object_indices
        if not object_indices:
            # nothing to set
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.set_actor_root_state_object_indices = []

    def deferred_set_dof_state_tensor_indexed(self, dof_indices: List[Tensor]) -> None:
        self.set_dof_state_object_indices.extend(dof_indices)

    def set_dof_state_tensor_indexed(self) -> None:
        dof_indices: List[Tensor] = self.set_dof_state_object_indices
        if not dof_indices:
            # nothing to set
            return

        unique_dof_indices = torch.unique(torch.cat(dof_indices).to(torch.int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(unique_dof_indices),
            len(unique_dof_indices),
        )

        self.set_dof_state_object_indices = []

    def reset_idx(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        episode_reset=True,
        tensor_reset=True,
    ) -> None:
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if len(env_ids) == 0:
            return

        if self.randomize and episode_reset:
            self.apply_randomizations(self.randomization_params)

        # Do this before reset_target_pose so that the successes is 0 for the new episode
        if episode_reset and tensor_reset:
            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0

            self.prev_episode_successes[env_ids] = self.successes[env_ids]
            self.successes[env_ids] = 0

            self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
            self.true_objective[env_ids] = 0
            if hasattr(self, "object_pointcloud_velocity_initialized"):
                self.object_pointcloud_velocity_initialized[env_ids] = False
                self.prev_object_pointcloud_centroid_rel_palm[env_ids] = 0.0

            self.prev_episode_closest_keypoint_max_dist[env_ids] = torch.where(
                self.prev_episode_successes[env_ids] > 0,
                self.prev_total_episode_closest_keypoint_max_dist[env_ids]
                / self.prev_episode_successes[env_ids],
                self.total_episode_closest_keypoint_max_dist[env_ids],
            )
            self.total_episode_closest_keypoint_max_dist[env_ids] = 0
            self.prev_total_episode_closest_keypoint_max_dist[env_ids] = 0

            for key in self.rewards_episode.keys():
                self.rewards_episode[key][env_ids] = 0

            if self.save_states:
                self.dump_env_states(env_ids)

            self.extras["scalars"] = dict()
            self.extras["scalars"]["success_tolerance"] = self.success_tolerance

            if self.with_table_force_sensor:
                self.extras["max_table_sensor_force_norm_smoothed"] = (
                    self.max_table_sensor_force_norm_smoothed.mean().item()
                )
                self.table_sensor_forces_raw[env_ids, :] = 0
                self.table_sensor_forces_smoothed[env_ids, :] = 0
                self.max_table_sensor_force_norm_smoothed[env_ids] = 0

        # reset rigid body forces
        if tensor_reset:
            self.rb_forces[env_ids, :, :] = 0.0
            self.rb_torques[env_ids, :, :] = 0.0

        # reset object
        self.reset_object_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)

        # randomize start object poses
        self.reset_target_pose(
            env_ids, reset_buf_idxs, tensor_reset=tensor_reset, is_first_goal=True
        )

        robot_indices = self.robot_indices[env_ids].to(torch.int32)

        # reset random force probabilities
        if tensor_reset:
            self.random_force_prob[env_ids] = self._sample_log_uniform(
                min_value=self.force_prob_range[0],
                max_value=self.force_prob_range[1],
                num_samples=len(env_ids),
            )
            self.random_torque_prob[env_ids] = self._sample_log_uniform(
                min_value=self.torque_prob_range[0],
                max_value=self.torque_prob_range[1],
                num_samples=len(env_ids),
            )
            self.random_lin_vel_impulse_prob[env_ids] = self._sample_log_uniform(
                min_value=self.lin_vel_impulse_prob_range[0],
                max_value=self.lin_vel_impulse_prob_range[1],
                num_samples=len(env_ids),
            )
            self.random_ang_vel_impulse_prob[env_ids] = self._sample_log_uniform(
                min_value=self.ang_vel_impulse_prob_range[0],
                max_value=self.ang_vel_impulse_prob_range[1],
                num_samples=len(env_ids),
            )

        # reset robot
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            delta_max = self.arm_hand_dof_upper_limits - self.hand_arm_default_dof_pos
            delta_min = self.arm_hand_dof_lower_limits - self.hand_arm_default_dof_pos

            rand_dof_floats = torch_rand_float(
                0.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device
            )

            rand_delta = delta_min + (delta_max - delta_min) * rand_dof_floats

            noise_coeff = torch.zeros_like(
                self.hand_arm_default_dof_pos, device=self.device
            )

            noise_coeff[0 : self.num_arm_dofs] = self.reset_dof_pos_noise_arm
            noise_coeff[
                self.num_arm_dofs : self.num_hand_arm_dofs
            ] = self.reset_dof_pos_noise_fingers

            robot_pos = self.hand_arm_default_dof_pos + noise_coeff * rand_delta
            robot_pos = tensor_clamp(
                robot_pos,
                self.arm_hand_dof_lower_limits,
                self.arm_hand_dof_upper_limits,
            )

            self.arm_hand_dof_pos[env_ids, :] = robot_pos
            if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                self.blue_robot_arm_hand_dof_pos[env_ids, :] = robot_pos.clone()
                self.blue_robot_arm_hand_dof_vel[env_ids, :] = 0.0

            rand_vel_floats = torch_rand_float(
                -1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device
            )
            self.arm_hand_dof_vel[env_ids, :] = (
                self.reset_dof_vel_noise * rand_vel_floats
            )
            self.prev_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos
            self.cur_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos

        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            self.arm_hand_dof_pos[env_ids, :] = self.dof_resets[
                reset_buf_idxs[env_ids].cpu(), :, 0
            ].to(self.device)
            self.arm_hand_dof_vel[env_ids, :] = self.dof_resets[
                reset_buf_idxs[env_ids].cpu(), :, 1
            ].to(self.device)
            robot_pos = self.arm_hand_dof_pos[env_ids, : self.num_hand_arm_dofs]
            robot_pos = tensor_clamp(
                robot_pos,
                self.arm_hand_dof_lower_limits,
                self.arm_hand_dof_upper_limits,
            )

            self.prev_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos
            self.cur_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos

        if self.should_load_initial_states:
            if len(env_ids) > self.num_initial_states:
                print(
                    f"Not enough initial states to load {len(env_ids)}/{self.num_initial_states}..."
                )
            else:
                if self.initial_state_idx + len(env_ids) > self.num_initial_states:
                    self.initial_state_idx = 0

                dof_states_to_load = self.initial_dof_state_tensors[
                    self.initial_state_idx : self.initial_state_idx + len(env_ids)
                ]
                self.dof_state.reshape([self.num_envs, -1, *self.dof_state.shape[1:]])[
                    env_ids
                ] = dof_states_to_load.clone()
                root_state_tensors_to_load = self.initial_root_state_tensors[
                    self.initial_state_idx : self.initial_state_idx + len(env_ids)
                ]
                cube_object_idx = self.object_indices[0]
                self.root_state_tensor.reshape(
                    [self.num_envs, -1, *self.root_state_tensor.shape[1:]]
                )[env_ids, cube_object_idx] = root_state_tensors_to_load[
                    :, cube_object_idx
                ].clone()

                self.initial_state_idx += len(env_ids)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )

        self.deferred_set_dof_state_tensor_indexed([robot_indices])
        self.deferred_set_actor_root_state_tensor_indexed(
            self._extra_object_indices(env_ids)
        )

    def update_queue(
        self, queue: torch.Tensor, current_values: torch.Tensor
    ) -> torch.Tensor:
        N, T, D = queue.shape
        assert current_values.shape == (N, D), (
            f"current_values.shape: {current_values.shape}, expected: ({N}, {D})"
        )

        # Update queue
        # queue is reset on episode start
        # This means we need to fill the entire queue with the current values
        # queue shape is (N, T, D), where N is num envs, T is queue length, D is dimension of the buffer
        is_episode_start = self.progress_buf == 1
        assert is_episode_start.shape == (N,), (
            f"is_episode_start.shape: {is_episode_start.shape}, expected: ({N})"
        )

        queue[:] = torch.where(
            is_episode_start.unsqueeze(1).unsqueeze(1),
            current_values.unsqueeze(1).repeat_interleave(T, dim=1),
            queue,
        )

        # Roll the queue down by 1, then update index=0 with the new action
        queue[:, 1:] = queue[:, :-1].clone()
        queue[:, 0] = current_values.clone()
        return queue

    def pre_physics_step(
        self, actions, joint_pos_targets: Optional[torch.Tensor] = None
    ):
        PRINT_TIME_SINCE_LAST_STEP = False
        if PRINT_TIME_SINCE_LAST_STEP:
            if not hasattr(self, "last_time"):
                self.last_time = time.time()
            print(
                f"Time since last step: {time.time() - self.last_time:.3f} s, {1.0 / (time.time() - self.last_time):.1f} Hz"
            )
            self.last_time = time.time()

        actions = actions.to(self.device)

        # Update actions queue
        self.action_queue = self.update_queue(
            queue=self.action_queue, current_values=actions
        )

        # Modify actions to be delayed
        use_action_delay = self.cfg["env"]["useActionDelay"]
        if use_action_delay:
            # Sample a delay index from the queue
            delay_index = torch.randint(
                0, self.action_queue.shape[1], (self.num_envs,), device=self.device
            )
            actions = self.action_queue[
                torch.arange(self.num_envs), delay_index
            ].clone()

        self.actions = actions.clone()
        action_penalty_dim = min(
            self.actions.shape[1], self.prev_actions_for_penalty.shape[1]
        )
        self.action_delta_for_penalty.zero_()
        self.action_delta_for_penalty[:, :action_penalty_dim] = (
            self.actions[:, :action_penalty_dim]
            - self.prev_actions_for_penalty[:, :action_penalty_dim]
        )
        self.prev_actions_for_penalty[:, :action_penalty_dim] = self.actions[
            :, :action_penalty_dim
        ]

        if self.privileged_actions:
            torque_actions = actions[:, :3]
            actions = actions[:, 3:]

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        combined_random_env_ids = torch.cat(
            [reset_env_ids, reset_goal_env_ids, reset_goal_env_ids]
        )
        uniques, counts = combined_random_env_ids.unique(return_counts=True)
        reset_goal_env_ids = uniques[counts == 2]
        self.reset_target_pose(reset_goal_env_ids, None, is_first_goal=False)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids, None)
            self.prev_actions_for_penalty[
                reset_env_ids, :action_penalty_dim
            ] = self.actions[reset_env_ids, :action_penalty_dim]
            self.action_delta_for_penalty[reset_env_ids, :] = 0.0
            self.prev_target_delta_for_penalty[reset_env_ids, :] = 0.0
            self.target_delta_for_penalty[reset_env_ids, :] = 0.0
            self.target_accel_for_penalty[reset_env_ids, :] = 0.0

        control_actions = actions[:, : self.num_hand_arm_dofs]
        assert control_actions.shape == (self.num_envs, self.num_hand_arm_dofs), (
            f"control_actions.shape: {control_actions.shape}, expected: "
            f"({self.num_envs}, {self.num_hand_arm_dofs})"
        )

        if self.policy_action_interface == "joint_target":
            lower_limits = self.arm_hand_dof_lower_limits[: self.num_hand_arm_dofs]
            upper_limits = self.arm_hand_dof_upper_limits[: self.num_hand_arm_dofs]
            target_center = tensor_clamp(
                self.hand_arm_default_dof_pos[: self.num_hand_arm_dofs],
                lower_limits,
                upper_limits,
            )
            positive_span = upper_limits - target_center
            negative_span = target_center - lower_limits
            targets = torch.where(
                control_actions >= 0.0,
                target_center.unsqueeze(0)
                + control_actions * positive_span.unsqueeze(0),
                target_center.unsqueeze(0)
                + control_actions * negative_span.unsqueeze(0),
            )
            self.cur_targets[:, : self.num_arm_dofs] = (
                self.arm_moving_average * targets[:, : self.num_arm_dofs]
                + (1.0 - self.arm_moving_average)
                * self.prev_targets[:, : self.num_arm_dofs]
            )
            self.cur_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs] = (
                self.hand_moving_average
                * targets[:, self.num_arm_dofs : self.num_hand_arm_dofs]
                + (1.0 - self.hand_moving_average)
                * self.prev_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs]
            )
            self.cur_targets[:, : self.num_hand_arm_dofs] = tensor_clamp(
                self.cur_targets[:, : self.num_hand_arm_dofs],
                lower_limits,
                upper_limits,
            )
        else:
            if self.use_relative_control:
                # arm relative to current position
                targets = (
                    self.arm_hand_dof_pos[:, : self.num_arm_dofs]
                    + self.hand_dof_speed_scale
                    * self.dt
                    * self.actions[:, : self.num_arm_dofs]
                )
                self.cur_targets[:, : self.num_arm_dofs] = tensor_clamp(
                    targets,
                    self.arm_hand_dof_lower_limits[: self.num_arm_dofs],
                    self.arm_hand_dof_upper_limits[: self.num_arm_dofs],
                )
            else:
                # arm relative to previous target
                targets = (
                    self.prev_targets[:, : self.num_arm_dofs]
                    + self.hand_dof_speed_scale
                    * self.dt
                    * self.actions[:, : self.num_arm_dofs]
                )
                self.cur_targets[:, : self.num_arm_dofs] = tensor_clamp(
                    targets,
                    self.arm_hand_dof_lower_limits[: self.num_arm_dofs],
                    self.arm_hand_dof_upper_limits[: self.num_arm_dofs],
                )

            # Smooth arm
            self.cur_targets[:, : self.num_arm_dofs] = (
                self.arm_moving_average * self.cur_targets[:, : self.num_arm_dofs]
                + (1.0 - self.arm_moving_average)
                * self.prev_targets[:, : self.num_arm_dofs]
            )

            # hand
            self.cur_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs] = scale(
                control_actions[:, self.num_arm_dofs : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[
                    self.num_arm_dofs : self.num_hand_arm_dofs
                ],
                self.arm_hand_dof_upper_limits[
                    self.num_arm_dofs : self.num_hand_arm_dofs
                ],
            )
            self.cur_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs] = (
                self.hand_moving_average
                * self.cur_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs]
                + (1.0 - self.hand_moving_average)
                * self.prev_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs]
            )
            self.cur_targets[
                :, self.num_arm_dofs : self.num_hand_arm_dofs
            ] = tensor_clamp(
                self.cur_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[
                    self.num_arm_dofs : self.num_hand_arm_dofs
                ],
                self.arm_hand_dof_upper_limits[
                    self.num_arm_dofs : self.num_hand_arm_dofs
                ],
            )

        # Default CHECK_WITH_COMPUTED_JOINT_POS_TARGETS = False
        # Set to True to check if the computed joint pos targets are correct
        CHECK_WITH_COMPUTED_JOINT_POS_TARGETS = False
        if CHECK_WITH_COMPUTED_JOINT_POS_TARGETS:
            computed_joint_pos_targets = compute_joint_pos_targets(
                actions=self.actions.cpu().numpy(),
                prev_targets=self.prev_targets.cpu().numpy(),
                hand_moving_average=self.hand_moving_average,
                arm_moving_average=self.arm_moving_average,
                hand_dof_speed_scale=self.hand_dof_speed_scale,
                dt=self.dt,
            )
            computed_joint_pos_targets = (
                torch.from_numpy(computed_joint_pos_targets).float().to(self.device)
            )
            assert computed_joint_pos_targets.shape == (
                self.num_envs,
                self.num_hand_arm_dofs,
            ), (
                f"computed_joint_pos_targets.shape: {computed_joint_pos_targets.shape}, expected: ({self.num_envs}, {self.num_hand_arm_dofs})"
            )
            assert self.cur_targets.shape == computed_joint_pos_targets.shape, (
                f"self.cur_targets.shape: {self.cur_targets.shape}, expected: {computed_joint_pos_targets.shape}"
            )

            num_errors = 0
            for i, name in enumerate(self.joint_names):
                val_orig = self.cur_targets[0, i].item()
                val_computed = computed_joint_pos_targets[0, i].item()
                print(
                    f"{name} (idx {i}): original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}"
                )
                if abs(val_orig - val_computed) > 1e-2:
                    num_errors += 1
                    print("--------------------------------")
                    print(
                        f"Error: {name} (idx {i}): original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}"
                    )
                    print("--------------------------------")
            print("=" * 100)
            print(f"num_errors: {num_errors}")
            print("=" * 100)
            breakpoint()

        if joint_pos_targets is not None:
            HACK_OVERWRITE = False
            if HACK_OVERWRITE:
                # SUPER HACK
                joint_pos_targets[:, self.num_arm_dofs :] = 0.0
                joint_pos_targets[:, self.num_arm_dofs + 0] = 1.85
                joint_pos_targets[:, self.num_arm_dofs + 1] = 0.2
            self.cur_targets[:, : self.num_hand_arm_dofs] = joint_pos_targets.clone()

        # print(f"self.cur_targets: {self.cur_targets[0, 7:]}")
        # print(f"self.arm_dof_pos: {self.arm_hand_dof_pos[0, 7:]}")
        # print()

        if self._DO_NOT_MOVE:
            self.cur_targets[:, :] = self.prev_targets[:, :]

        target_delta = (
            self.cur_targets[:, : self.num_hand_arm_dofs]
            - self.prev_targets[:, : self.num_hand_arm_dofs]
        )
        self.target_delta_for_penalty[:, :] = target_delta
        self.target_accel_for_penalty[:, :] = (
            target_delta - self.prev_target_delta_for_penalty
        )
        if len(reset_env_ids) > 0:
            self.target_delta_for_penalty[reset_env_ids, :] = 0.0
            self.target_accel_for_penalty[reset_env_ids, :] = 0.0
        self.prev_target_delta_for_penalty[:, :] = self.target_delta_for_penalty

        self.prev_targets[:, :] = self.cur_targets[:, :]

        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.cur_targets[:, self.num_hand_arm_dofs :] = self.cur_targets[
                :, : self.num_hand_arm_dofs
            ].clone()
            self.blue_robot_arm_hand_dof_pos[:] = self.cur_targets[
                :, self.num_hand_arm_dofs :
            ].clone()
            self.blue_robot_arm_hand_dof_vel[:] = 0.0

            blue_robot_indices = self.blue_robot_indices.to(torch.int32)
            self.deferred_set_dof_state_tensor_indexed([blue_robot_indices])

        self.set_dof_state_tensor_indexed()
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

        # Random forces
        if self.force_scale > 0.0 or self.torque_scale > 0.0:
            if self.force_scale > 0.0:
                self.rb_forces *= torch.pow(
                    self.force_decay, self.dt / self.force_decay_interval
                )

                # apply new forces
                force_indices = (
                    torch.rand(self.num_envs, device=self.device)
                    < self.random_force_prob
                ).nonzero()
                self.rb_forces[force_indices, self.object_rb_handles, :] = (
                    torch.randn(
                        self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                        device=self.device,
                    )
                    * self.object_rb_masses
                    * self.force_scale
                )

                if self.force_only_when_lifted:
                    # self.rb_forces is (N, R, 3), assuming there are R rigid bodies per env
                    # self.lifted_object is (N,), True if the object is lifted
                    self.rb_forces[:, self.object_rb_handles, :] *= (
                        self.lifted_object.unsqueeze(1).unsqueeze(2)
                    )

            if self.torque_scale > 0.0:
                self.rb_torques *= torch.pow(
                    self.torque_decay, self.dt / self.torque_decay_interval
                )

                # apply new torques
                torque_indices = (
                    torch.rand(self.num_envs, device=self.device)
                    < self.random_torque_prob
                ).nonzero()
                self.rb_torques[torque_indices, self.object_rb_handles, :] = (
                    torch.randn(
                        self.rb_torques[
                            torque_indices, self.object_rb_handles, :
                        ].shape,
                        device=self.device,
                    )
                    * self.object_rb_masses  # in theory should do inertia, but harder to do this, so just use mass for now
                    * self.torque_scale
                )

                if self.torque_only_when_lifted:
                    # self.rb_torques is (N, R, 3), assuming there are R rigid bodies per env
                    # self.lifted_object is (N,), True if the object is lifted
                    self.rb_torques[:, self.object_rb_handles, :] *= (
                        self.lifted_object.unsqueeze(1).unsqueeze(2)
                    )

            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                gymtorch.unwrap_tensor(self.rb_torques),
                gymapi.ENV_SPACE,
            )

        # Random velocity impulses
        if self.lin_vel_impulse_scale > 0.0 or self.ang_vel_impulse_scale > 0.0:
            if self.lin_vel_impulse_scale > 0.0:
                if self.lin_vel_impulse_only_when_lifted:
                    lin_vel_impulse_env_ids = (
                        (
                            (
                                torch.rand(self.num_envs, device=self.device)
                                < self.random_lin_vel_impulse_prob
                            )
                            * self.lifted_object
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                else:
                    lin_vel_impulse_env_ids = (
                        (
                            torch.rand(self.num_envs, device=self.device)
                            < self.random_lin_vel_impulse_prob
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                random_lin_vel_impulses = (
                    torch.randn(self.num_envs, 3, device=self.device)
                    * self.lin_vel_impulse_scale
                )
                self.root_state_tensor[
                    self.object_indices[lin_vel_impulse_env_ids], 7:10
                ] = random_lin_vel_impulses[lin_vel_impulse_env_ids, :]
                self.deferred_set_actor_root_state_tensor_indexed(
                    [self.object_indices[lin_vel_impulse_env_ids]]
                )

            if self.ang_vel_impulse_scale > 0.0:
                if self.ang_vel_impulse_only_when_lifted:
                    ang_vel_impulse_env_ids = (
                        (
                            (
                                torch.rand(self.num_envs, device=self.device)
                                < self.random_ang_vel_impulse_prob
                            )
                            * self.lifted_object
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                else:
                    ang_vel_impulse_env_ids = (
                        (
                            torch.rand(self.num_envs, device=self.device)
                            < self.random_ang_vel_impulse_prob
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                random_ang_vel_impulses = (
                    torch.randn(self.num_envs, 3, device=self.device)
                    * self.ang_vel_impulse_scale
                )
                self.root_state_tensor[
                    self.object_indices[ang_vel_impulse_env_ids], 10:13
                ] = random_ang_vel_impulses[ang_vel_impulse_env_ids, :]
                self.deferred_set_actor_root_state_tensor_indexed(
                    [self.object_indices[ang_vel_impulse_env_ids]]
                )

        # Keyboard applied velocity impulses
        # See keyboard callbacks for more details
        if self.need_apply_vel_impulse_from_keyboard:
            self.root_state_tensor[self.object_indices, 7:13] += (
                torch.from_numpy(self.vel_impulse_from_keyboard)
                .float()
                .to(self.device)[None, :]
            )
            self.deferred_set_actor_root_state_tensor_indexed([self.object_indices])
            self.need_apply_vel_impulse_from_keyboard = False
            self.vel_impulse_from_keyboard[:] = 0.0

        # Teleport object
        if self.need_teleport_object_from_keyboard:
            self.root_state_tensor[self.object_indices, :] = (
                self.object_init_state.clone()
            )
            self.deferred_set_actor_root_state_tensor_indexed([self.object_indices])
            self.need_teleport_object_from_keyboard = False

        self._apply_dynamic_tabletop_motion()

        self.set_actor_root_state_tensor_indexed()

        if self.good_reset_boundary > 0:
            self.temp_root_states_buf[:, self.temp_buffer_index] = (
                self.root_state_tensor.reshape(
                    self.num_envs, -1, self.root_state_tensor.shape[1:]
                ).cpu()
            )
            self.temp_dof_states_buf[:, self.temp_buffer_index] = (
                self.dof_state.reshape(
                    self.num_envs, -1, self.dof_state.shape[1:]
                ).cpu()
            )
            self.temp_buffer_index += 1

        # apply torques
        if self.privileged_actions:
            torque_actions = torque_actions.unsqueeze(1)
            torque_amount = self.privileged_actions_torque
            torque_actions *= torque_amount
            self.action_torques[:, self.object_rb_handles, :] = torque_actions
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                None,
                gymtorch.unwrap_tensor(self.action_torques),
                gymapi.ENV_SPACE,
            )

        USE_LIVE_PLOTTER = False
        if USE_LIVE_PLOTTER:
            self._use_live_plotter()

        RECORD_DATA = self.cfg["env"]["record_data"]
        if RECORD_DATA:
            self._record_data()

    def _use_live_plotter(self):
        if not hasattr(self, "live_plotter"):
            from live_plotter import FastLivePlotter

            # Plot table force raw and smoothed
            # self.live_plotter = FastLivePlotter(
            #     n_plots=1,
            #     titles=["Table Force"],
            #     xlabels=["idx"],
            #     ylabels=["force"],
            #     # ylims=[(self.joint_lower_limits[0], self.joint_upper_limits[0])],
            #     legends=[["raw", "smoothed", "max smoothed"]],
            # )
            # self.table_force_raw_history = []
            # self.table_force_smoothed_history = []
            # self.max_table_sensor_force_norm_smoothed_history = []

            # Plot joint pos and target
            # self.live_plotter = FastLivePlotter(
            #     n_plots=len(self.joint_names),
            #     titles=self.joint_names,
            #     xlabels=["idx"] * len(self.joint_names),
            #     ylabels=["joint pos"] * len(self.joint_names),
            #     ylims=[(self.joint_lower_limits[i], self.joint_upper_limits[i]) for i in range(len(self.joint_names))],
            #     legends=[["pos", "target"]] * len(self.joint_names),
            # )
            # self.joint_pos_history = []
            # self.joint_target_history = []

            # Plot the object velocity penalty
            self.live_plotter = FastLivePlotter(
                n_plots=5,
                titles=[
                    "Linear Velocity Penalty",
                    "Angular Velocity Penalty",
                    "Cumulative Linear Velocity Penalty",
                    "Cumulative Angular Velocity Penalty",
                    "Cumulative Total Reward",
                ],
            )
            self.object_lin_vel_penalty_history = []
            self.object_ang_vel_penalty_history = []
            self.cumulative_object_lin_vel_penalty_history = []
            self.cumulative_object_ang_vel_penalty_history = []
            self.cumulative_total_reward_history = []

        # # Plot table force raw and smoothed
        # ENV_IDX = 0
        # if self.with_table_force_sensor:
        #     table_force = self.table_sensor_forces_raw[ENV_IDX, :3].norm(dim=-1).item()
        #     table_force_smoothed = self.table_sensor_forces_smoothed[ENV_IDX, :3].norm(dim=-1).item()
        #     max_table_sensor_force_norm_smoothed = self.max_table_sensor_force_norm_smoothed[ENV_IDX].item()
        #     self.table_force_raw_history.append(table_force)
        #     self.table_force_smoothed_history.append(table_force_smoothed)
        #     self.max_table_sensor_force_norm_smoothed_history.append(max_table_sensor_force_norm_smoothed)
        #     # Should be (N, 2)
        #     self.live_plotter.plot(
        #         y_data_list=[
        #             np.stack([
        #                 np.array(self.table_force_raw_history),
        #                 np.array(self.table_force_smoothed_history),
        #                 np.array(self.max_table_sensor_force_norm_smoothed_history),
        #             ], axis=-1),
        #         ]
        #     )

        # Plot joint pos and target
        # ENV_IDX = 0
        # joint_pos = self.arm_hand_dof_pos[ENV_IDX].cpu().numpy().copy()
        # joint_target = self.cur_targets[ENV_IDX].cpu().numpy().copy()
        # assert joint_pos.shape == joint_target.shape == (len(self.joint_names),), f"{joint_pos.shape} != {joint_target.shape} != {len(self.joint_names)}"
        # self.joint_pos_history.append(joint_pos)
        # self.joint_target_history.append(joint_target)
        # joint_pos_history = np.stack(self.joint_pos_history, axis=0)
        # joint_target_history = np.stack(self.joint_target_history, axis=0)
        # joint_pos_and_target_history = np.stack([joint_pos_history, joint_target_history], axis=-1)
        # assert joint_pos_and_target_history.shape == (len(self.joint_pos_history), len(self.joint_names), 2), f"{joint_pos_and_target_history.shape} != ({len(self.joint_pos_history)}, {len(self.joint_names)}, 2)"
        # # Should be (N, 2)
        # self.live_plotter.plot(
        #     y_data_list=[
        #         joint_pos_and_target_history[:, i, :] for i in range(len(self.joint_names))
        #     ]
        # )

        # Plot object velocity penalty
        ENV_IDX = 0
        if "rewards_episode" in self.extras and "episode_cumulative" in self.extras:
            # Note to self: these names probably got mixed up
            # episode_cumulative should be rewards_episode and vice versa
            #
            # Should be:
            # rewards_episode is the rewards for the current step
            # episode_cumulative is the cumulative rewards over the current episode
            #
            # Right now:
            # episode_cumulative is the rewards for the current step
            # rewards_episode is the cumulative rewards over the current episode

            rewards_episode = self.extras["rewards_episode"]
            episode_cumulative = self.extras["episode_cumulative"]

            # object_lin_vel_penalty = rewards_episode["object_lin_vel_penalty"].cpu().numpy()[ENV_IDX].item()
            # object_ang_vel_penalty = rewards_episode["object_ang_vel_penalty"].cpu().numpy()[ENV_IDX].item()
            # cumulative_object_lin_vel_penalty = episode_cumulative["object_lin_vel_penalty"].cpu().numpy()[ENV_IDX].item()
            # cumulative_object_ang_vel_penalty = episode_cumulative["object_ang_vel_penalty"].cpu().numpy()[ENV_IDX].item()

            cumulative_object_lin_vel_penalty = (
                rewards_episode["object_lin_vel_penalty"].cpu().numpy()[ENV_IDX].item()
            )
            cumulative_object_ang_vel_penalty = (
                rewards_episode["object_ang_vel_penalty"].cpu().numpy()[ENV_IDX].item()
            )
            object_lin_vel_penalty = (
                episode_cumulative["object_lin_vel_penalty"]
                .cpu()
                .numpy()[ENV_IDX]
                .item()
            )
            object_ang_vel_penalty = (
                episode_cumulative["object_ang_vel_penalty"]
                .cpu()
                .numpy()[ENV_IDX]
                .item()
            )
            cumulative_total_reward = (
                rewards_episode["total_reward"].cpu().numpy()[ENV_IDX].item()
            )

            self.object_lin_vel_penalty_history.append(object_lin_vel_penalty)
            self.object_ang_vel_penalty_history.append(object_ang_vel_penalty)
            self.cumulative_object_lin_vel_penalty_history.append(
                cumulative_object_lin_vel_penalty
            )
            self.cumulative_object_ang_vel_penalty_history.append(
                cumulative_object_ang_vel_penalty
            )
            self.cumulative_total_reward_history.append(cumulative_total_reward)

            self.live_plotter.plot(
                y_data_list=[
                    np.array(self.object_lin_vel_penalty_history),
                    np.array(self.object_ang_vel_penalty_history),
                    np.array(self.cumulative_object_lin_vel_penalty_history),
                    np.array(self.cumulative_object_ang_vel_penalty_history),
                    np.array(self.cumulative_total_reward_history),
                ]
            )
        else:
            print("No rewards_episode or episode_cumulative found in extras")

    def _record_data(self):
        from recorded_data import RecordedData

        N_TIMESTEPS = self.cfg["env"]["record_data_num_steps"]

        # Get data from sim
        robot_root_state = self.root_state_tensor[self.robot_indices, :13].cpu().numpy()
        object_root_state = (
            self.root_state_tensor[self.object_indices, :13].cpu().numpy()
        )
        robot_joint_position = self.arm_hand_dof_pos.cpu().numpy()
        table_root_state = self.root_state_tensor[self.table_indices, :13].cpu().numpy()
        if hasattr(self, "goal_object_indices"):
            goal_root_state = (
                self.root_state_tensor[self.goal_object_indices, :13].cpu().numpy()
            )
        robot_joint_velocity = self.arm_hand_dof_vel.cpu().numpy()
        robot_joint_pos_target = (
            self.cur_targets[:, : self.num_hand_arm_dofs].cpu().numpy()
        )
        observations = self.obs_buf.cpu().numpy()
        actions = self.actions.cpu().numpy()

        # Initialize arrays if not already initialized
        if not hasattr(self, "robot_root_states_array"):
            self.robot_root_states_array = []
            self.object_root_states_array = []
            self.robot_joint_positions_array = []
            self.robot_joint_names = self.joint_names

            self.table_root_states_array = []
            if hasattr(self, "goal_object_indices"):
                self.goal_root_states_array = []
            self.robot_joint_velocities_array = []
            self.robot_joint_pos_targets_array = []

            self.observations_array = []
            self.actions_array = []

        # Append data to arrays
        self.robot_root_states_array.append(robot_root_state)
        self.object_root_states_array.append(object_root_state)
        self.robot_joint_positions_array.append(robot_joint_position)
        self.table_root_states_array.append(table_root_state)
        if hasattr(self, "goal_object_indices"):
            self.goal_root_states_array.append(goal_root_state)
        self.robot_joint_velocities_array.append(robot_joint_velocity)
        self.robot_joint_pos_targets_array.append(robot_joint_pos_target)
        self.observations_array.append(observations)
        self.actions_array.append(actions)
        print(f"Recorded {len(self.robot_root_states_array)} / {N_TIMESTEPS} steps")

        # Save data to file
        if len(self.robot_root_states_array) >= N_TIMESTEPS:
            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            this_dir = Path(__file__).parent
            root_dir = this_dir.parent.parent.parent
            recorded_data_path = root_dir / "recorded_data" / f"{datetime_str}.npz"
            recorded_data_path.parent.mkdir(parents=True, exist_ok=True)

            self.robot_root_states_array = np.stack(
                self.robot_root_states_array, axis=0
            )
            self.object_root_states_array = np.stack(
                self.object_root_states_array, axis=0
            )
            self.robot_joint_positions_array = np.stack(
                self.robot_joint_positions_array, axis=0
            )
            self.table_root_states_array = np.stack(
                self.table_root_states_array, axis=0
            )
            if hasattr(self, "goal_object_indices"):
                self.goal_root_states_array = np.stack(
                    self.goal_root_states_array, axis=0
                )
            self.robot_joint_velocities_array = np.stack(
                self.robot_joint_velocities_array, axis=0
            )
            self.robot_joint_pos_targets_array = np.stack(
                self.robot_joint_pos_targets_array, axis=0
            )
            self.observations_array = np.stack(self.observations_array, axis=0)
            self.actions_array = np.stack(self.actions_array, axis=0)

            assert self.robot_root_states_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                13,
            ), (
                f"{self.robot_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
            )
            assert self.object_root_states_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                13,
            ), (
                f"{self.object_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
            )
            assert self.robot_joint_positions_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                len(self.robot_joint_names),
            ), (
                f"{self.robot_joint_positions_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {len(self.robot_joint_names)})"
            )
            assert self.table_root_states_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                13,
            ), (
                f"{self.table_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
            )
            if hasattr(self, "goal_object_indices"):
                assert self.goal_root_states_array.shape == (
                    N_TIMESTEPS,
                    self.num_envs,
                    13,
                ), (
                    f"{self.goal_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
                )
            assert self.robot_joint_velocities_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                len(self.robot_joint_names),
            ), (
                f"{self.robot_joint_velocities_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {len(self.robot_joint_names)})"
            )
            assert self.robot_joint_pos_targets_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                len(self.robot_joint_names),
            ), (
                f"{self.robot_joint_pos_targets_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {len(self.robot_joint_names)})"
            )
            assert self.observations_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                self.obs_buf.shape[1],
            ), (
                f"{self.observations_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {self.obs_buf.shape[1]})"
            )
            assert self.actions_array.shape == (
                N_TIMESTEPS,
                self.num_envs,
                self.actions.shape[1],
            ), (
                f"{self.actions_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {self.actions.shape[1]})"
            )

            time_array = np.arange(N_TIMESTEPS) * self.dt

            ENV_IDX = 0
            recorded_data = RecordedData(
                robot_root_states_array=self.robot_root_states_array[:, ENV_IDX],
                object_root_states_array=self.object_root_states_array[:, ENV_IDX],
                robot_joint_positions_array=self.robot_joint_positions_array[
                    :, ENV_IDX
                ],
                time_array=time_array,
                robot_joint_names=self.robot_joint_names,
                table_root_states_array=self.table_root_states_array[:, ENV_IDX],
                goal_root_states_array=self.goal_root_states_array[:, ENV_IDX]
                if hasattr(self, "goal_object_indices")
                else None,
                robot_joint_velocities_array=self.robot_joint_velocities_array[
                    :, ENV_IDX
                ],
                robot_joint_pos_targets_array=self.robot_joint_pos_targets_array[
                    :, ENV_IDX
                ],
                observations_array=self.observations_array[:, ENV_IDX],
                actions_array=self.actions_array[:, ENV_IDX],
                object_name=self.cfg["env"]["objectName"],
            )
            recorded_data.to_file(recorded_data_path)
            print(f"Saved recorded data to {recorded_data_path}")
            breakpoint()

            # Reset arrays
            self.robot_root_states_array = []
            self.object_root_states_array = []
            self.robot_joint_positions_array = []
            self.robot_joint_names = self.joint_names

            self.table_root_states_array = []
            if hasattr(self, "goal_object_indices"):
                self.goal_root_states_array = []
            self.robot_joint_velocities_array = []
            self.robot_joint_pos_targets_array = []

            self.observations_array = []
            self.actions_array = []

    @property
    def use_sharpa(self) -> bool:
        return "sharpa" in self.cfg["env"]["asset"]["robot"].lower()

    @property
    def use_right_sharpa(self) -> bool:
        return "right_sharpa" in self.cfg["env"]["asset"]["robot"].lower()

    @property
    def use_left_sharpa(self) -> bool:
        return "left_sharpa" in self.cfg["env"]["asset"]["robot"].lower()

    @property
    def hand_moving_average(self) -> float:
        if self.cfg["env"]["handMovingAverageFinal"] is None or not hasattr(
            self, "_tyler_curriculum_scale"
        ):
            return self.cfg["env"]["handMovingAverage"]
        else:
            return self.interpolate(
                init=self.cfg["env"]["handMovingAverage"],
                final=self.cfg["env"]["handMovingAverageFinal"],
                alpha=self._tyler_curriculum_scale,
            )

    @property
    def arm_moving_average(self) -> float:
        if self.cfg["env"]["armMovingAverageFinal"] is None or not hasattr(
            self, "_tyler_curriculum_scale"
        ):
            return self.cfg["env"]["armMovingAverage"]
        else:
            return self.interpolate(
                init=self.cfg["env"]["armMovingAverage"],
                final=self.cfg["env"]["armMovingAverageFinal"],
                alpha=self._tyler_curriculum_scale,
            )

    @property
    def hand_dof_speed_scale(self) -> float:
        if self.cfg["env"]["dofSpeedScaleFinal"] is None or not hasattr(
            self, "_tyler_curriculum_scale"
        ):
            return self.cfg["env"]["dofSpeedScale"]
        else:
            return self.interpolate(
                init=self.cfg["env"]["dofSpeedScale"],
                final=self.cfg["env"]["dofSpeedScaleFinal"],
                alpha=self._tyler_curriculum_scale,
            )

    @staticmethod
    def interpolate(init, final, alpha: float) -> float:
        assert 0 <= alpha <= 1, f"alpha must be between 0 and 1, got {alpha}"
        return init + (final - init) * alpha

    def post_physics_step(self):
        self.frame_since_restart += 1

        self.progress_buf += 1
        self.randomize_buf += 1

        self._extra_curriculum()

        self._update_tyler_curriculum()

        self.populate_sim_buffers()
        rewards, is_success = self.compute_kuka_reward()
        self.populate_obs_and_states_buffers()

        if self.good_reset_boundary > 0:
            add_indices = torch.where(is_success)[0]
            add_indices = add_indices[add_indices >= self.good_reset_boundary]
            add_indices = add_indices[
                self.temp_buffer_index[add_indices] > self.success_steps
            ]

            if len(add_indices) > 0:
                rs_to_add = torch.stack(
                    [
                        self.temp_root_states_buf[
                            idx,
                            torch.arange(
                                self.temp_buffer_index[idx] - self.success_steps
                            ),
                        ]
                        for idx in add_indices
                    ]
                )
                dof_to_add = torch.stack(
                    [
                        self.temp_dof_states_buf[
                            idx,
                            torch.arange(
                                self.temp_buffer_index[idx] - self.success_steps
                            ),
                        ]
                        for idx in add_indices
                    ]
                )

                num_to_add = len(rs_to_add)

                next_index = self.buffer_index + num_to_add
                self.buffer_length = min(
                    self.buffer_length + num_to_add, self.max_buffer_size
                )

                if next_index >= self.max_buffer_size:
                    num_to_add -= self.max_buffer_size - self.buffer_index
                    self.root_state_resets[self.buffer_index :] = rs_to_add[
                        : self.max_buffer_size - self.buffer_index
                    ]
                    self.root_state_resets[:num_to_add] = rs_to_add[
                        self.max_buffer_size - self.buffer_index :
                    ]
                    self.dof_resets[self.buffer_index :] = dof_to_add[
                        : self.max_buffer_size - self.buffer_index
                    ]
                    self.dof_resets[:num_to_add] = dof_to_add[
                        self.max_buffer_size - self.buffer_index :
                    ]
                else:
                    self.root_state_resets[self.buffer_index : next_index] = rs_to_add
                    self.dof_resets[self.buffer_index : next_index] = dof_to_add

                self.buffer_index = next_index % self.max_buffer_size

                print(
                    f"Added {len(rs_to_add)} states, lifted {self.lifted_object[add_indices].sum().item()}/{len(add_indices)} objects"
                )

            self.temp_buffer_index[torch.where(is_success)[0]] = 0
            self.temp_buffer_index[torch.where(self.reset_buf)[0]] = 0

        self.clamp_obs()

        self._eval_stats(is_success)

        if self.save_states:
            self.accumulate_env_states()

        self._capture_video_if_needed()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            sphere_pose = gymapi.Transform()
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            YELLOW = (1, 1, 0)
            WHITE = (1, 1, 1)
            BLACK = (0, 0, 0)
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.01, 8, 8, sphere_pose, color=YELLOW
            )
            sphere_geom_white = gymutil.WireframeSphereGeometry(
                0.02, 8, 8, sphere_pose, color=WHITE
            )
            sphere_geom_black = gymutil.WireframeSphereGeometry(
                0.01, 8, 8, sphere_pose, color=BLACK
            )

            palm_center_pos_cpu = self.palm_center_pos.cpu().numpy()
            palm_rot_cpu = self._palm_rot.cpu().numpy()

            for i in range(self.num_envs):
                palm_center_transform = gymapi.Transform()
                palm_center_transform.p = gymapi.Vec3(*palm_center_pos_cpu[i])
                palm_center_transform.r = gymapi.Quat(*palm_rot_cpu[i])
                gymutil.draw_lines(
                    sphere_geom_white,
                    self.gym,
                    self.viewer,
                    self.envs[i],
                    palm_center_transform,
                )

            for j in range(self.num_fingertips):
                fingertip_pos_cpu = self.fingertip_pos_offset[:, j].cpu().numpy()
                fingertip_rot_cpu = self.fingertip_rot[:, j].cpu().numpy()

                for i in range(self.num_envs):
                    fingertip_transform = gymapi.Transform()
                    fingertip_transform.p = gymapi.Vec3(*fingertip_pos_cpu[i])
                    fingertip_transform.r = gymapi.Quat(*fingertip_rot_cpu[i])

                    gymutil.draw_lines(
                        sphere_geom,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        fingertip_transform,
                    )

            for i in range(self.num_envs):
                rb_forces_cpu = (
                    self.rb_forces[i, self.object_rb_handles, :]
                    .cpu()
                    .numpy()
                    .squeeze(axis=0)
                )
                assert rb_forces_cpu.shape == (3,), (
                    f"rb_forces_cpu.shape: {rb_forces_cpu.shape}"
                )
                object_pos_cpu = self.object_pos[i].cpu().numpy()
                assert object_pos_cpu.shape == (3,), (
                    f"object_pos_cpu.shape: {object_pos_cpu.shape}"
                )
                start_pos = gymapi.Vec3(*object_pos_cpu)
                MAX_FORCE_NORM = self.force_scale * 0.1  # Often mass is about 0.1 kg
                MAX_VECTOR_LENGTH = 0.3
                force_norm = np.linalg.norm(rb_forces_cpu)
                if force_norm > MAX_FORCE_NORM:
                    rb_forces_cpu = rb_forces_cpu / force_norm * MAX_FORCE_NORM
                vector = rb_forces_cpu * MAX_VECTOR_LENGTH / (MAX_FORCE_NORM + 1e-6)
                end_pos = start_pos + gymapi.Vec3(*vector)
                PURPLE = (1, 0, 1)
                self._draw_debug_line_of_spheres(
                    env=self.envs[i],
                    start_pos=start_pos,
                    end_pos=end_pos,
                    color=PURPLE,
                )

            for i in range(self.num_envs):
                rb_torques_cpu = (
                    self.rb_torques[i, self.object_rb_handles, :]
                    .cpu()
                    .numpy()
                    .squeeze(axis=0)
                )
                assert rb_torques_cpu.shape == (3,), (
                    f"rb_torques_cpu.shape: {rb_torques_cpu.shape}"
                )
                object_pos_cpu = self.object_pos[i].cpu().numpy()
                assert object_pos_cpu.shape == (3,), (
                    f"object_pos_cpu.shape: {object_pos_cpu.shape}"
                )
                start_pos = gymapi.Vec3(*object_pos_cpu)
                MAX_TORQUE_NORM = self.torque_scale * 0.1  # Often mass is about 0.1 kg
                MAX_VECTOR_LENGTH = 0.3
                torque_norm = np.linalg.norm(rb_torques_cpu)
                if torque_norm > MAX_TORQUE_NORM:
                    rb_torques_cpu = rb_torques_cpu / torque_norm * MAX_TORQUE_NORM
                vector = rb_torques_cpu * MAX_VECTOR_LENGTH / (MAX_TORQUE_NORM + 1e-6)
                end_pos = start_pos + gymapi.Vec3(*vector)
                CYAN = (0, 1, 1)
                self._draw_debug_line_of_spheres(
                    env=self.envs[i],
                    start_pos=start_pos,
                    end_pos=end_pos,
                    color=CYAN,
                )

            for j in range(self.num_keypoints):
                keypoint_pos_cpu = self.obj_keypoint_pos[:, j].cpu().numpy()
                goal_keypoint_pos_cpu = self.goal_keypoint_pos[:, j].cpu().numpy()
                keypoint_pos_fixed_size_cpu = (
                    self.obj_keypoint_pos_fixed_size[:, j].cpu().numpy()
                )
                goal_keypoint_pos_fixed_size_cpu = (
                    self.goal_keypoint_pos_fixed_size[:, j].cpu().numpy()
                )

                for i in range(self.num_envs):
                    keypoint_transform = gymapi.Transform()
                    keypoint_transform.p = gymapi.Vec3(*keypoint_pos_cpu[i])
                    gymutil.draw_lines(
                        sphere_geom,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        keypoint_transform,
                    )

                    goal_keypoint_transform = gymapi.Transform()
                    goal_keypoint_transform.p = gymapi.Vec3(*goal_keypoint_pos_cpu[i])
                    gymutil.draw_lines(
                        sphere_geom,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        goal_keypoint_transform,
                    )

                    keypoint_transform_fixed_size = gymapi.Transform()
                    keypoint_transform_fixed_size.p = gymapi.Vec3(
                        *keypoint_pos_fixed_size_cpu[i]
                    )
                    gymutil.draw_lines(
                        sphere_geom_black,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        keypoint_transform_fixed_size,
                    )

                    goal_keypoint_transform_fixed_size = gymapi.Transform()
                    goal_keypoint_transform_fixed_size.p = gymapi.Vec3(
                        *goal_keypoint_pos_fixed_size_cpu[i]
                    )
                    gymutil.draw_lines(
                        sphere_geom_black,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        goal_keypoint_transform_fixed_size,
                    )

            # Visualize object and goal pose
            for i in range(self.num_envs):
                object_transform = gymapi.Transform(
                    p=gymapi.Vec3(*self.object_pos[i]),
                    r=gymapi.Quat(*self.object_rot[i]),
                )
                goal_transform = gymapi.Transform(
                    p=gymapi.Vec3(*self.goal_pos[i]),
                    r=gymapi.Quat(*self.goal_rot[i]),
                )
                self._draw_transform(transform=object_transform, env_idx=i)
                self._draw_transform(transform=goal_transform, env_idx=i)

    def _init_obs_action_queue(self):
        obs_queue_length = self.cfg["env"]["obsDelayMax"]
        action_queue_length = self.cfg["env"]["actionDelayMax"]
        object_state_queue_length = self.cfg["env"]["objectStateDelayMax"]
        self.obs_queue = torch.zeros(
            self.num_envs,
            obs_queue_length,
            self.num_observations,
            dtype=torch.float,
            device=self.device,
        )
        self.action_queue = torch.zeros(
            self.num_envs,
            action_queue_length,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
        )

        # Along QUEUE_LENGTH dimension, index=0 is the most recent observation/action
        # At each step, we update the queue by shifting the queue down, then updating index=0 with the new observation/action
        # If use index=0 for all time, then there is no delay
        # But we can sample delay index to simulate stochastic delay
        # Need to be careful on reset, need to flush the queue with the first obs/action

        # The object state queue is similar, but we want to have more delay on object state than the rest of the observations because it is
        # More noise and more latency/delay with FoundationPose and more likely to be incorrect in the real world
        # So we want to have more delay on object state than the rest of the observations
        # These will store the object state WITHOUT any noise
        # If we want to add noise to these, we will add noise on top of it after the object state is sampled from the queue
        self.object_state_queue = torch.zeros(
            self.num_envs,
            object_state_queue_length,
            13,
            dtype=torch.float,
            device=self.device,
        )

    def _init_tyler_curriculum(self):
        self._tyler_curriculum_scale = 0.0
        self._last_tyler_curriculum_update = time.time()
        if "init_tyler_curriculum_scale" in self.cfg["env"]:
            print(
                f"Initializing _tyler_curriculum_scale to {self.cfg['env']['init_tyler_curriculum_scale']}"
            )
            self._tyler_curriculum_scale = self.cfg["env"][
                "init_tyler_curriculum_scale"
            ]

    def _update_tyler_curriculum(self):
        # Vary _tyler_curriculum_scale from 0.0 to 1.0 over time
        # 0.0 means easy and 1.0 means hard

        # If gets at least 50% of max consecutive successes and been at least 5 minutes since last update, turn off extra obs more
        mean_successes = self.prev_episode_successes.mean().item()
        minutes_elapsed_since_last_update = (
            time.time() - self._last_tyler_curriculum_update
        ) / 60
        success_ratio = mean_successes / self.max_consecutive_successes
        curriculum_success_ratio = self.cfg["env"]["curriculumSuccessRatio"]
        doing_well = success_ratio > curriculum_success_ratio

        time_to_update = self.cfg["env"]["timeToUpdateTylerCurriculum"]
        update_step_size = self.cfg["env"]["updateStepSizeTylerCurriculum"]

        enough_time_since_last_update = (
            minutes_elapsed_since_last_update > time_to_update
        )
        if doing_well and enough_time_since_last_update:
            self._tyler_curriculum_scale += update_step_size
            if self._tyler_curriculum_scale > 1.0:
                self._tyler_curriculum_scale = 1.0
            self._last_tyler_curriculum_update = time.time()

        self.extras["tyler_curriculum_scale"] = self._tyler_curriculum_scale
        self.extras["mean_successes"] = mean_successes
        self.extras["mean_success_ratio"] = (
            mean_successes / self.max_consecutive_successes
        )
        self.extras["minutes_elapsed_since_last_update"] = (
            minutes_elapsed_since_last_update
        )

    def _initialize_camera_sensor(self, cam_pos, cam_target) -> None:
        self.camera_properties = gymapi.CameraProperties()
        RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE = 4
        self.camera_properties.width = int(
            self.camera_properties.width / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )
        self.camera_properties.height = int(
            self.camera_properties.height / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )

        self.camera_handles = []
        self.camera_env_indices = []
        self.camera_object_labels = []
        cycle_video_objects = bool(self.cfg["env"].get("cycleVideoObjects", False))
        if (
            cycle_video_objects
            and hasattr(self, "object_asset_indices")
            and hasattr(self, "object_asset_labels")
        ):
            for asset_idx, label in enumerate(self.object_asset_labels):
                env_ids = (self.object_asset_indices == asset_idx).nonzero(
                    as_tuple=False
                )
                if len(env_ids) == 0:
                    continue
                self.camera_env_indices.append(int(env_ids[0].item()))
                self.camera_object_labels.append(str(label))

        if len(self.camera_env_indices) == 0:
            self.camera_env_indices = [self.index_to_view]
            self.camera_object_labels = ["env_0"]

        for camera_env_idx in self.camera_env_indices:
            camera_handle = self.gym.create_camera_sensor(
                self.envs[camera_env_idx],
                self.camera_properties,
            )
            self.gym.set_camera_location(
                camera_handle, self.envs[camera_env_idx], cam_pos, cam_target
            )
            self.camera_handles.append(camera_handle)

        self.video_camera_idx = int(self.cfg["env"].get("videoCameraStartIdx", 0))
        self.video_camera_stride = max(
            1, int(self.cfg["env"].get("videoCameraStride", 1))
        )
        self.camera_handle = self.camera_handles[0]
        self.index_to_view = self.camera_env_indices[0]
        self.current_video_object_label = self.camera_object_labels[0]

        # self.video_frames is important for understanding the state of video recording
        #   Case 1: self.video_frames is None:
        #     * This means that we are not recording video
        #   Case 2: self.video_frames = []
        #     * This means that we should start recording video
        #     * BUT, we want our videos to start at the first frame of an episode
        #     * So, we are waiting for this
        #   Case 3: self.video_frames = [np.array(frame) for frame in ...]
        #     * These are image frames that will be assembled into a video when enough frames are capture
        self.video_frames: Optional[List[np.ndarray]] = None

    def _select_next_video_camera(self) -> None:
        if not hasattr(self, "camera_handles") or len(self.camera_handles) == 0:
            return
        camera_idx = self.video_camera_idx % len(self.camera_handles)
        self.camera_handle = self.camera_handles[camera_idx]
        self.index_to_view = self.camera_env_indices[camera_idx]
        self.current_video_object_label = self.camera_object_labels[camera_idx]
        self.video_camera_idx += self.video_camera_stride

    def _modify_render_settings_if_headless(self) -> None:
        # If not headless, leave things as they are
        if self.viewer is not None:
            return

        # If headless, we should default to having self.enable_viewer_sync=False to speed up env stepping
        self.enable_viewer_sync = False

    def _capture_video_if_needed(self) -> None:
        # If capture_video is False, we don't need to capture video
        if not self.cfg["env"]["capture_video"]:
            return

        # If enableCameraSensors is False, we can't capture video
        assert self.cfg["env"]["enableCameraSensors"], (
            "capture_video is only supported if enableCameraSensors is True"
        )

        should_start_video_capture_at_start_of_next_episode = (
            self.video_frames is None
            and self.control_steps % self.cfg["env"]["capture_video_freq"] == 0
            # and (self.control_steps > 0)  # Don't record video on first step
        )
        if should_start_video_capture_at_start_of_next_episode:
            self._select_next_video_camera()
            print("-" * 80)
            print(
                f"At self.control_steps = {self.control_steps}, should start video capture at start of next episode "
                f"for env {self.index_to_view} ({self.current_video_object_label})"
            )
            print("-" * 80)
            self.video_frames = []
            return

        should_start_video_capture_now = (
            self.video_frames is not None
            and len(self.video_frames) == 0
            # and self.progress_buf[self.index_to_view].item() <= 1  # Only start video capture on first step of episode so that videos don't start in the middle of an episode
            # Actually doesn't work because progress_buf gets reset to 0 not only at start of episode but on success
            and self.reset_buf[self.index_to_view].item()
            == 1  # Only start video capture after reset of an env
        )
        video_capture_in_progress = (
            self.video_frames is not None and len(self.video_frames) > 0
        )
        if should_start_video_capture_now or video_capture_in_progress:
            self._capture_video(video_capture_in_progress)

    def _capture_video(self, video_capture_in_progress: bool) -> None:
        assert self.video_frames is not None
        if not video_capture_in_progress:
            print("-" * 80)
            print(
                f"Starting to capture video frames for env {self.index_to_view} "
                f"({self.current_video_object_label})..."
            )
            print("-" * 80)
            # Video capture requires that self.enable_viewer_sync=True
            # If there is a viewer, we need to save the previous value of self.enable_viewer_sync so we can restore it later
            if self.viewer is not None:
                self.enable_viewer_sync_before = self.enable_viewer_sync
            # If there is no viewer, we always want self.enable_viewer_sync=False to speed up env stepping
            else:
                self.enable_viewer_sync_before = False

        # Store image
        self.enable_viewer_sync = True
        self.gym.render_all_camera_sensors(self.sim)
        color_image = self.gym.get_camera_image(
            self.sim,
            self.envs[self.index_to_view],
            self.camera_handle,
            gymapi.IMAGE_COLOR,
        )
        if color_image.size == 0:
            print(
                f"Warning: color_image is empty on {self.control_steps}th step, make sure you have this change to vec_task.py"
            )
            print(
                "https://github.com/tylerlum/human2sim2robot/blob/a5fd55baf83fbd04c585e2d596967ba08a38d540/human2sim2robot/sim_training/tasks/base/vec_task.py#L544"
            )
            return
        NUM_RGBA = 4
        color_image = color_image.reshape(
            self.camera_properties.height, self.camera_properties.width, NUM_RGBA
        )
        self.video_frames.append(color_image)

        if len(self.video_frames) == self.cfg["env"]["capture_video_len"]:
            object_label = str(getattr(self, "current_video_object_label", "env_0"))
            object_label = "".join(
                char if char.isalnum() or char in {"_", "-"} else "_"
                for char in object_label
            )
            video_filename = (
                f"{DATETIME_STR}_video_{self.control_steps}_{object_label}.mp4"
            )
            videos_dir = Path("videos")
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = videos_dir / video_filename
            print("-" * 80)
            print(f"Saving video to {video_path} ...")

            if not self.enable_viewer_sync_before:
                self.video_frames.pop(0)  # Remove first frame because it was not synced

            import imageio
            import wandb

            imageio.mimsave(
                video_path, self.video_frames, fps=int(1.0 / self.control_dt)
            )
            if wandb.run is not None:
                wandb_video = wandb.Video(
                    str(video_path), fps=int(1.0 / self.control_dt)
                )
                wandb.log({"video": wandb_video})
                # self.wandb_dict["video"] = wandb.Video(
                #     str(video_path), fps=int(1.0 / self.control_dt)
                # )
            print("DONE")
            print("-" * 80)

            # Reset variables
            self.video_frames = None
            self.enable_viewer_sync = self.enable_viewer_sync_before

    def _draw_transform(
        self, transform: gymapi.Transform, line_length: float = 0.2, env_idx: int = 0
    ) -> None:
        env = self.envs[env_idx]

        origin = transform.transform_point(gymapi.Vec3(0, 0, 0))
        x_dir = transform.transform_point(gymapi.Vec3(line_length, 0, 0))
        y_dir = transform.transform_point(gymapi.Vec3(0, line_length, 0))
        z_dir = transform.transform_point(gymapi.Vec3(0, 0, line_length))

        RED = (1, 0, 0)
        GREEN = (0, 1, 0)
        BLUE = (0, 0, 1)

        for color, dir in zip([RED, GREEN, BLUE], [x_dir, y_dir, z_dir]):
            gymutil.draw_line(
                p1=origin,
                p2=dir,
                color=gymapi.Vec3(*color),  # type: ignore
                gym=self.gym,
                viewer=self.viewer,
                env=env,
            )
            self._draw_debug_line_of_spheres(
                env=env,
                start_pos=origin,
                end_pos=dir,
                color=color,
            )

    def _draw_debug_line_of_spheres(
        self,
        env,
        start_pos: gymapi.Vec3,
        end_pos: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = 0.01,
        num_spheres: int = 10,
    ) -> None:
        for i in range(num_spheres):
            fraction = (i + 1) / (num_spheres + 1)
            pos = start_pos + ((end_pos - start_pos) * fraction)
            self._draw_debug_sphere(
                env=env,
                position=pos,
                color=color,
                radius=radius,
            )

    def _draw_debug_sphere(
        self,
        env,
        position: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = 0.005,
        num_lats: int = 10,
        num_lons: int = 10,
    ) -> None:
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius, num_lats, num_lons, color=color
        )
        gymutil.draw_lines(
            sphere_geom, self.gym, self.viewer, env, gymapi.Transform(p=position)
        )

    def accumulate_env_states(self):
        root_state_tensor = self.root_state_tensor.reshape(
            [self.num_envs, -1, *self.root_state_tensor.shape[1:]]
        ).clone()
        dof_state = self.dof_state.reshape(
            [self.num_envs, -1, *self.dof_state.shape[1:]]
        ).clone()

        for env_idx in range(self.num_envs):
            env_root_state_tensor = root_state_tensor[env_idx]
            self.episode_root_state_tensors[env_idx].append(env_root_state_tensor)

            env_dof_state = dof_state[env_idx]
            self.episode_dof_states[env_idx].append(env_dof_state)

    def dump_env_states(self, env_ids):
        def write_tensor_to_bin_stream(tensor, stream):
            bin_buff = io.BytesIO()
            torch.save(tensor, bin_buff)
            bin_buff = bin_buff.getbuffer()
            stream.write(int(len(bin_buff)).to_bytes(4, "big"))
            stream.write(bin_buff)

        with open(self.save_states_filename, "ab") as save_states_file:
            bin_stream = io.BytesIO()

            for env_idx in env_ids:
                ep_len = len(self.episode_root_state_tensors[env_idx])
                if ep_len <= 20:
                    continue

                states_to_save = min(ep_len // 10, 50)
                state_indices = random.sample(range(ep_len), states_to_save)

                print(f"Adding {states_to_save} states {state_indices}")
                bin_stream.write(int(states_to_save).to_bytes(4, "big"))

                root_states = [
                    self.episode_root_state_tensors[env_idx][si] for si in state_indices
                ]
                dof_states = [
                    self.episode_dof_states[env_idx][si] for si in state_indices
                ]

                root_states = torch.stack(root_states)
                dof_states = torch.stack(dof_states)

                write_tensor_to_bin_stream(root_states, bin_stream)
                write_tensor_to_bin_stream(dof_states, bin_stream)

                self.episode_root_state_tensors[env_idx] = []
                self.episode_dof_states[env_idx] = []

            bin_data = bin_stream.getbuffer()
            if bin_data.nbytes > 0:
                print(f"Writing {len(bin_data)} to file {self.save_states_filename}")
                save_states_file.write(bin_data)

    def load_initial_states(self):
        loaded_root_states = []
        loaded_dof_states = []

        with open(self.load_states_filename, "rb") as states_file:

            def read_nbytes(n_):
                res = states_file.read(n_)
                if len(res) < n_:
                    raise RuntimeError(
                        f"Could not read {n_} bytes from the binary file. Perhaps reached the end of file"
                    )
                return res

            while True:
                try:
                    num_states = int.from_bytes(read_nbytes(4), byteorder="big")
                    print(f"num_states_chunk {num_states}")

                    root_states_len = int.from_bytes(read_nbytes(4), byteorder="big")
                    print(f"root tensors len {root_states_len}")
                    root_states_bytes = read_nbytes(root_states_len)

                    dof_states_len = int.from_bytes(read_nbytes(4), byteorder="big")
                    print(f"dof_states_len {dof_states_len}")
                    dof_states_bytes = read_nbytes(dof_states_len)

                except Exception as exc:
                    print(exc)
                    break
                finally:
                    # parse binary buffers
                    def parse_tensors(bin_data):
                        with io.BytesIO(bin_data) as buffer:
                            tensors = torch.load(buffer)
                            return tensors

                    root_state_tensors = parse_tensors(root_states_bytes)
                    dof_state_tensors = parse_tensors(dof_states_bytes)
                    loaded_root_states.append(root_state_tensors)
                    loaded_dof_states.append(dof_state_tensors)

        self.initial_root_state_tensors = torch.cat(loaded_root_states)
        self.initial_dof_state_tensors = torch.cat(loaded_dof_states)
        assert (
            self.initial_dof_state_tensors.shape[0]
            == self.initial_root_state_tensors.shape[0]
        )
        self.num_initial_states = len(self.initial_root_state_tensors)

        print(
            f"{self.num_initial_states} states loaded from file {self.load_states_filename}!"
        )

    def set_robot_asset_rigid_shape_properties(
        self,
        robot_asset: gymapi.Asset,
        friction: Optional[float],
        fingertip_friction: Optional[float],
    ):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(robot_asset),
        )

        # Different friction for normal links (low friction) and fingertips (high friction)
        for i in range(len(rigid_shape_props)):
            if friction is not None:
                rigid_shape_props[i].friction = friction

        # Rigid bodies (links) are not the same as rigid shapes (collision geometries)
        # Each rigid body can have >=1 rigid shapes
        rb_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print(f"rb_names = {rb_names}")
        rb_shape_indices = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        assert_equals(len(rb_names), len(rb_shape_indices))
        rb_name_to_shape_indices = {
            name: (x.start, x.count) for name, x in zip(rb_names, rb_shape_indices)
        }
        fingertip_names = self.fingertips
        for name in fingertip_names:
            start, count = rb_name_to_shape_indices[name]
            for i in range(start, start + count):
                if fingertip_friction is not None:
                    rigid_shape_props[i].friction = fingertip_friction

        if not self.use_sharpa:
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            return

        # Turn off self-collisions for adjacent links
        from isaacgymenvs.tasks.simtoolreal.adjacent_links import (
            LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS,
            RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS,
        )

        if self.use_sharpa:
            if self.use_right_sharpa:
                link_to_adjacent_links = RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS
            elif self.use_left_sharpa:
                link_to_adjacent_links = LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS
            else:
                raise ValueError(f"Invalid use_sharpa: {self.use_sharpa}")

        assert set(link_to_adjacent_links.keys()).issubset(rb_names), (
            f"Some links are not in the asset {robot_asset}, rb_names: {rb_names}, link_to_adjacent_links: {link_to_adjacent_links}, only in link_to_adjacent_links: {set(link_to_adjacent_links.keys()) - set(rb_names)}, only in rb_names: {set(rb_names) - set(link_to_adjacent_links.keys())}"
        )
        assert set(sum(link_to_adjacent_links.values(), [])).issubset(rb_names), (
            f"Some links are not in the asset {robot_asset}, rb_names: {rb_names}, link_to_adjacent_links: {link_to_adjacent_links}, only in link_to_adjacent_links: {set(sum(link_to_adjacent_links.values(), [])) - set(rb_names)}, only in rb_names: {set(rb_names) - set(sum(link_to_adjacent_links.values(), []))}"
        )

        no_collision_pairs = set()
        for link, adjacent_links in link_to_adjacent_links.items():
            for adjacent_link in adjacent_links:
                no_collision_pairs.add(tuple(sorted((link, adjacent_link))))
        no_collision_pairs = sorted(list(no_collision_pairs))

        # Set collision_filters
        # collision if (filterA & filterB) == 0
        # Assign unique bit per link (up to 32 bits)
        link_bitmask = {name: 1 << i for i, name in enumerate(rb_names)}

        # For each no-collision pair, share bits so they don't collide
        for a, b in no_collision_pairs:
            bit_a = link_bitmask[a]
            bit_b = link_bitmask[b]
            # Add b's bit to a, and a's bit to b → ensures (filterA & filterB) != 0
            link_bitmask[a] |= bit_b
            link_bitmask[b] |= bit_a

        # Update filters on shapes again
        for name, (start, count) in rb_name_to_shape_indices.items():
            bit = link_bitmask[name]
            for i in range(start, start + count):
                rigid_shape_props[i].filter = bit

        self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

    def set_table_asset_rigid_shape_properties(
        self, table_asset: gymapi.Asset, friction: float
    ):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(table_asset),
        )
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = friction
        self.gym.set_asset_rigid_shape_properties(table_asset, rigid_shape_props)

    def set_object_asset_rigid_shape_properties(
        self, object_asset: gymapi.Asset, friction: float
    ):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(object_asset),
        )
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = friction
        self.gym.set_asset_rigid_shape_properties(object_asset, rigid_shape_props)

    def set_object_masses_and_inertias(
        self,
        envs: List[gymapi.Env],
        objects: List[int],
        masses: List[float],
        inertias: List[Tuple[float, float, float]],
    ):
        for env, object, mass, inertia in zip(envs, objects, masses, inertias):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            OBJECT_NUM_RIGID_BODIES = 1
            assert_equals(len(object_rb_props), OBJECT_NUM_RIGID_BODIES)
            for i in range(OBJECT_NUM_RIGID_BODIES):
                object_rb_props[i].mass = mass
                object_rb_props[i].inertia.x.x = inertia[0]
                object_rb_props[i].inertia.y.y = inertia[1]
                object_rb_props[i].inertia.z.z = inertia[2]
            self.gym.set_actor_rigid_body_properties(env, object, object_rb_props)

    @property
    def VISUALIZE_PD_TARGET_AS_BLUE_ROBOT(self) -> bool:
        if "VISUALIZE_PD_TARGET_AS_BLUE_ROBOT" in self.cfg["env"]:
            return self.cfg["env"]["VISUALIZE_PD_TARGET_AS_BLUE_ROBOT"]
        return False

    def sample_random_unit_axis(self, shape) -> Tensor:
        v = torch_rand_float(0.0, 1.0, shape, device=self.device)
        v = v / torch.norm(v, dim=-1, keepdim=True)
        return v

    def sample_delta_quat_xyzw(
        self, input_quat_xyzw: Tensor, delta_rotation_degrees: float
    ) -> Tensor:
        N, D = input_quat_xyzw.shape
        assert D == 4, (
            f"input_quat_xyzw.shape: {input_quat_xyzw.shape}, expected: (N, 4)"
        )

        quat_wxyz = torch.cat((input_quat_xyzw[:, 3:], input_quat_xyzw[:, :3]), dim=1)
        quat_matrix = quaternion_to_matrix(quat_wxyz)

        delta_rotation_radians = delta_rotation_degrees * np.pi / 180.0
        random_direction = self.sample_random_unit_axis(shape=(N, 3))
        sampled_rotation_magnitude = torch_rand_float(
            -delta_rotation_radians, delta_rotation_radians, (N, 1), device=self.device
        )
        sampled_rotation_axis_angles = random_direction * sampled_rotation_magnitude
        sampled_rotation_matrix = axis_angle_to_matrix(sampled_rotation_axis_angles)
        new_matrix = quat_matrix @ sampled_rotation_matrix

        new_quat_wxyz = matrix_to_quaternion(new_matrix)
        new_quat_xyzw = torch.cat(
            (new_quat_wxyz[:, 1:], new_quat_wxyz[:, 0:1]), dim=1
        ).clone()
        assert new_quat_xyzw.shape == (N, 4), (
            f"new_quat_xyzw.shape: {new_quat_xyzw.shape}, expected: (N, 4)"
        )
        return new_quat_xyzw
