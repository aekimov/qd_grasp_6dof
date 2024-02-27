import pdb

import numpy as np
from pathlib import Path

from utils.common_tools import project_from_to_value

from environments.src.robot_grasping import RobotGrasping
import environments
import environments.src.env_constants as env_consts
import environments.src.robots.shadow_hand_consts as sh_consts

import configs.qd_config as qd_cfg

"""
    # ---------------------------------------------------------------------------------------- #
    #                                   ALLEGRO DEXTEROUS HAND
    # ---------------------------------------------------------------------------------------- #
"""


def init_urdf_shadow_hand():
    root_3d_models_robots = \
        Path(environments.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

    urdf = Path(root_3d_models_robots / sh_consts.SHADOW_HAND_GRIP_RELATIVE_PATH_URDF)
    return str(urdf)


class ShadowHand(RobotGrasping):

    def __init__(self, **kwargs):
        urdf = init_urdf_shadow_hand()

        list_id_gripper_fingers = sh_consts.LIST_ID_GRIPPER_FINGERS
        list_id_gripper_fingers_actuated = sh_consts.LIST_ID_GRIPPER_FINGERS_ACTUATORS

        gripper_6dof_infos = sh_consts.GRIPPER_6DOF_INFOS
        gripper_parameters = sh_consts.GRIPPER_PARAMETERS

        gripper_default_joint_states = sh_consts.DEFAULT_JOINT_STATES

        max_standoff_gripper = sh_consts.MAX_HAND_STANDOFF_ALLEGRO
        wrist_palm_offset_gripper = sh_consts.WRIST_PALM_OFFSET
        half_palm_depth_offset_gripper = sh_consts.HALF_PALM_DEPTH

        pose_relative_to_contact_point_d_min = sh_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MIN
        pose_relative_to_contact_point_d_max = sh_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MAX

        super().__init__(
            robot_urdf_path=urdf,
            list_id_gripper_fingers=list_id_gripper_fingers,
            list_id_gripper_fingers_actuated=list_id_gripper_fingers_actuated,
            gripper_6dof_infos=gripper_6dof_infos,
            gripper_parameters=gripper_parameters,
            gripper_default_joint_states=gripper_default_joint_states,
            max_standoff_gripper=max_standoff_gripper,
            half_palm_depth_offset_gripper=half_palm_depth_offset_gripper,
            wrist_palm_offset_gripper=wrist_palm_offset_gripper,
            pose_relative_to_contact_point_d_min=pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max=pose_relative_to_contact_point_d_max,

            **kwargs,
        )

    def _is_gripper_closed(self, action):
        return action[self.i_action_grip_close] < 0

    def _get_gripper_command(self, action):
        action_grip_genome_val = action[self.i_action_grip_close]
        fingers_cmd = [action_grip_genome_val, action_grip_genome_val]
        return fingers_cmd

    def step(self, action):
        assert action is not None
        assert len(action) == self.n_actions

        # Update info
        self.info['closed gripper'] = self._is_gripper_closed(action)

        # Convert action to a gym-grasp compatible command
        gripper_command = self._get_gripper_command(action)
        arm_command = action[:self.i_action_grip_close]
        robot_command = np.hstack([arm_command, gripper_command])

        # Send the command to the robot
        return super().step(robot_command)

    def _set_robot_default_state(self):
        pdb.set_trace()
        for j_id, pos in sh_consts.DEFAULT_JOINT_STATES.items():
            self.bullet_client.resetJointState(self.robot_id, j_id, targetValue=pos)

    def _reset_robot(self):
        self._set_robot_default_state()

    def _cvt_genome2synergy_label(self, synergy_label, debug=False):
        synergy_label_projected = np.round(project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=sh_consts.FIXED_INTERVAL_SYNERGIES,
            x_start=synergy_label
        ))

        synergy_str = sh_consts.SYNERGIES_ID_TO_STR[synergy_label_projected]
        list_id_grip_fingers_actuated = sh_consts.SYNERGIES_STR_TO_J_ID_GRIP_FINGERS_ACTUATED[synergy_str]

        return list_id_grip_fingers_actuated

    def _cvt_genome2init_joint_states(self, init_joint_state_genes):
        raise NotImplementedError('Undefined _cvt_genome2init_joint_states for the current gripper.')
