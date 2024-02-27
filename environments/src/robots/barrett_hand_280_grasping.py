import pdb

import numpy as np
from pathlib import Path

from environments.src.robot_grasping import RobotGrasping
import environments
import environments.src.env_constants as env_consts
import environments.src.robots.barrett_hand_280_consts as bh280_consts

from utils.common_tools import project_from_to_value
import configs.qd_config as qd_cfg

"""
    # ---------------------------------------------------------------------------------------- #
    #                                 BARRETT 3-FINGERS HAND (280)
    # ---------------------------------------------------------------------------------------- #
"""


def init_urdf_bh280():
    root_3d_models_robots = \
        Path(environments.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

    urdf = Path(root_3d_models_robots / bh280_consts.BARRETT_HAND_280_GRIP_RELATIVE_PATH_URDF)
    return str(urdf)


class BarrettHand280(RobotGrasping):

    def __init__(self, **kwargs):
        urdf = init_urdf_bh280()

        list_id_gripper_fingers = bh280_consts.LIST_ID_GRIPPER_FINGERS
        list_id_gripper_fingers_actuated = bh280_consts.LIST_ID_GRIPPER_FINGERS_ACTUATORS

        gripper_6dof_infos = bh280_consts.GRIPPER_6DOF_INFOS
        gripper_parameters = bh280_consts.GRIPPER_PARAMETERS

        gripper_default_joint_states = bh280_consts.DEFAULT_JOINT_STATES

        max_standoff_gripper = bh280_consts.MAX_HAND_STANDOFF_BARRETT_HAND_280
        wrist_palm_offset_gripper = bh280_consts.WRIST_PALM_OFFSET
        half_palm_depth_offset_gripper = bh280_consts.HALF_PALM_DEPTH

        pose_relative_to_contact_point_d_min = bh280_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MIN
        pose_relative_to_contact_point_d_max = bh280_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MAX

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
        for j_id, pos in bh280_consts.DEFAULT_JOINT_STATES.items():
            self.bullet_client.resetJointState(self.robot_id, j_id, targetValue=pos)

    def _reset_robot(self):
        self._set_robot_default_state()

    def _cvt_genome2synergy_label(self, synergy_label, debug=False):
        raise RuntimeError('Synergies undefined for the 3-fingers barrett hand.')

    def _cvt_genome2init_joint_states(self, init_joint_state_genes):

        assert len(init_joint_state_genes) == bh280_consts.N_INIT_JOINT_STATE_GENES
        joint_ids_to_states = {}

        for i_init_joint_state_gene, init_joint_state_gene in enumerate(init_joint_state_genes):
            j_id = bh280_consts.J_ID_FROM_ORDERED_INIT_JOINT_STATE_GENES[i_init_joint_state_gene]

            init_joint_state = project_from_to_value(
                interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
                interval_to=[
                    bh280_consts.J_ID_INIT_JOINT_STATE_BOUNDARIES[j_id]['min'],
                    bh280_consts.J_ID_INIT_JOINT_STATE_BOUNDARIES[j_id]['max']
                ],
                x_start=init_joint_state_gene
            )

            joint_ids_to_states[j_id] = init_joint_state

        return joint_ids_to_states
