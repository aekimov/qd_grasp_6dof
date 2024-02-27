import pdb

from pathlib import Path

from environments.src.robot_grasping import RobotGrasping
import environments
import environments.src.env_constants as env_consts
import environments.src.robots.panda_2f_consts as p2f_consts


"""
    # ---------------------------------------------------------------------------------------- #
    #                 FRANKA EMIKA PANDA STANDARD 2-FINGERS PARALLEL GRIPPER
    # ---------------------------------------------------------------------------------------- #
"""


def init_urdf_franka_emika_panda():

    root_3d_models_robots = \
        Path(environments.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

    urdf = Path(root_3d_models_robots / p2f_consts.PANDA_2_FINGERS_GRIP_RELATIVE_PATH_URDF)
    return str(urdf)


class FrankaEmikaPanda2Fingers(RobotGrasping):

    def __init__(self, **kwargs):

        urdf = init_urdf_franka_emika_panda()

        list_id_gripper_fingers = p2f_consts.LIST_ID_GRIPPER_FINGERS
        list_id_gripper_fingers_actuated = p2f_consts.LIST_ID_GRIPPER_FINGERS_ACTUATORS
        gripper_6dof_infos = p2f_consts.GRIPPER_6DOF_INFOS
        gripper_parameters = p2f_consts.GRIPPER_PARAMETERS
        gripper_default_joint_states = p2f_consts.DEFAULT_JOINT_STATES

        max_standoff_gripper = p2f_consts.MAX_HAND_STANDOFF_PANDA
        wrist_palm_offset_gripper = p2f_consts.WRIST_PALM_OFFSET
        half_palm_depth_offset_gripper = p2f_consts.HALF_PALM_DEPTH

        pose_relative_to_contact_point_d_min = p2f_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MIN
        pose_relative_to_contact_point_d_max = p2f_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MAX

        super().__init__(
            robot_urdf_path=urdf,
            list_id_gripper_fingers=list_id_gripper_fingers,
            list_id_gripper_fingers_actuated=list_id_gripper_fingers_actuated,
            gripper_6dof_infos=gripper_6dof_infos,
            gripper_parameters=gripper_parameters,
            gripper_default_joint_states=gripper_default_joint_states,
            max_standoff_gripper=max_standoff_gripper,
            wrist_palm_offset_gripper=wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper=half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min=pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max=pose_relative_to_contact_point_d_max,
            **kwargs,
        )

    def _cvt_genome2synergy_label(self, synergy_label, debug=False):
        raise RuntimeError('Synergies undefined for parallel grippers.')

    def _cvt_genome2init_joint_states(self, init_joint_state_genes):
        raise NotImplementedError('Undefined _cvt_genome2init_joint_states for the current gripper.')