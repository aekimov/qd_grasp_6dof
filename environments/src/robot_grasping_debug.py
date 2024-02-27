
import environments.src.env_constants as env_consts

PATH_TO_DEBUG_SPHERE_GREEN = "environments/3d_models/debug_shape/green_sphere.urdf"
PATH_TO_DEBUG_SPHERE_BLUE = "environments/3d_models/debug_shape/sphere_blue.urdf"
PATH_TO_DEBUG_SPHERE_RED = "environments/3d_models/debug_shape/sphere_red.urdf"


def apply_all_gripper_shaking_debug(robot_grasp_env):
    grip_6dof_infos = robot_grasp_env.sim_engine.gripper_6dof_infos

    are_all_shakes_successful = True
    for i_grip_joint in env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']:
        gripper_joint_infos = grip_6dof_infos[i_grip_joint]
        is_being_grasped, n_shake_success = robot_grasp_env.apply_gripper_shaking(
            joint_index=i_grip_joint,
            gripper_joint_infos=gripper_joint_infos,
        )
        at_least_one_failure = are_all_shakes_successful and not is_being_grasped
        if at_least_one_failure:
            are_all_shakes_successful = False

        return are_all_shakes_successful


