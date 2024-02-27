
import sys
import argparse
import numpy as np
import seaborn as sns

from visualization.vis_tools import load_run_output_files, load_all_inds, get_folder_path_from_str, init_env, \
    load_all_fitnesses, get_all_gripper_6dof_pose_from_infos, load_all_infos


LINE_WIDTH_SCALE_FACTOR = 2
HEATMAP_COLOR_PALETTE = "coolwarm"

GRASP_XYZ_POSE_BIN_SIZE_FACTOR = 0.1
PURPLE_COLOR_RGBA = (162/255, 25/255, 255/255, 1)

DUMMY_QUARTERNION_ORIENT = [0, 0, 0, 1]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, required=True)
    parser.add_argument("-a", "--arrow-mode",
                        action="store_true",
                        help="Display 6-DoF poses as oriented arrows. False if not triggered: display voxels.")
    return parser.parse_args()


def generate_obj_id(env, info_shape_contact, color):
    base_collision_shape = env.bullet_client.createCollisionShape(**info_shape_contact)
    base_vis_shape = env.bullet_client.createVisualShape(**info_shape_contact, rgbaColor=color)
    return env.bullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=base_collision_shape,
        baseVisualShapeIndex=base_vis_shape,
        useMaximalCoordinates=False
    )


def get_max_theoretical_fitness(cfg):

    n_shake_all_joints = cfg['evaluate']['shaking_params']['n_shake']
    n_joint_perturbated = len(cfg['evaluate']['shaking_params']['perturbated_joint_ids'])
    n_fit_max_per_eval = n_shake_all_joints * n_joint_perturbated

    is_run_using_domain_randomization_fit = cfg['evaluate']['kwargs']['domain_randomization_fitness']
    if is_run_using_domain_randomization_fit:
        n_eval_domain_rand = cfg['evaluate']['n_domain_randomization_perturbations']
        assert n_eval_domain_rand is not None
        return n_fit_max_per_eval * n_eval_domain_rand

    return n_fit_max_per_eval


def get_arrow_tip_pose(contact_pos, end_eff_or_euler, arrow_side):
    theta_x, theta_y, theta_z = end_eff_or_euler

    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    rot_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1],
    ])
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1],
    ])

    contact_center_arrow_tip_pose = np.array([0, 0, arrow_side, 0])
    contact_center_arrow_tip_pose_right_side = np.array([0, arrow_side / 8, arrow_side - arrow_side / 4, 0])
    contact_center_arrow_tip_pose_left_side = np.array([0, - arrow_side / 8, arrow_side - arrow_side / 4, 0])

    rot_mat = np.matmul(np.matmul(rot_z, rot_y), rot_x)

    arrow_tip_pose = np.matmul(rot_mat, contact_center_arrow_tip_pose)
    arrow_tip_right_side_pose = np.matmul(rot_mat, contact_center_arrow_tip_pose_right_side)
    arrow_tip_left_side_pose = np.matmul(rot_mat, contact_center_arrow_tip_pose_left_side)

    end_arrow_point = np.array(contact_pos) + arrow_tip_pose[:3]
    end_arrow_right_side_point = np.array(contact_pos) + arrow_tip_right_side_pose[:3]
    end_arrow_left_side_point = np.array(contact_pos) + arrow_tip_left_side_pose[:3]

    return end_arrow_point, end_arrow_right_side_point, end_arrow_left_side_point


def cvt_fit2linewidth(fit, max_fit, min_fit):
    # fit -> [0, 1]
    return (max_fit - min_fit) * fit * LINE_WIDTH_SCALE_FACTOR


def draw_heat_arrow(
        contact_pos, fit, end_eff_or_euler, min_fit, max_fit, env, init_obj_xyzw, info_shape_contact, arrow_side
):

    # Sample color based on fit
    cmap = sns.color_palette(HEATMAP_COLOR_PALETTE, as_cmap=True)
    rgba = cmap(fit)
    contact_obj_color = list(rgba)
    i_alpha = 3
    contact_obj_color[i_alpha] = 1

    # Get arrow coordinates
    end_arrow_point, end_arrow_right_side_point, end_arrow_left_side_point = get_arrow_tip_pose(
        contact_pos=contact_pos,
        end_eff_or_euler=end_eff_or_euler,
        arrow_side=arrow_side
    )

    line_color = contact_obj_color[:3]
    line_width = cvt_fit2linewidth(fit=fit, max_fit=max_fit, min_fit=min_fit)

    env.bullet_client.addUserDebugLine(
        lineFromXYZ=contact_pos,
        lineToXYZ=end_arrow_point,
        lineColorRGB=line_color,
        lineWidth=line_width
    )
    env.bullet_client.addUserDebugLine(
        lineFromXYZ=end_arrow_right_side_point,
        lineToXYZ=end_arrow_point,
        lineColorRGB=line_color,
        lineWidth=line_width
    )
    env.bullet_client.addUserDebugLine(
        lineFromXYZ=end_arrow_left_side_point,
        lineToXYZ=end_arrow_point,
        lineColorRGB=line_color,
        lineWidth=line_width
    )

    draw_success_archive_cell = False
    if draw_success_archive_cell:
        contact_obj_id = generate_obj_id(env=env, info_shape_contact=info_shape_contact, color=contact_obj_color)
        env.bullet_client.resetBasePositionAndOrientation(contact_obj_id, contact_pos, init_obj_xyzw)


def get_outcome_archive_dims(run_infos):
    return {
        'len_bin_x': run_infos['outcome_archive_dims']['len_bin_x'],
        'len_bin_y': run_infos['outcome_archive_dims']['len_bin_y'],
        'len_bin_z': run_infos['outcome_archive_dims']['len_bin_z'],
    }


def get_outcome_archive_vis_dim(env, outcome_archive_dims):

    len_bin_x = outcome_archive_dims['len_bin_x']
    len_bin_y = outcome_archive_dims['len_bin_y']
    len_bin_z = outcome_archive_dims['len_bin_z']

    # Arrow dims
    arrow_side = len_bin_x

    # Contact point object
    co_dim_x, co_dim_y, co_dim_z = len_bin_x, len_bin_y, len_bin_z
    display_scale_factor = 4
    info_shape_contact = {"shapeType": env.bullet_client.GEOM_BOX,
                          "halfExtents": [
                              co_dim_x / display_scale_factor,
                              co_dim_y / display_scale_factor,
                              co_dim_z / display_scale_factor
                          ]}

    return info_shape_contact, arrow_side


def display_success_archive_3d_arrows_heatmap(env, all_6dof_pose_data, outcome_archive_dims):

    all_contact_points = [
        all_6dof_pose_data[scs_ind_id]['6dof_pose']['xyz'] for scs_ind_id in all_6dof_pose_data
    ]
    all_end_eff_or_euler = [
        env.bullet_client.getEulerFromQuaternion(all_6dof_pose_data[scs_ind_id]['6dof_pose']['quaternions'])
        for scs_ind_id in all_6dof_pose_data
    ]
    fitnesses = [all_6dof_pose_data[scs_ind_id]['fitness'] for scs_ind_id in all_6dof_pose_data]

    info_shape_contact, arrow_side = get_outcome_archive_vis_dim(env, outcome_archive_dims)

    init_obj_pos, init_obj_xyzw = env.bullet_client.getBasePositionAndOrientation(env.obj_id)
    min_fit, max_fit = np.quantile(fitnesses, q=0.2), np.quantile(fitnesses, q=0.8)

    env.bullet_client.configureDebugVisualizer(env.bullet_client.COV_ENABLE_GUI, 0)
    env.bullet_client.resetDebugVisualizerCamera(0.5, 30, -30, [0, 0, 0])

    for contact_pos, fit, end_eff_or_euler in zip(all_contact_points, fitnesses, all_end_eff_or_euler):

        draw_heat_arrow(
            contact_pos=contact_pos,
            fit=fit,
            end_eff_or_euler=end_eff_or_euler,
            min_fit=min_fit,
            max_fit=max_fit,
            env=env,
            init_obj_xyzw=init_obj_xyzw,
            info_shape_contact=info_shape_contact,
            arrow_side=arrow_side
        )

    input(f"{len(all_contact_points)} individuals successfully plotted. Press enter to close...")
    env.close()


def display_success_archive_voxels_heatmap(env, all_6dof_pose_data, outcome_archive_dims, cfg):

    all_contact_points = [
        all_6dof_pose_data[scs_ind_id]['6dof_pose']['xyz'] for scs_ind_id in all_6dof_pose_data
    ]
    all_end_eff_or_euler = [
        env.bullet_client.getEulerFromQuaternion(all_6dof_pose_data[scs_ind_id]['6dof_pose']['quaternions'])
        for scs_ind_id in all_6dof_pose_data
    ]
    fitnesses = [all_6dof_pose_data[scs_ind_id]['fitness'] for scs_ind_id in all_6dof_pose_data]

    max_theoretical_fitess = get_max_theoretical_fitness(cfg)
    info_shape_grasp_pose = get_info_shape_grasp_pose(env=env, outcome_archive_dims=outcome_archive_dims)

    env.bullet_client.configureDebugVisualizer(env.bullet_client.COV_ENABLE_GUI, 0)
    env.bullet_client.resetDebugVisualizerCamera(0.5, 30, -30, [0, 0, 0])

    for contact_pos, fit, end_eff_or_euler in zip(all_contact_points, fitnesses, all_end_eff_or_euler):
        draw_grasp_xyz_pose(
            grasp_xyz_pose=contact_pos,
            env=env,
            info_shape_grasp_pose=info_shape_grasp_pose,
            fitness=fit,
            max_theoretical_fitess=max_theoretical_fitess
        )

    input(f"{len(all_contact_points)} individuals successfully plotted. Press enter to close...")
    env.close()


def get_all_6dof_pose_data(folder):
    individuals = load_all_inds(folder)
    fitnesses = load_all_fitnesses(folder)
    infos, info_keys = load_all_infos(folder)

    all_gripper_6dof_pose = get_all_gripper_6dof_pose_from_infos(info_keys=info_keys, infos=infos)

    all_6dof_pose_data = {
        scs_ind_id: {
            '6dof_pose': gripper_6dof_pose,
            'fitness': fitnesses[scs_ind_id],
            'infos': infos[scs_ind_id, :],
            'info_keys': info_keys,
        }
        for scs_ind_id, (ind, gripper_6dof_pose) in enumerate(zip(individuals, all_gripper_6dof_pose))
    }

    return all_6dof_pose_data


def get_info_shape_grasp_pose(env, outcome_archive_dims):
    co_dim_x = outcome_archive_dims['len_bin_x']
    co_dim_y = outcome_archive_dims['len_bin_y']
    co_dim_z = outcome_archive_dims['len_bin_z']

    display_scale_factor = GRASP_XYZ_POSE_BIN_SIZE_FACTOR
    info_shape_grasp_pose = {
        "shapeType": env.bullet_client.GEOM_BOX,
        "halfExtents": [
            co_dim_x * display_scale_factor,
            co_dim_y * display_scale_factor,
            co_dim_z * display_scale_factor
        ]
    }
    return info_shape_grasp_pose


def draw_grasp_xyz_pose(
        grasp_xyz_pose, env, info_shape_grasp_pose, fitness, max_theoretical_fitess
):
    normalized_fitness = fitness / max_theoretical_fitess
    cmap = sns.color_palette(HEATMAP_COLOR_PALETTE, as_cmap=True)
    rgba = cmap(normalized_fitness)  # [1, 1, 1, 1] #[162/255, 25/255, 255/255] #cmap(0.5)

    relative_opacity_flg = False
    if relative_opacity_flg:
        non_null_opacity_offset = 0.01  # fit=0 => slightly visible
        normalized_opacity = max(min(normalized_fitness, 1.0), non_null_opacity_offset)
        i_opacity = 3
        rgba = list(rgba)
        rgba[i_opacity] = normalized_opacity


    grasp_xyz_obj_color = list(rgba)
    grasp_xyz_obj_id = generate_obj_id(env=env, info_shape_contact=info_shape_grasp_pose, color=grasp_xyz_obj_color)
    env.bullet_client.resetBasePositionAndOrientation(grasp_xyz_obj_id, grasp_xyz_pose, DUMMY_QUARTERNION_ORIENT)


def replay_trajs(run_folder_path, arrow_mode):

    # Load run folder
    folder = get_folder_path_from_str(run_folder_path)

    # Extract corresponding data
    run_details, run_infos, cfg = load_run_output_files(folder)

    # Init grasp gym env
    env = init_env(cfg, display=True)
    env.bullet_client.removeBody(env.robot_id)

    individuals = load_all_inds(folder)
    outcome_archive_dims = get_outcome_archive_dims(run_infos)

    # Get the 6dof data to replay
    all_6dof_pose_data = get_all_6dof_pose_data(folder=folder)

    if arrow_mode:
        display_success_archive_3d_arrows_heatmap(
            env=env,
            all_6dof_pose_data=all_6dof_pose_data,
            outcome_archive_dims=outcome_archive_dims
        )
    else:
        display_success_archive_voxels_heatmap(
            env=env,
            all_6dof_pose_data=all_6dof_pose_data,
            outcome_archive_dims=outcome_archive_dims,
            cfg=cfg,
        )

    print('Ending.')


def get_plot_success_archive_heatmap_kwargs():
    args = arg_parser()
    return {'run_folder_path': args.runs, 'arrow_mode': args.arrow_mode}


def main():
    kwargs = get_plot_success_archive_heatmap_kwargs()
    replay_trajs(**kwargs)


if __name__ == "__main__":
    sys.exit(main())


