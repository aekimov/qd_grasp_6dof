
import numpy as np

from utils.common_tools import project_from_to_value

from algorithms.evaluation.grasp_strategies_routines import search_first_contact_point_clean, \
    get_normal_surface_point, get_gripper_pose_relatively_to_contact_point, \
    get_gripper_pose_relatively_to_contact_point_allegro_compatible, \
    search_opposite_contact, calculate_normal_from_surface, angle_between_2_vectors, \
    is_inside_antipodal_selectability_cone, entreaxe_selectability, entraxe_world_generation, compute_new_gripper_pose


import configs.qd_config as qd_cfg
import configs.eval_config as eval_cfg


EXTREMITY_WIRST = 0.112


def cvt_genome_to_6dof_pose_cartesian(robot_grasp_env, genome):
    gripper_pos_xyz_genome = genome[:3]
    gripper_orient_rpy_genome = genome[3:6]

    aabb_min = robot_grasp_env.search_space_bb.aabb_max
    aabb_max = robot_grasp_env.search_space_bb.aabb_min

    gripper_pos_xyz = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[aabb_min[i_val], aabb_max[i_val]],
            x_start=val
        ) for i_val, val in enumerate(gripper_pos_xyz_genome)
    ]
    gripper_orient_rpy = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=eval_cfg.FIXED_INTERVAL_EULER,
            x_start=val
        ) for val in gripper_orient_rpy_genome
    ]

    gripper_orient_quat = list(robot_grasp_env.bullet_client.getQuaternionFromEuler(gripper_orient_rpy))

    gripper_6dof_pose = {
        'xyz': gripper_pos_xyz,
        'euler_rpy': None,
        'quaternions': gripper_orient_quat,
    }
    return gripper_6dof_pose


def cvt_genome_to_6dof_pose_spherical(robot_grasp_env, genome):
    dist2obj_centroid_genome = genome[0]
    longitude_angle_genome = genome[1]
    colatitude_angle_genome = genome[2]

    dist2obj_centroid = project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[0, robot_grasp_env.sim_engine.search_space_bb_side],
            x_start=dist2obj_centroid_genome
        )
    longitude_angle = project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[0, 2*np.pi],
            x_start=longitude_angle_genome
        )
    colatitude_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=colatitude_angle_genome
    )

    alpha_genome = genome[3]
    beta_genome = genome[4]
    gamma_genome = genome[5]

    alpha = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=alpha_genome
    )
    beta = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=beta_genome
    )
    gamma = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=gamma_genome
    )
    gripper_orient_rpy = [alpha, beta, gamma]

    # Spherical to Cartesian conversion
    x_gripper = dist2obj_centroid * np.sin(colatitude_angle) * np.cos(longitude_angle)
    y_gripper = dist2obj_centroid * np.sin(colatitude_angle) * np.sin(longitude_angle)
    z_gripper = dist2obj_centroid * np.cos(colatitude_angle)
    gripper_pos_xyz = [x_gripper, y_gripper, z_gripper]

    gripper_orient_quat = list(robot_grasp_env.bullet_client.getQuaternionFromEuler(gripper_orient_rpy))

    gripper_6dof_pose = {
        'xyz': gripper_pos_xyz,
        'euler_rpy': None,
        'quaternions': gripper_orient_quat,
    }
    return gripper_6dof_pose


def cvt_genome_to_preset_6dof_pose_approach_strat(robot_grasp_env, genome):

    # Extract values from the genome
    dist2obj_centroid_genome = genome[0]
    longitude_angle_genome = genome[1]
    colatitude_angle_genome = genome[2]

    alpha_genome = genome[3]
    beta_genome = genome[4]
    gamma_genome = genome[5]

    nu_genome = genome[6]
    d_genome = genome[7]
    ksi_genome = genome[8]
    omega_genome = genome[9]

    # Compute the contact point finder position and orientation
    dist2obj_centroid = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, robot_grasp_env.sim_engine.search_space_bb_side],
        x_start=dist2obj_centroid_genome
    )
    longitude_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=longitude_angle_genome
    )
    colatitude_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=colatitude_angle_genome
    )
    x_gripper = dist2obj_centroid * np.sin(colatitude_angle) * np.cos(longitude_angle)
    y_gripper = dist2obj_centroid * np.sin(colatitude_angle) * np.sin(longitude_angle)
    z_gripper = dist2obj_centroid * np.cos(colatitude_angle)
    pos_robot_xyz = np.array([x_gripper, y_gripper, z_gripper])

    alpha = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=alpha_genome
    )
    beta = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=beta_genome
    )
    gamma = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=gamma_genome
    )
    gripper_orient_rpy = np.array([alpha, beta, gamma])

    # Compute the gripper position and orientation from the contact point
    nu_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi / 4],
        x_start=nu_genome
    )
    d_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, robot_grasp_env.max_standoff_gripper],
        x_start=d_genome
    )
    ksi_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=ksi_genome
    )
    omega_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=omega_genome
    )

    hand_pose_from_contact_params = {
        'nu': nu_projected,
        'd': d_projected,
        'ksi': ksi_projected,
        'omega': omega_projected,
    }

    return pos_robot_xyz, gripper_orient_rpy, hand_pose_from_contact_params


def cvt_genome_to_preset_6dof_pose_rand_sample_contact_strat_list(robot_grasp_env, genome):

    # Extract values from the genome
    contact_point_finder_list_id_gen = genome[0]
    nu_genome = genome[1]
    d_genome = genome[2]
    ksi_genome = genome[3]
    omega_genome = genome[4]

    contact_point_finder_list_id_gen, nu_genome, d_genome, ksi_genome, omega_genome = np.clip(
        [contact_point_finder_list_id_gen, nu_genome, d_genome, ksi_genome, omega_genome],
        a_min=-qd_cfg.GENOTYPE_MAX_VAL,
        a_max=qd_cfg.GENOTYPE_MAX_VAL,
    )

    # Get contact point on the object surface based on list index (no locality information)
    contact_point_finder_list_id = int(np.round(
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[0, len(robot_grasp_env.sim_engine.uniform_obj_contact_points) - 1],
            x_start=contact_point_finder_list_id_gen
        )
    ))
    contact_point_xyz = robot_grasp_env.sim_engine.uniform_obj_contact_points[contact_point_finder_list_id]

    # Compute the gripper position and orientation from the contact point
    nu_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],  # in rand_sample contact, the cone is not bounded to [0, pi/4]
        x_start=nu_genome
    )
    d_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, robot_grasp_env.max_standoff_gripper],
        x_start=d_genome
    )
    ksi_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=ksi_genome
    )
    omega_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=omega_genome
    )

    hand_pose_from_contact_params = {
        'nu': nu_projected,
        'd': d_projected,
        'ksi': ksi_projected,
        'omega': omega_projected,
    }

    return contact_point_xyz, hand_pose_from_contact_params


def cvt_genome_to_preset_6dof_pose_approach_strat_contact_point_finder(robot_grasp_env, genome):

    # Extract values from the genome
    contact_point_finder_xyz_pose_genome_values = genome[:3]
    nu_genome = genome[3]
    d_genome = genome[4]
    ksi_genome = genome[5]
    omega_genome = genome[6]
    # Get contact point finder voxel 3D pose
    contact_point_finder_xyz_pose = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[ss_min, ss_max],
            x_start=gen_val
        )
        for gen_val, ss_min, ss_max in zip(
            contact_point_finder_xyz_pose_genome_values,
            robot_grasp_env.search_space_bb_object.aabb_min,
            robot_grasp_env.search_space_bb_object.aabb_max
        )
    ]

    # Compute the gripper position and orientation from the contact point
    nu_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi / 4],
        x_start=nu_genome
    )

    d_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[robot_grasp_env.pose_relative_to_contact_point_d_min, robot_grasp_env.pose_relative_to_contact_point_d_max],
        x_start=d_genome
    )

    ksi_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=ksi_genome
    )
    omega_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=omega_genome
    )

    hand_pose_from_contact_params = {
        'nu': nu_projected,
        'd': d_projected,
        'ksi': ksi_projected,
        'omega': omega_projected,
    }

    return contact_point_finder_xyz_pose, hand_pose_from_contact_params


def cvt_genome_to_preset_6dof_pose_contact_strat_contact_point_finder(robot_grasp_env, genome):

    # Extract values from the genome
    contact_point_finder_xyz_pose_genome_values = genome[:3]
    nu_genome = genome[3]
    d_genome = genome[4]
    ksi_genome = genome[5]
    omega_genome = genome[6]

    # Get contact point finder voxel 3D pose
    contact_point_finder_xyz_pose = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[ss_min, ss_max],
            x_start=gen_val
        )
        for gen_val, ss_min, ss_max in zip(
            contact_point_finder_xyz_pose_genome_values,
            robot_grasp_env.search_space_bb_object.aabb_min,
            robot_grasp_env.search_space_bb_object.aabb_max
        )
    ]

    # Compute the gripper position and orientation from the contact point
    nu_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=nu_genome
    )

    d_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[robot_grasp_env.pose_relative_to_contact_point_d_min, robot_grasp_env.pose_relative_to_contact_point_d_max],
        x_start=d_genome
    )

    ksi_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=ksi_genome
    )
    omega_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=omega_genome
    )

    hand_pose_from_contact_params = {
        'nu': nu_projected,
        'd': d_projected,
        'ksi': ksi_projected,
        'omega': omega_projected,
    }

    return contact_point_finder_xyz_pose, hand_pose_from_contact_params


def cvt_genome_to_preset_6dof_pose_antipodal_strat(robot_grasp_env, genome):

    # Extract values from the genome
    dist2obj_centroid_genome = genome[0]
    longitude_angle_genome = genome[1]
    colatitude_angle_genome = genome[2]

    alpha_genome = genome[3]
    beta_genome = genome[4]
    gamma_genome = genome[5]

    tau_genome = genome[6]

    # Compute the contact point finder position and orientation

    dist2obj_centroid = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, robot_grasp_env.sim_engine.search_space_bb_side],
        x_start=dist2obj_centroid_genome
    )
    longitude_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=longitude_angle_genome
    )
    colatitude_angle = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=colatitude_angle_genome
    )
    x_gripper = dist2obj_centroid * np.sin(colatitude_angle) * np.cos(longitude_angle)
    y_gripper = dist2obj_centroid * np.sin(colatitude_angle) * np.sin(longitude_angle)
    z_gripper = dist2obj_centroid * np.cos(colatitude_angle)
    pos_robot_xyz = np.array([x_gripper, y_gripper, z_gripper])

    alpha = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=alpha_genome
    )
    beta = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=beta_genome
    )
    gamma = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=gamma_genome
    )
    gripper_orient_rpy = np.array([alpha, beta, gamma])

    # Compute the gripper position and orientation from the contact point
    tau_genome_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=tau_genome
    )

    hand_orientation_from_entraxe = {
        'tau': tau_genome_projected,
    }

    return pos_robot_xyz, gripper_orient_rpy, hand_orientation_from_entraxe


def cvt_genome_to_preset_6dof_pose_antipodal_strat_contact_point_finder(robot_grasp_env, genome):

    # Extract values from the genome
    contact_point_finder_xyz_pose_genome_values = genome[:3]
    tau_genome = genome[3]

    # Get contact point finder voxel 3D pose
    contact_point_finder_xyz_pose = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[ss_min, ss_max],
            x_start=gen_val
        )
        for gen_val, ss_min, ss_max in zip(
            contact_point_finder_xyz_pose_genome_values,
            robot_grasp_env.search_space_bb_object.aabb_min,
            robot_grasp_env.search_space_bb_object.aabb_max
        )
    ]

    # Compute the gripper position and orientation from the contact point
    tau_genome_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=tau_genome
    )

    hand_orientation_from_entraxe = {
        'tau': tau_genome_projected,
    }

    return contact_point_finder_xyz_pose, hand_orientation_from_entraxe


def get_6dof_pose_approach(
        robot_grasp_env,
        default_object_kwargs,
        default_gripper_pose_kwargs,
        debug=False,
):
    """Similar to get_6dof_pose_approach, without non-necessary displays."""

    object_id = default_object_kwargs['object_id']
    start_pos_robot_xyz = default_gripper_pose_kwargs['start_pos_robot_xyz']
    euler_robot = default_gripper_pose_kwargs['euler_robot']
    hand_pose_from_contact_params = default_gripper_pose_kwargs['hand_pose_from_contact_params']

    # ------------------------------------------------------------------------------------------------------------ #
    #                                              1) FIRST CONTACT POINT
    # ------------------------------------------------------------------------------------------------------------ #

    is_contact_point_found, contact_point, vector_director = search_first_contact_point_clean(
        bullet_client=robot_grasp_env.bullet_client,
        object_id=object_id,
        start_pos_robot_xyz=start_pos_robot_xyz,
        euler_robot=euler_robot,
    )
    if not is_contact_point_found:
        return None

    normal_at_contact_point = get_normal_surface_point(
        bullet_client=robot_grasp_env.bullet_client,
        list_of_points_for_each_triangle_object_mesh=robot_grasp_env.list_of_points_for_each_triangle_obj_mesh,
        object_normals_to_triangles=robot_grasp_env.object_normals_to_triangles,
        contact_point=contact_point,
        debug=debug
    )

    # ------------------------------------------------------------------------------------------------------------ #
    #                                           2) SET 6DOF POSE RELATIVELY
    # ------------------------------------------------------------------------------------------------------------ #

    gripper_6dof_pose = get_gripper_pose_relatively_to_contact_point(
        bullet_client=robot_grasp_env.bullet_client,
        hand_pose_from_contact_params=hand_pose_from_contact_params,
        normal_at_contact_point=normal_at_contact_point,
        contact_point=contact_point,
        wrist_palm_offset_gripper=robot_grasp_env.wrist_palm_offset_gripper,
        debug=debug
    )

    return gripper_6dof_pose


def cvt_genome_to_6dof_pose_approach_strat(robot_grasp_env, genome):
    start_pos_robot_xyz, euler_robot, hand_pose_from_contact_params = \
        cvt_genome_to_preset_6dof_pose_approach_strat(robot_grasp_env=robot_grasp_env, genome=genome)

    default_gripper_pose_kwargs = {
        'start_pos_robot_xyz': start_pos_robot_xyz,
        'euler_robot': euler_robot,
        'hand_pose_from_contact_params': hand_pose_from_contact_params,
    }
    default_object_kwargs = {
        'object_id': robot_grasp_env.obj_id,
        'path2obj_point_cloud': robot_grasp_env.path2obj_point_cloud,
    }

    approach_kwargs = {
        'robot_grasp_env': robot_grasp_env,
        'default_object_kwargs': default_object_kwargs,
        'default_gripper_pose_kwargs': default_gripper_pose_kwargs,
    }
    gripper_6dof_pose = get_6dof_pose_approach(**approach_kwargs)

    return gripper_6dof_pose


def cvt_genome_to_6dof_pose_rand_sample_contact_strategy_list_search(robot_grasp_env, genome):

    contact_point_xyz, hand_pose_from_contact_params = \
        cvt_genome_to_preset_6dof_pose_rand_sample_contact_strat_list(robot_grasp_env=robot_grasp_env, genome=genome)

    # Apply standard approach-based method
    normal_at_contact_point = get_normal_surface_point(
        bullet_client=robot_grasp_env.bullet_client,
        list_of_points_for_each_triangle_object_mesh=robot_grasp_env.list_of_points_for_each_triangle_obj_mesh,
        object_normals_to_triangles=robot_grasp_env.object_normals_to_triangles,
        contact_point=contact_point_xyz,
        debug=False
    )
    gripper_6dof_pose = get_gripper_pose_relatively_to_contact_point(
        bullet_client=robot_grasp_env.bullet_client,
        hand_pose_from_contact_params=hand_pose_from_contact_params,
        normal_at_contact_point=normal_at_contact_point,
        contact_point=contact_point_xyz,
        wrist_palm_offset_gripper=robot_grasp_env.wrist_palm_offset_gripper,
        debug=False
    )
    return gripper_6dof_pose


def cvt_genome_to_6dof_pose_contact_strategy_search(robot_grasp_env, genome, robot):

    contact_point_finder_xyz_pose, hand_pose_from_contact_params = \
        cvt_genome_to_preset_6dof_pose_contact_strat_contact_point_finder(robot_grasp_env=robot_grasp_env, genome=genome)

    # Get closer point on the object surface
    query = [contact_point_finder_xyz_pose]
    closest_contact_point_id = robot_grasp_env.sim_engine.k_tree_uniform_contact_points.kneighbors(X=query)[1][0][0]
    closest_contact_point = robot_grasp_env.sim_engine.uniform_obj_contact_points[closest_contact_point_id]

    # Apply standard approach-based method
    normal_at_contact_point = get_normal_surface_point(
        bullet_client=robot_grasp_env.bullet_client,
        list_of_points_for_each_triangle_object_mesh=robot_grasp_env.list_of_points_for_each_triangle_obj_mesh,
        object_normals_to_triangles=robot_grasp_env.object_normals_to_triangles,
        contact_point=closest_contact_point,
        debug=False
    )

    gripper_6dof_pose = get_gripper_pose_relatively_to_contact_point_allegro_compatible(
        bullet_client=robot_grasp_env.bullet_client,
        hand_pose_from_contact_params=hand_pose_from_contact_params,
        normal_at_contact_point=normal_at_contact_point,
        contact_point=closest_contact_point,
        wrist_palm_offset_gripper=robot_grasp_env.wrist_palm_offset_gripper,
        debug=False,
        robot=robot,
    )

    return gripper_6dof_pose


def cvt_genome_to_6dof_pose_approach_strat_contact_point_finder_search(robot_grasp_env, genome, robot):
    contact_point_finder_xyz_pose, hand_pose_from_contact_params = \
        cvt_genome_to_preset_6dof_pose_approach_strat_contact_point_finder(robot_grasp_env=robot_grasp_env, genome=genome)

    # Get closer point on the object surface
    query = [contact_point_finder_xyz_pose]
    closest_contact_point_id = robot_grasp_env.sim_engine.k_tree_uniform_contact_points.kneighbors(X=query)[1][0][0]
    closest_contact_point = robot_grasp_env.sim_engine.uniform_obj_contact_points[closest_contact_point_id]

    normal_at_contact_point = get_normal_surface_point(
        bullet_client=robot_grasp_env.bullet_client,
        list_of_points_for_each_triangle_object_mesh=robot_grasp_env.list_of_points_for_each_triangle_obj_mesh,
        object_normals_to_triangles=robot_grasp_env.object_normals_to_triangles,
        contact_point=closest_contact_point,
        debug=False
    )

    gripper_6dof_pose = get_gripper_pose_relatively_to_contact_point_allegro_compatible(
        bullet_client=robot_grasp_env.bullet_client,
        hand_pose_from_contact_params=hand_pose_from_contact_params,
        normal_at_contact_point=normal_at_contact_point,
        contact_point=closest_contact_point,
        wrist_palm_offset_gripper=robot_grasp_env.wrist_palm_offset_gripper,
        debug=False,
        robot=robot,
    )

    return gripper_6dof_pose


def cvt_genome_to_6dof_pose_antipodal_strat(robot_grasp_env, genome):
    start_pos_robot_xyz, euler_robot, hand_orientation_from_entraxe = \
        cvt_genome_to_preset_6dof_pose_antipodal_strat(robot_grasp_env=robot_grasp_env, genome=genome)

    default_gripper_pose_kwargs = {
        'start_pos_robot_xyz': start_pos_robot_xyz,
        'euler_robot': euler_robot,
        'hand_orientation_from_entraxe': hand_orientation_from_entraxe,
    }
    default_object_kwargs = {
        'object_id': robot_grasp_env.obj_id,
        'path2obj_point_cloud': robot_grasp_env.path2obj_point_cloud,
    }
    antipodal_kwargs = {
        'robot_grasp_env': robot_grasp_env,
        'default_object_kwargs': default_object_kwargs,
        'default_gripper_pose_kwargs': default_gripper_pose_kwargs,
    }
    gripper_6dof_pose = get_6dof_pose_antipodal(**antipodal_kwargs)

    return gripper_6dof_pose


def cvt_genome_to_6dof_pose_antipodal_strat_contact_point_finder_search(robot_grasp_env, genome):
    contact_point_finder_xyz_pose, hand_orientation_from_entraxe = \
        cvt_genome_to_preset_6dof_pose_antipodal_strat_contact_point_finder(robot_grasp_env=robot_grasp_env, genome=genome)

    default_gripper_pose_kwargs = {
        'contact_point_finder_xyz_pose': contact_point_finder_xyz_pose,
        'hand_orientation_from_entraxe': hand_orientation_from_entraxe,
    }
    default_object_kwargs = {
        'object_id': robot_grasp_env.obj_id,
        'path2obj_point_cloud': robot_grasp_env.path2obj_point_cloud,
    }

    antipodal_kwargs = {
        'robot_grasp_env': robot_grasp_env,
        'default_object_kwargs': default_object_kwargs,
        'default_gripper_pose_kwargs': default_gripper_pose_kwargs,
    }
    gripper_6dof_pose = get_6dof_pose_antipodal(**antipodal_kwargs)

    return gripper_6dof_pose


def get_6dof_pose_antipodal(
        robot_grasp_env,
        default_object_kwargs,
        default_gripper_pose_kwargs,
        debug=False,
):
    object_id = default_object_kwargs['object_id']

    # ------------------------------------------------------------------------------------------------------------ #
    #                                             1) FIRST CONTACT POINT
    # ------------------------------------------------------------------------------------------------------------ #

    is_cpf_variant = 'contact_point_finder_xyz_pose' in default_gripper_pose_kwargs
    if is_cpf_variant:
        contact_point_finder_xyz_pose = default_gripper_pose_kwargs['contact_point_finder_xyz_pose']
        hand_orientation_from_entraxe = default_gripper_pose_kwargs['hand_orientation_from_entraxe']

        # Get closer point on the object surface
        query = [contact_point_finder_xyz_pose]
        closest_contact_point_id = robot_grasp_env.sim_engine.k_tree_uniform_contact_points.kneighbors(X=query)[1][0][0]
        contact_point = robot_grasp_env.sim_engine.uniform_obj_contact_points[closest_contact_point_id]
    else:
        start_pos_robot_xyz = default_gripper_pose_kwargs['start_pos_robot_xyz']
        euler_robot = default_gripper_pose_kwargs['euler_robot']
        hand_orientation_from_entraxe = default_gripper_pose_kwargs['hand_orientation_from_entraxe']

        is_contact_point_found, contact_point, vector_director = search_first_contact_point_clean(
            bullet_client=robot_grasp_env.bullet_client,
            object_id=object_id,
            start_pos_robot_xyz=start_pos_robot_xyz,
            euler_robot=euler_robot,
        )
        if not is_contact_point_found:
            return None

    normal_at_contact_point = get_normal_surface_point(
        bullet_client=robot_grasp_env.bullet_client,
        list_of_points_for_each_triangle_object_mesh=robot_grasp_env.list_of_points_for_each_triangle_obj_mesh,
        object_normals_to_triangles=robot_grasp_env.object_normals_to_triangles,
        contact_point=contact_point,
        debug=debug
    )

    # ------------------------------------------------------------------------------------------------------------ #
    #                                              2) OPPOSITE POINT
    # ------------------------------------------------------------------------------------------------------------ #

    is_opposite_point_found, live_point_opposite, vector_director_antipodal = search_opposite_contact(
        bullet_client=robot_grasp_env.bullet_client,
        normal_at_contact_point=normal_at_contact_point,
        contact_point=contact_point,
        object_id=object_id
    )
    if not is_opposite_point_found:
        return None

    normal_vector_opposite = calculate_normal_from_surface(
        bullet_client=robot_grasp_env.bullet_client,
        list_of_points_for_each_triangle=robot_grasp_env.list_of_points_for_each_triangle_obj_mesh,
        contact_point=live_point_opposite,
        object_normals_to_triangles=robot_grasp_env.object_normals_to_triangles,
        debug=debug
    )

    # ------------------------------------------------------------------------------------------------------------ #
    #                                            3) ANTIPODALITY
    # ------------------------------------------------------------------------------------------------------------ #

    gamma = angle_between_2_vectors(
        bullet_client=robot_grasp_env.bullet_client,
        first_vector=normal_at_contact_point,
        second_vector=normal_vector_opposite,
        debug=debug
    )

    are_antipodal_points_found = is_inside_antipodal_selectability_cone(gamma)
    if not are_antipodal_points_found:
        return None

    # ------------------------------------------------------------------------------------------------------------ #
    #                                            4) ENTRAXE VALIDITY
    # ------------------------------------------------------------------------------------------------------------ #

    is_entraxe_selectable = entreaxe_selectability(point_first=contact_point, point_second=live_point_opposite)
    if not is_entraxe_selectable:
        return None

    # ------------------------------------------------------------------------------------------------------------ #
    #                                          5) GRIPPER ORIENTATION
    # ------------------------------------------------------------------------------------------------------------ #

    center_axis_R0, center_axis_mid_point_world = entraxe_world_generation(
        point_in=contact_point,
        point_out=live_point_opposite
    )

    tau_choosen = hand_orientation_from_entraxe['tau']

    gripper_wrist_pose, gripper_wrist_orient_quat = compute_new_gripper_pose(
        bullet_client=robot_grasp_env.bullet_client,
        tau=tau_choosen,
        center_axis_R0=center_axis_R0,
        gripper_distance_to_axis=EXTREMITY_WIRST,
        center_axis_mid_point_world=center_axis_mid_point_world,
    )

    # ------------------------------------------------------------------------------------------------------------ #
    #                                     6) CHECK GRIPPER OBJECT OVERLAPPING
    # ------------------------------------------------------------------------------------------------------------ #

    robot_grasp_env.bullet_client.stepSimulation()
    if robot_grasp_env.sim_engine.is_there_object_robot_overlapping(robot_grasp_env.bullet_client):
        return None
    else:
        pass

    gripper_6dof_pose = {
        'xyz': gripper_wrist_pose,
        'euler_rpy': None,
        'quaternions': gripper_wrist_orient_quat
    }

    return gripper_6dof_pose




