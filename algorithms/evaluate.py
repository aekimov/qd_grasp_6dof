
import numpy as np
import time

import environments.src.env_constants as env_consts
import environments.src.genome_to_6dof_poses as cvt_gen_6dof

import configs.eval_config as eval_cfg
import configs.qd_config as qd_cfg

INVALID_GRASP_POSE_FITNESS = 0.
INVALID_GRASP_POSE_BEHAVIOR = [0., 0., 0.]
INVALID_GRASP_POSE_IS_GRASP_SUCCESSFUL_FLG = 0.
INVALID_GRASP_POSE_IS_VALID_FLG = 0.
INVALID_GRASP_POSE_IS_OBJECT_TOUCHED_FLG = 0.
INVALID_GRASP_POSE_IS_ROBUST_GRASP_FLG = 0.
INVALID_GRASP_XYZ_POSE_X_DEFAULT_VALUE = 0.
INVALID_GRASP_XYZ_POSE_Y_DEFAULT_VALUE = 0.
INVALID_GRASP_XYZ_POSE_Z_DEFAULT_VALUE = 0.
INVALID_GRASP_XYZ_QUATERNION_1_DEFAULT_VALUE = 0.
INVALID_GRASP_XYZ_QUATERNION_2_DEFAULT_VALUE = 0.
INVALID_GRASP_XYZ_QUATERNION_3_DEFAULT_VALUE = 0.
INVALID_GRASP_XYZ_QUATERNION_4_DEFAULT_VALUE = 0.


def evaluate_grasp_ind_routine(individual, env, eval_kwargs):

    raise_exceptions_flg = True
    try:
        behavior, fitness, info = evaluate_grasp_ind(
            individual=individual,
            env=env,
            eval_kwargs=eval_kwargs
        )
    except Exception as e:
        # Might be raised in some pathological cases due to simulator issues
        if raise_exceptions_flg:
            raise e
        behavior, fitness, info = exception_handler_evaluate_grasp_ind(
            individual=individual, eval_kwargs=eval_kwargs,
        )

    return behavior, fitness, info


def init_output_dict():
    gripper_6dof_output_data = {
        'is_success': None,
        'is_robust_grasp': None,
        'is_overlap': None,
        'is_obj_touched': None,
        '6dof_data': {
            j_id: {'is_success': None, 'n_shake_success': None}
            for j_id in env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']
        }
    }
    return gripper_6dof_output_data


def get_overlap_output_dict():
    gripper_6dof_output_data = {
        'is_success': False,
        'is_robust_grasp': False,
        'is_overlap': True,
        'is_obj_touched': False,
        '6dof_data': None,
    }
    return gripper_6dof_output_data


def get_grasp_failure_output_dict(is_obj_touched):
    gripper_6dof_output_data = {
        'is_success': False,
        'is_robust_grasp': False,
        'is_overlap': False,
        'is_obj_touched': is_obj_touched,
        '6dof_data': None,
    }
    return gripper_6dof_output_data


def evaluate_6dof_pose(
        env, gripper_6dof_pose, synergy_label=None, init_joint_state_genes=None, domain_randomization_args=None
):
    env.reset()

    # Initialize running variables
    gripper_6dof_output_data = init_output_dict()
    robot_id = env.robot_id
    fingers_joint_infos = env.fingers_joint_infos

    if domain_randomization_args is not None:
        if domain_randomization_args['randomize_object_state']:
            env.add_noise_to_object_state()
        if domain_randomization_args['randomize_friction_coefficients']:
            env.add_noise_to_friction_coefficients()

    # Apply 6DoF pose to grasp env
    env.set_6dof_gripper_pose(gripper_6dof_pose)
    debug_snippet_flg = False
    if debug_snippet_flg:
        gripper_6dof_pose = force_gripper_6dof_pose_debug_snippet(gripper_6dof_pose)
        env.set_6dof_gripper_pose(gripper_6dof_pose)

    if init_joint_state_genes is not None:
        env.set_joint_states_from_genes(init_joint_state_genes)

    # Check for overlapping
    is_there_overlap = env.is_there_overlapping()
    if is_there_overlap:
        return get_overlap_output_dict()
    else:
        gripper_6dof_output_data['is_overlap'] = False

    # Close gripper
    is_obj_touched = env.close_gripper(
        robot_id=robot_id,
        fingers_joint_infos=fingers_joint_infos,
        synergy_label=synergy_label,
    )
    gripper_6dof_output_data['is_obj_touched'] = is_obj_touched

    # Discard if not touching the object
    is_grasp_attempt_a_failure = not env.is_grasping_candidate()
    if is_grasp_attempt_a_failure:
        return get_grasp_failure_output_dict(is_obj_touched=is_obj_touched)

    # Candidate grasp: apply perturbation
    gripper_6dof_output_data = env.apply_all_gripper_shaking(gripper_6dof_output_data)

    return gripper_6dof_output_data


def get_fitness(gripper_6dof_output_data, eval_kwargs, eval_6dof_pose_kwargs):
    if gripper_6dof_output_data['6dof_data'] is None:
        return INVALID_GRASP_POSE_FITNESS

    if eval_kwargs['domain_randomization_fitness']:
        return get_mixture_domain_randomization_fitness(
            gripper_6dof_output_data=gripper_6dof_output_data, eval_6dof_pose_kwargs=eval_6dof_pose_kwargs
        )
    else:
        return np.array(
            [j_id_shake_data['n_shake_success'] for j_id, j_id_shake_data in gripper_6dof_output_data['6dof_data'].items()]
        ).sum()


def get_mixture_domain_randomization_fitness(gripper_6dof_output_data, eval_6dof_pose_kwargs):

    if not gripper_6dof_output_data['is_success']:
        return eval_cfg.MIN_DOMAIN_RANDOMIZATION_FITNESS_VALUE

    dr_eval_6dof_pose_kwargs = eval_6dof_pose_kwargs.copy()
    dr_eval_6dof_pose_kwargs['domain_randomization_args'] = {
        'randomize_object_state': True, 'randomize_friction_coefficients': True
    }

    all_noisy_gripper_6dof_output_data = [
        evaluate_6dof_pose(**dr_eval_6dof_pose_kwargs)
        for i_noisy_trial in range(eval_cfg.DOMAIN_RANDOMIZATION_N_NOISY_TRIALS)
    ]

    n_shake_successes = np.array([
        np.array(
            [j_id_shake_data['n_shake_success'] for j_id, j_id_shake_data in
             single_grip_6dof_out_data['6dof_data'].items()]
        ).sum()
        if single_grip_6dof_out_data['6dof_data'] is not None else 0
        for single_grip_6dof_out_data in all_noisy_gripper_6dof_output_data
    ]).sum()

    dr_fitness = n_shake_successes
    #print(f'dr_fitness=', dr_fitness)
    return dr_fitness


def get_synergy_label(individual, eval_kwargs):
    return individual[-1] if eval_kwargs['with_synergy'] else None


def get_init_joint_state_genes(individual, eval_kwargs):
    if not eval_kwargs['with_init_joint_state_in_genome']:
        return None

    if eval_kwargs['with_synergy']:
        # The last gene describes the synergy label -> init joints poses are shifted by 1 value from the end
        init_joint_states_genes = individual[-eval_kwargs['n_init_joint_states']-1:-1]

        # Note: There is currently no gripper with both synergy and init joint state in genome. Carefully debug the
        # extraction of genome values if you want to add one that match this case.
    else:
        # No synergy : init joints poses are shifted the last values and can be extracted from the end
        init_joint_states_genes = individual[-eval_kwargs['n_init_joint_states']:]

    return init_joint_states_genes


def invalid_6dof_pose_evaluation_outcome():
    invalid_pose_behavior = INVALID_GRASP_POSE_BEHAVIOR
    invalid_pose_fitness = INVALID_GRASP_POSE_FITNESS

    invalid_pose_info = [
        INVALID_GRASP_POSE_IS_GRASP_SUCCESSFUL_FLG,
        INVALID_GRASP_POSE_IS_VALID_FLG,
        INVALID_GRASP_POSE_IS_OBJECT_TOUCHED_FLG,
        INVALID_GRASP_POSE_IS_ROBUST_GRASP_FLG,

        INVALID_GRASP_XYZ_POSE_X_DEFAULT_VALUE,
        INVALID_GRASP_XYZ_POSE_Y_DEFAULT_VALUE,
        INVALID_GRASP_XYZ_POSE_Z_DEFAULT_VALUE,
        INVALID_GRASP_XYZ_QUATERNION_1_DEFAULT_VALUE,
        INVALID_GRASP_XYZ_QUATERNION_2_DEFAULT_VALUE,
        INVALID_GRASP_XYZ_QUATERNION_3_DEFAULT_VALUE,
        INVALID_GRASP_XYZ_QUATERNION_4_DEFAULT_VALUE,
    ]

    return invalid_pose_behavior, invalid_pose_fitness, invalid_pose_info


def cvt_genome_to_6dof_pose(individual, env, eval_kwargs):
    search_representation = eval_kwargs['search_representation']
    robot = eval_kwargs['robot']

    if search_representation == qd_cfg.SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER:
        gripper_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_approach_strat_contact_point_finder_search(
            robot_grasp_env=env, genome=individual, robot=eval_kwargs['robot']
        )
    elif search_representation == qd_cfg.SearchSpaceRepresentation.ANTIPODAL_STRATEGY_CONTACT_POINT_FINDER:
        gripper_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_antipodal_strat_contact_point_finder_search(
            robot_grasp_env=env, genome=individual
        )
    elif search_representation == qd_cfg.SearchSpaceRepresentation.RANDOM_SAMPLE_CONTACT_STRATEGY_LIST:
        gripper_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_rand_sample_contact_strategy_list_search(
            robot_grasp_env=env, genome=individual
        )
    elif search_representation == qd_cfg.SearchSpaceRepresentation.CONTACT_STRATEGY_CONTACT_POINT_FINDER:
        gripper_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_contact_strategy_search(
            robot_grasp_env=env, genome=individual, robot=robot
        )
    elif search_representation == qd_cfg.SearchSpaceRepresentation.SPHERICAL:
        gripper_6dof_pose = cvt_gen_6dof.cvt_genome_to_6dof_pose_spherical(
            robot_grasp_env=env, genome=individual
        )
    else:
        raise NotImplementedError('error in search representation type')

    return gripper_6dof_pose


def evaluate_grasp_ind(individual, env, eval_kwargs):

    # [-1, 1] =====> [x, y, z, r, p, y]
    env.reset()

    gripper_6dof_pose = cvt_genome_to_6dof_pose(individual=individual, env=env, eval_kwargs=eval_kwargs)

    if gripper_6dof_pose is None:
        return invalid_6dof_pose_evaluation_outcome()

    synergy_label = get_synergy_label(individual=individual, eval_kwargs=eval_kwargs)
    init_joint_state_genes = get_init_joint_state_genes(individual=individual, eval_kwargs=eval_kwargs)
    eval_6dof_pose_kwargs = {
        'env': env,
        'gripper_6dof_pose': gripper_6dof_pose,
        'synergy_label': synergy_label,
        'init_joint_state_genes': init_joint_state_genes,
    }

    gripper_6dof_output_data = evaluate_6dof_pose(**eval_6dof_pose_kwargs)

    if env.is_debug_mode():
        time.sleep(2)
        env.delete_debug_bodies()

    is_valid = not gripper_6dof_output_data['is_overlap']
    is_obj_touched = gripper_6dof_output_data['is_obj_touched']
    is_robust_grasp = gripper_6dof_output_data['is_robust_grasp']
    is_grasp_successful = gripper_6dof_output_data['is_success']

    fitness = get_fitness(
        gripper_6dof_output_data=gripper_6dof_output_data,
        eval_kwargs=eval_kwargs,
        eval_6dof_pose_kwargs=eval_6dof_pose_kwargs
    )
    behavior = gripper_6dof_pose['xyz']

    # Build the info array such that all values are floats
    outcome_info = [float(is_grasp_successful), float(is_valid), float(is_obj_touched), float(is_robust_grasp)]
    outcome_6dof = gripper_6dof_pose['xyz'] + gripper_6dof_pose['quaternions']
    info = outcome_info + outcome_6dof

    # IMPORTANT: To optimize computation, infos are treated as vectors insteads of dicts.
    # Each column key can be found in configs.eval_config.INFO_KEYS

    return behavior, fitness, info


def exception_handler_evaluate_grasp_ind(individual, eval_kwargs):
    raise NotImplementedError()


def force_gripper_6dof_pose_debug_snippet(gripper_6dof_pose):
    """Debug function. Change these values to debug hand closure."""
    gripper_6dof_pose['xyz'] = [0, 0, 0.6]
    gripper_6dof_pose['quaternions'] = [0, 0, 0, 1]
    return gripper_6dof_pose

