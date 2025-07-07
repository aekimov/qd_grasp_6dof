
import evolutionary_process
import sys
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('fork')  # Only works on Unix systems

from utils.args_processor import get_qd_algo_args, get_input_arguments
from utils.common_tools import get_new_run_name

from utils.io_run_data import export_dict_pickle

import configs.exec_config as ex_cfg
import configs.eval_config as eval_cfg
import environments.src.env_constants as env_consts


def dump_input_arguments(qd_algo_args, global_config):
    """Dump arguments that will be given to the QD algorithm to generate grasping trajectories."""
    export_dict_pickle(run_name=qd_algo_args['run_name'], dict2export=global_config, file_name='config')


def end_of_run_routine(archive_success_len):
    """Must only be called in the main thread."""
    if archive_success_len == 0:
        print("Empty success archive.")
        return ex_cfg.RETURN_SUCCESS_CODE

    print(f"End of running. Size of output success archive : {archive_success_len}")
    print("Success.")

    return ex_cfg.RETURN_SUCCESS_CODE


def init_run_dump_folder(global_config):
    """Must only be called in the main thread."""
    run_name = get_new_run_name(
        log_path=global_config['output']['log_path'], folder_name=global_config['output']['folder_name']
    )
    return run_name


def init_grasping_env(env_class, env_kwargs):
    env = env_class(**env_kwargs)
    return env


def get_global_config(input_args, env):

    stabilized_obj_pose, _ = env.bullet_client.getBasePositionAndOrientation(env.obj_id)

    search_space_bounding_box = env.search_space_bb
    n_domain_randomization_perturbations = eval_cfg.DOMAIN_RANDOMIZATION_N_NOISY_TRIALS \
        if input_args['eval_kwargs']['domain_randomization_fitness'] else None
    shaking_params = env_consts.SHAKING_PARAMETERS

    global_config = {
        'algorithm': input_args['qd_method'],

        'evo_proc': {
            'archive_type': input_args['archive_type'],
            'pop_size': input_args['pop_size'],
            'n_budget_rollouts': input_args['n_budget_rollouts'],
            'mut_strat': input_args['mut_strat'],
            'prob_cx': input_args['prob_cx'],
            'sigma_mut': input_args['sigma_mut'],
            'select_off_strat': input_args['select_off_strat'],
            'replace_pop_strat': input_args['replace_pop_strat'],
            'archive_limit_strat': input_args['archive_limit_strat'],
            'is_novelty_required': input_args['is_novelty_required'],
            'is_pop_based': input_args['is_pop_based'],
            'qd_method_genotype_len': input_args['qd_method_genotype_len'],
            'include_invalid_inds': input_args['include_invalid_inds'],
        },

        'env': {
            'kwargs': input_args['env_kwargs'],
        },

        'evaluate': {
            'kwargs': input_args['eval_kwargs'],
            'search_space_bb': search_space_bounding_box,
            'n_domain_randomization_perturbations': n_domain_randomization_perturbations,
            'shaking_params': shaking_params,
        },

        'robot': {
            'name': input_args['robot'],
            'env_class': input_args['env_class'],
            'name_str': input_args['robot_str']
        },

        'object': {
            'name': input_args['object'],
            'stabilized_obj_pose': stabilized_obj_pose,
        },

        'output': {
            'log_path': input_args['log_path'],
            'folder_name': input_args['folder_name'],
        },

        'parallelize': input_args['parallelize'],
        'search_representation': input_args['search_representation'],
        'debug': input_args['debug'],

    }

    return global_config


def run_qd_routine(**qd_algo_args):

    if qd_algo_args['parallelize']:
        with Pool() as multiproc_pool:
            qd_algo_args['multiproc_pool'] = multiproc_pool
            archive_success_len = evolutionary_process.run_qd(**qd_algo_args)
    else:
        archive_success_len = evolutionary_process.run_qd(**qd_algo_args)

    return archive_success_len


def main():
    """QD-Grasp entry point."""

    # Initialize arguments
    input_args = get_input_arguments()
    env = init_grasping_env(env_class=input_args['env_class'], env_kwargs=input_args['env_kwargs'])
    global_config = get_global_config(input_args=input_args, env=env)

    dump_folder_name = init_run_dump_folder(global_config=global_config)

    qd_algo_args = get_qd_algo_args(cfg=global_config, env=env, dump_folder_name=dump_folder_name)

    # Locally save params
    dump_input_arguments(qd_algo_args=qd_algo_args, global_config=global_config)

    # QD algorithm execution
    archive_success_len = run_qd_routine(**qd_algo_args)

    # End of running
    return end_of_run_routine(archive_success_len)


if __name__ == "__main__":
    sys.exit(main())


