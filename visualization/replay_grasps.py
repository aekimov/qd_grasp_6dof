import pdb

import numpy as np
import argparse
import sys
import random

from visualization.vis_tools import load_run_output_files, load_all_inds, get_folder_path_from_str, init_env, \
	load_all_infos, load_all_fitnesses, get_info_from_key, get_all_gripper_6dof_pose_from_infos

from algorithms.evaluate import get_synergy_label, get_init_joint_state_genes

import environments.src.robot_grasping_debug as rg_db


def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, required=True)
	parser.add_argument("-rb", "--robust", action="store_true", help="Display grasps that succeed after full shaking.")
	parser.add_argument("-s", "--shake", action="store_true", help="Apply shaking on the grasps.")
	parser.add_argument("-si", "--shuffle-inds", action="store_true", help="Randomly shuffle inds.")
	parser.add_argument("-i", "--i_ind", help="Index of a specific ind to display.", type=int, default=None)
	parser.add_argument("-fs", "--fitness-sorted", action="store_true",
						help="Display grasps from higher to lower fitness.")
	return parser.parse_args()


def get_replay_grasps_kwargs():
	args = arg_parser()

	return {
		'run_folder_path': args.runs,
		'display_flags': {
			'only_robust_inds': args.robust,
			'i_ind2display': args.i_ind,
			'shake': args.shake,
			'shuffle_inds': args.shuffle_inds,
			'fitness_sorted': args.fitness_sorted,
		}
	}


def replay_6dof_specific_pose(env, all_6dof_pose_data, display_flags):
	robot_id = env.robot_id
	fingers_joint_infos = env.fingers_joint_infos

	i_ind2display = display_flags['i_ind2display']
	gripper_6dof_pose = all_6dof_pose_data[i_ind2display]['6dof_pose']

	is_robust_grasp = get_info_from_key(
		infos=all_6dof_pose_data[i_ind2display]['infos'],
		info_keys=all_6dof_pose_data[i_ind2display]['info_keys'],
		key='is_robust_grasp'
	)

	fitness = all_6dof_pose_data[i_ind2display]['fitness']
	synergy_label = all_6dof_pose_data[i_ind2display]['synergy_label']
	init_joint_state_genes = all_6dof_pose_data[i_ind2display]['init_joint_state_genes']

	print(f'Displaying successful grasp n°{i_ind2display} | is_robust_grasp={is_robust_grasp} fitness={fitness}')

	n_replay = 20
	for i_replay in range(n_replay):
		env.reset()

		env.set_6dof_gripper_pose(gripper_6dof_pose)

		if init_joint_state_genes is not None:
			env.set_joint_states_from_genes(init_joint_state_genes)

		env.close_gripper(
			robot_id=robot_id,
			fingers_joint_infos=fingers_joint_infos,
			synergy_label=synergy_label
		)

		if display_flags['shake']:
			are_all_shakes_successful = rg_db.apply_all_gripper_shaking_debug(env)
			print(f'(id={robot_id}) are_all_shakes_successful={are_all_shakes_successful}')

	print(f'Display over.')


def replay_all_6dof_poses(env, all_6dof_pose_data, display_flags):
	robot_id = env.robot_id
	fingers_joint_infos = env.fingers_joint_infos

	all_all_6dof_pose_ids = list(all_6dof_pose_data.keys())
	if display_flags['shuffle_inds']:
		random.shuffle(all_all_6dof_pose_ids)
	if display_flags['fitness_sorted']:
		all_fits = [all_6dof_pose_data[ind_id]['fitness'] for ind_id in all_6dof_pose_data]
		ind_ids_sorted_descent_order = np.argsort(all_fits)[::-1]
		all_all_6dof_pose_ids = ind_ids_sorted_descent_order

	for scs_ind_id in all_all_6dof_pose_ids:
		print('scs_ind_id=', scs_ind_id)
		gripper_6dof_pose = all_6dof_pose_data[scs_ind_id]['6dof_pose']
		is_robust_grasp = get_info_from_key(
			infos=all_6dof_pose_data[scs_ind_id]['infos'],
			info_keys=all_6dof_pose_data[scs_ind_id]['info_keys'],
			key='is_robust_grasp'
		)
		fitness = all_6dof_pose_data[scs_ind_id]['fitness']
		synergy_label = all_6dof_pose_data[scs_ind_id]['synergy_label']
		init_joint_state_genes = all_6dof_pose_data[scs_ind_id]['init_joint_state_genes']

		if not is_robust_grasp and display_flags['only_robust_inds']:
			continue

		print(f'Displaying successful grasp n°{scs_ind_id} | is_robust_grasp={is_robust_grasp} fitness={fitness}')

		env.reset()

		env.set_6dof_gripper_pose(gripper_6dof_pose)

		if init_joint_state_genes is not None:
			env.set_joint_states_from_genes(init_joint_state_genes)

		env.close_gripper(
			robot_id=robot_id,
			fingers_joint_infos=fingers_joint_infos,
			synergy_label=synergy_label
		)
		if display_flags['shake']:
			are_all_shakes_successful = rg_db.apply_all_gripper_shaking_debug(env)
			print(f'(id={scs_ind_id}) are_all_shakes_successful={are_all_shakes_successful}')

	print(f'Display over.')


def replay_6dof_poses(env, all_6dof_pose_data, display_flags):
	if display_flags['i_ind2display'] is not None:
		replay_6dof_specific_pose(
			env=env,
			all_6dof_pose_data=all_6dof_pose_data,
			display_flags=display_flags
		)
	else:
		replay_all_6dof_poses(
			env=env,
			all_6dof_pose_data=all_6dof_pose_data,
			display_flags=display_flags
		)


def get_all_synergy_labels(individuals, cfg):

	if not cfg['evaluate']['kwargs']['with_synergy']:
		undefined_all_synergy = [None] * len(individuals)
		return undefined_all_synergy

	all_synergy_labels = [
		get_synergy_label(individual=ind, eval_kwargs=cfg['evaluate']['kwargs']) for ind in individuals
	]
	return all_synergy_labels


def get_all_init_joint_state_genes(individuals, cfg):

	before_init_joint_state_commit = 'with_init_joint_state_in_genome' not in cfg['evaluate']['kwargs']
	if before_init_joint_state_commit:
		undefined_all_init_joint_state_genes = [None] * len(individuals)
		return undefined_all_init_joint_state_genes

	if not cfg['evaluate']['kwargs']['with_init_joint_state_in_genome']:
		undefined_all_init_joint_state_genes = [None] * len(individuals)
		return undefined_all_init_joint_state_genes

	all_init_joint_state_genes = [
		get_init_joint_state_genes(individual=ind, eval_kwargs=cfg['evaluate']['kwargs']) for ind in individuals
	]
	return all_init_joint_state_genes


def get_all_6dof_grasping_data(folder, cfg):
	individuals = load_all_inds(folder)
	infos, info_keys = load_all_infos(folder)
	fitnesses = load_all_fitnesses(folder)

	all_gripper_6dof_pose = get_all_gripper_6dof_pose_from_infos(info_keys=info_keys, infos=infos)

	all_synergy_labels = get_all_synergy_labels(individuals=individuals, cfg=cfg)

	all_init_joint_state_genes = get_all_init_joint_state_genes(individuals=individuals, cfg=cfg)

	all_6dof_pose_data = {
		scs_ind_id: {
			'6dof_pose': gripper_6dof_pose,
			'fitness': fitnesses[scs_ind_id],
			'infos': infos[scs_ind_id, :],
			'info_keys': info_keys,
			'synergy_label': synergy_label,
			'init_joint_state_genes': init_joint_state_genes,
		}
		for scs_ind_id, (ind, gripper_6dof_pose, synergy_label, init_joint_state_genes) in enumerate(
			zip(individuals, all_gripper_6dof_pose, all_synergy_labels, all_init_joint_state_genes)
		)
	}

	return all_6dof_pose_data


def init_replay_6dof_grasp_poses(run_folder_path, display_flags, display=True, remove_gripper=False):
	# Load run folder
	folder = get_folder_path_from_str(run_folder_path)

	# Extract corresponding data
	run_details, run_infos, cfg = load_run_output_files(folder)

	# Init grasp gym env
	env = init_env(cfg, display=display, remove_gripper=remove_gripper)

	# Get the 6dof data to replay
	all_6dof_pose_data = get_all_6dof_grasping_data(folder=folder, cfg=cfg)
	return env, all_6dof_pose_data, display_flags


def replay_grasps(run_folder_path, display_flags):
	env, all_6dof_pose_data, display_flags = init_replay_6dof_grasp_poses(run_folder_path, display_flags)

	replay_6dof_poses(
		env=env,
		all_6dof_pose_data=all_6dof_pose_data,
		display_flags=display_flags
	)

	print('Ending.')


def main():
	kwargs = get_replay_grasps_kwargs()
	replay_grasps(**kwargs)


if __name__ == "__main__":
	sys.exit(main())
