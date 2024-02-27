import os.path

from sklearn.neighbors import NearestNeighbors as Nearest
import numpy as np
from pathlib import Path

import algorithms.stats as stats
from utils.io_run_data import load_dict_pickle
import configs.eval_config as eval_cfg

import environments.src.env_constants as env_consts


N_DECIMAL_PRECISION = 2
OPTIMAL_FITNESS = 2.0


class StatsTracker:
    def __init__(self, qd_method, targeted_object, robot_name_str):

        self._outcome_archive_cvg_hist = []  # coverage of As w.r.t. the number of evaluations

        self._success_archive_cvg_hist = []  # coverage of As w.r.t. the number of evaluations
        self._success_archive_qd_score_hist = []  # sum of the normalized fitness of the archive

        self._grasp_pose_cvg_hist = []
        self._robust_grasp_pose_cvg_hist = []
        self._all_possible_grasp_poses_to_fit = None
        self._n_possible_robust_grasp_poses = None

        self._success_archive_sparsity_3_hist = []
        self._success_archive_sparsity_4_hist = []

        self._outcome_ratio_hist = []  # ratio of evaluated ind that touches the object
        self._success_ratio_hist = []  # ratio of evaluated ind that grasp the object

        self._n_evals = []  # number of evaluations corresponding to each export (each generation)
        self._n_evals_including_invalid = []  # number of evaluations corresponding to each export (each generation)
        self._run_time = []  # running time corresponding to each export (each generation)

        self._rolling_n_touched = 0  # number of evaluation considered in update() in which the object is touched
        self._rolling_n_success = 0  # number of evaluation considered in update() in which the object is grasped
        self._rolling_n_rollout = 0  # number of rollouts considered in update()

        self._first_saved_ind_gen = None  # generation at which the first successful ind had been generated
        self._first_saved_ind_n_evals = None  # number of domain evaluation when the 1st scs ind had been generated

        self.qd_method = None

        self._init_attributes(qd_method=qd_method, targeted_object=targeted_object, robot_name_str=robot_name_str)

    @property
    def first_saved_ind_gen(self):
        return self._first_saved_ind_gen

    @property
    def first_saved_ind_n_evals(self):
        return self._first_saved_ind_n_evals

    def _init_attributes(self, qd_method, targeted_object, robot_name_str):
        self.qd_method = qd_method
        self._all_possible_grasp_poses_to_fit = self._load_all_possible_grasp_poses_to_fit(
            targeted_object=targeted_object, robot_name_str=robot_name_str
        )
        self._sanity_checks_all_possible_grasp_poses_to_fit()
        self._n_possible_robust_grasp_poses = self._extract_n_possible_robust_grasps_poses()

    def _load_all_possible_grasp_poses_to_fit(self, targeted_object, robot_name_str):

        root_all_grasp_poses = Path(stats.__file__).resolve().parent / 'all_possible_grasp_poses_to_fit'

        file_pkl_name = str(root_all_grasp_poses) + f'/{robot_name_str}/random_sample_contact_{targeted_object}.pkl'

        outcome_space_xyz_poses_to_fit = load_dict_pickle(file_pkl_name) if os.path.exists(file_pkl_name) else None

        return outcome_space_xyz_poses_to_fit

    def _sanity_checks_all_possible_grasp_poses_to_fit(self):

        if self._all_possible_grasp_poses_to_fit is None:
            print('Stats_tracker sanity checks) Warning: self._all_possible_grasp_poses_to_fit is not defined. '
                  'The corresponding set has either not been computed, or the path file might be wrongly established.')
            return

        # Warning: OPTIMAL_FITNESS must be the same in the current execution and in the previously computed
        # self._all_possible_grasp_poses_to_fit

        # Check 1: is max value match the current max value?
        max_fit_in_local_data = max(self._all_possible_grasp_poses_to_fit.values())
        assert max_fit_in_local_data == OPTIMAL_FITNESS

        n_shake_sequences = env_consts.SHAKING_PARAMETERS['n_shake']
        shaked_joints_per_sequences = len(env_consts.SHAKING_PARAMETERS['perturbated_joint_ids'])
        tot_n_shake = shaked_joints_per_sequences * n_shake_sequences
        max_fit_in_curr_run = tot_n_shake
        assert max_fit_in_curr_run == OPTIMAL_FITNESS
        print('Stats_tracker sanity checks) Check 1 - is max value match the current max value: OK')

        # Warning: make sure the rounding must be done on 2 decimals (1 cm)!!!

        # Check 2: is the defined rounding corresponding to the number of decimal in the local file?
        for xyz_pose in self._all_possible_grasp_poses_to_fit:
            assert len(xyz_pose) == 3
            n_decimal_values_x = len(str(abs(xyz_pose[0])).split('.')[-1])
            n_decimal_values_y = len(str(abs(xyz_pose[1])).split('.')[-1])
            n_decimal_values_z = len(str(abs(xyz_pose[2])).split('.')[-1])

            assert n_decimal_values_x <= N_DECIMAL_PRECISION  # <= is required as left 0 are discarded
            assert n_decimal_values_y <= N_DECIMAL_PRECISION
            assert n_decimal_values_z <= N_DECIMAL_PRECISION

        print('Stats_tracker sanity checks) Check 2 - is the defined rounding corresponding to the number of decimal '
              'in the local file: OK')

    def _extract_n_possible_robust_grasps_poses(self):
        if self._all_possible_grasp_poses_to_fit is None:
            return None

        n_possible_robust_grasp_poses = np.sum(
            [possible_fit == OPTIMAL_FITNESS for possible_fit in self._all_possible_grasp_poses_to_fit.values()]
        )
        return n_possible_robust_grasp_poses

    def update(self, pop, outcome_archive, curr_n_evals, curr_n_evals_including_invalid, timer, timer_label):
        outcome_archive_cvg = self._get_outcome_archive_cvg(outcome_archive)
        success_archive_cvg = self._get_success_archive_cvg(outcome_archive)
        success_archive_qd_score = self._get_success_archive_qd_score(outcome_archive)
        outcome_ratio, success_ratio = self._get_outcome_and_success_ratios(pop, curr_n_evals)
        curr_run_time = timer.get_on_the_fly_time(label=timer_label)

        success_archive_sparsity_3 = self._get_success_sparsity_3_cvg(outcome_archive)
        success_archive_sparsity_4 = self._get_success_sparsity_4_cvg(outcome_archive)

        self._outcome_archive_cvg_hist.append(outcome_archive_cvg)
        self._success_archive_cvg_hist.append(success_archive_cvg)
        self._success_archive_sparsity_3_hist.append(success_archive_sparsity_3)
        self._success_archive_sparsity_4_hist.append(success_archive_sparsity_4)

        grasp_pose_cvg, robust_grasp_pose_cvg = self._get_grasp_pose_cvg(outcome_archive)
        self._grasp_pose_cvg_hist.append(grasp_pose_cvg)
        self._robust_grasp_pose_cvg_hist.append(robust_grasp_pose_cvg)

        self._success_archive_qd_score_hist.append(success_archive_qd_score)
        self._outcome_ratio_hist.append(outcome_ratio)
        self._success_ratio_hist.append(success_ratio)
        self._n_evals.append(curr_n_evals)
        self._n_evals_including_invalid.append(curr_n_evals_including_invalid)
        self._run_time.append(curr_run_time)

    def _get_outcome_archive_cvg(self, outcome_archive):
        return len(outcome_archive) / outcome_archive.max_size

    def _get_success_archive_cvg(self, outcome_archive):
        return outcome_archive.get_n_successful_cells() / outcome_archive.max_size

    def _get_success_archive_qd_score(self, outcome_archive):
        return np.sum(outcome_archive.fits)

    def _get_grasp_pose_cvg(self, outcome_archive):

        if self._all_possible_grasp_poses_to_fit is None:
            return None, None

        found_xyz_grasp_poses = outcome_archive.get_successful_inds_xyz_poses()
        if len(found_xyz_grasp_poses) == 0:
            grasp_pose_cvg, robust_grasp_pose_cvg = 0., 0.
            return grasp_pose_cvg, robust_grasp_pose_cvg

        # WARNING: make sur the rounding must be done on 2 decimals (1 cm)!!! (handled in the sanity check)
        found_xyz_grasp_poses = np.around(found_xyz_grasp_poses, decimals=N_DECIMAL_PRECISION)

        grasps_in_rand_sample_references = np.array([
            tuple(xyz_pose) in self._all_possible_grasp_poses_to_fit for xyz_pose in found_xyz_grasp_poses
        ])

        found_xyz_grasp_poses_in_ref_set = found_xyz_grasp_poses[grasps_in_rand_sample_references]

        found_grasp_fitnesses = outcome_archive.get_successful_inds_fitnesses()
        found_grasp_fitnesses_in_ref_set = found_grasp_fitnesses[grasps_in_rand_sample_references]
        n_found_grasp_optimal_fitness = np.sum([
            fit == OPTIMAL_FITNESS for fit in found_grasp_fitnesses_in_ref_set
        ])

        grasp_pose_cvg = \
            len(found_xyz_grasp_poses_in_ref_set) / len(self._all_possible_grasp_poses_to_fit)
        robust_grasp_pose_cvg = n_found_grasp_optimal_fitness / self._n_possible_robust_grasp_poses

        return grasp_pose_cvg, robust_grasp_pose_cvg

    def _get_success_sparsity_1_cvg(self, outcome_archive):
        """Sparsity metric from (Mahler et al. 2016).

        Works but very slow. Replaced by a combined computation that mix sparsity_1 and sparsity_2.

        For each point of the theoretically plausible success archive, we get the xyz point in the current success
        archive that minimize the distance with the theoretical point.
        The sparsity is the result of exp( - largest_found_distance).

        Note: 2 limitations
        - it assumes we know all the possible grasps (here replace by xyz position of the theoretically plausible
        success archive;
        - it might be dominated by outliers.
        """

        if outcome_archive.get_n_successful_cells() == 0:
            return None

        all_bins_centroid_xyz_poses = outcome_archive.all_bins_centroid_xyz_poses
        success_archive_xyz_poses = outcome_archive.get_successful_inds_xyz_poses() #.tolist()

        k_tree = Nearest(n_neighbors=1, metric='minkowski')
        k_tree.fit(success_archive_xyz_poses)

        all_smallest_dists = []
        for cell_pose in all_bins_centroid_xyz_poses:
            query = [cell_pose]
            smallest_dist = k_tree.kneighbors(X=query)[0][0][0]
            all_smallest_dists.append(smallest_dist)
        success_archive_sparsity = np.exp(-max(all_smallest_dists))
        return success_archive_sparsity

    def _get_success_sparsity_2_cvg(self, outcome_archive):
        """Sparsity metric from (Eppner et al. 2019).

        Works but very slow. Replaced by a combined computation that mix sparsity_1 and sparsity_2.

        Same as sparsity_1, but by taking the mean of distances instead of the max one (than might be dominated by
        outlier).

        Note: 1 limitation
        - it assumes we know all the possible grasps (here replace by xyz position of the theoretically plausible
        success archive
        """

        if outcome_archive.get_n_successful_cells() == 0:
            return None

        all_bins_centroid_xyz_poses = outcome_archive.all_bins_centroid_xyz_poses
        success_archive_xyz_poses = outcome_archive.get_successful_inds_xyz_poses()  # .tolist()

        k_tree = Nearest(n_neighbors=1, metric='minkowski')
        k_tree.fit(success_archive_xyz_poses)

        all_smallest_dists = []
        for cell_pose in all_bins_centroid_xyz_poses:
            query = [cell_pose]
            smallest_dist = k_tree.kneighbors(X=query)[0][0][0]
            all_smallest_dists.append(smallest_dist)
        success_archive_sparsity = np.exp(-np.array(all_smallest_dists).mean())
        return success_archive_sparsity

    def _get_success_sparsity_1_2_cvg(self, outcome_archive):

        if outcome_archive.get_n_successful_cells() == 0:
            return None, None

        all_bins_centroid_xyz_poses = outcome_archive.all_bins_centroid_xyz_poses
        success_archive_xyz_poses = outcome_archive.get_successful_inds_xyz_poses() #.tolist()

        k_tree = Nearest(n_neighbors=1, metric='minkowski')
        k_tree.fit(success_archive_xyz_poses)

        # ~ 9 seconds each
        all_smallest_dists = [
            k_tree.kneighbors(X=[cell_pose])[0][0][0] for cell_pose in all_bins_centroid_xyz_poses
        ]
        # multiproccessing parallelized : even more slow

        success_archive_sparsity_1 = np.exp(-max(all_smallest_dists))
        success_archive_sparsity_2 = np.exp(-np.array(all_smallest_dists).mean())
        return success_archive_sparsity_1, success_archive_sparsity_2

    def _get_success_sparsity_3_cvg(self, outcome_archive):
        """Sparsity metric from (Murugan et al. 2022). Similar to knn-based novelty.

        Median of the distance of each xyz poses in the success archive to the k_nn_sparsity (=15) nearest neighbours.
        The higher this value, the sparser the success archive.
        """

        k_nn_sparsity = 15

        if outcome_archive.get_n_successful_cells() < k_nn_sparsity:
            return None

        xyz_poses = outcome_archive.get_successful_inds_xyz_poses()
        k_tree = Nearest(n_neighbors=k_nn_sparsity + 1, metric='minkowski')
        k_tree.fit(xyz_poses)

        n_samples = k_tree.n_samples_fit_
        query = xyz_poses
        if n_samples >= k_nn_sparsity + 1:
            neighbours_distances = k_tree.kneighbors(X=query)[0][:, 1:]
        else:
            neighbours_distances = k_tree.kneighbors(X=query, n_neighbors=n_samples)[0][:, 1:]

        avg_distances_per_inds = np.mean(neighbours_distances, axis=1)
        success_archive_sparsity = np.median(avg_distances_per_inds)

        return success_archive_sparsity

    def _get_success_sparsity_4_cvg(self, outcome_archive):
        """Standard deviation sparsity metric.
        """
        if outcome_archive.get_n_successful_cells() == 0:
            return None

        success_archive_sparsity = outcome_archive.get_successful_inds_xyz_poses().std()
        return success_archive_sparsity

    def _get_outcome_and_success_ratios(self, pop, curr_n_evals):

        pop_n_touched = np.sum(pop.infos[:, eval_cfg.IS_OBJ_TOUCHED_KEY_ID])
        pop_n_success = np.sum(pop.infos[:, eval_cfg.IS_SUCCESS_KEY_ID])
        pop_n_rollout = len(pop)

        self._rolling_n_touched += pop_n_touched
        self._rolling_n_success += pop_n_success
        self._rolling_n_rollout += pop_n_rollout
        assert curr_n_evals == self._rolling_n_rollout  # might cause issues if called twice for a single gen ?

        outcome_ratio = self._rolling_n_touched / self._rolling_n_rollout
        success_ratio = self._rolling_n_success / self._rolling_n_rollout

        return outcome_ratio, success_ratio

    def ending_analysis(self, success_archive):
        pass

    def get_output_data(self):
        output_data_kwargs = {
            'outcome_archive_cvg_hist': np.array(self._outcome_archive_cvg_hist),
            'success_archive_cvg_hist': np.array(self._success_archive_cvg_hist),
            'success_archive_qd_score_hist': np.array(self._success_archive_qd_score_hist),
            'grasp_poses_cvg_hist': self._grasp_pose_cvg_hist,
            'robust_grasp_poses_cvg_hist': self._robust_grasp_pose_cvg_hist,
            'success_archive_sparsity_3_hist': np.array(self._success_archive_sparsity_3_hist),
            'success_archive_sparsity_4_hist': np.array(self._success_archive_sparsity_4_hist),
            'outcome_ratio_hist': np.array(self._outcome_ratio_hist),
            'success_ratio_hist': np.array(self._success_ratio_hist),
            'n_evals_hist': np.array(self._n_evals),
            'n_evals_including_invalid_hist': np.array(self._n_evals_including_invalid),
            'run_time_hist': np.array(self._run_time),
        }
        return output_data_kwargs



