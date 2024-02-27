import glob

import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

from algorithms.population import Population
from algorithms.archives.novelty_archive import NoveltyArchive
from algorithms.archives.elite_structured_archive import EliteStructuredArchive
from algorithms.archives.dummy_archive import DummyArchive
from algorithms.archives.outcome_archive import OutcomeArchive

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from visualization.vis_tools import load_all_inds, load_all_infos, get_info_from_key, load_all_fitnesses, \
    load_run_output_files, get_folder_path_from_str, get_all_6dof_grasping_data
from utils.timer import is_running_timeout

import configs.qd_config as qd_cfg


def select_offspring_standard(pop, pop_size, archive, select_off_strat):

    if select_off_strat == qd_cfg.SelectOffspringStrategy.RANDOM_FROM_POP:
        off_inds = pop.random_sample(n_sample=pop_size)

    elif select_off_strat == qd_cfg.SelectOffspringStrategy.RANDOM_FROM_ARCHIVE:
        off_inds = archive.random_sample(n_sample=pop_size)

    elif select_off_strat == qd_cfg.SelectOffspringStrategy.FITNESS_FROM_ARCHIVE:
        off_inds = archive.select_fitness_based(n_sample=pop_size)

    elif select_off_strat == qd_cfg.SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE:
        off_inds = archive.select_success_based(n_sample=pop_size, duplicate_scs_inds=False)

    elif select_off_strat == qd_cfg.SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE_WITH_DUPLICATES:
        off_inds = archive.select_success_based(n_sample=pop_size, duplicate_scs_inds=True)

    else:
        raise NotImplementedError

    return off_inds



def select_offspring_routine(
        pop, select_off_strat, archive, pop_size, genotype_len, sigma_mut, **kwargs
):

    off_inds = select_offspring_standard(
        pop=pop, pop_size=pop_size, archive=archive, select_off_strat=select_off_strat
    )

    off = Population(inds=off_inds, len_pop=pop_size, len_genotype=genotype_len, sigma_mut=sigma_mut)

    return off


def mutate_offspring(off, mut_strat):

    if mut_strat == qd_cfg.MutationStrategy.GAUSS:
        off.mutate_gauss()

    elif mut_strat == qd_cfg.MutationStrategy.NONE:
        pass

    else:
        raise AttributeError(f'Unknown mutation flag: mut_strat={mut_strat}')


def init_archive(archive_type, archive_kwargs):

    if archive_type == qd_cfg.ArchiveType.NOVELTY:
        archive_class = NoveltyArchive

    elif archive_type == qd_cfg.ArchiveType.ELITE_STRUCTURED:
        archive_class = EliteStructuredArchive

    elif archive_type == qd_cfg.ArchiveType.NONE:
        archive_class = DummyArchive
    else:
        raise NotImplementedError

    return archive_class(**archive_kwargs)


def re_evaluate_until_off_is_full_of_valid_inds(pop, off, archive, fixed_attr_dict, timer, **kwargs):
    n_targeted_valid_inds = len(off)
    n_evaluated_valid_inds = int(sum(off.are_valid_inds()))

    valid_inds, valid_bds, valid_fits, valid_infos = [], [], [], []
    additionnal_n_evals = 0
    run_timeout_flg = False

    is_there_valid_inds = n_evaluated_valid_inds > 0
    if is_there_valid_inds:
        valid_inds += off.inds[off.are_valid_inds()].tolist()
        valid_bds += off.bds[off.are_valid_inds()].tolist()
        valid_fits += off.fits[off.are_valid_inds()].tolist()
        valid_infos += off.infos[off.are_valid_inds()].tolist()

    while n_evaluated_valid_inds < n_targeted_valid_inds and not run_timeout_flg:

        valid_finder_off = select_offspring_routine(
            pop=pop,
            archive=archive,
            **fixed_attr_dict
        )
        mutate_offspring(
            off=valid_finder_off,
            mut_strat=kwargs['mut_strat'],
        )
        valid_finder_off.evaluate(
            evaluate_fn=kwargs['evaluation_function'], multiproc_pool=kwargs['multiproc_pool']
        )
        additionnal_n_evals += len(valid_finder_off)

        is_there_valid_inds = sum(valid_finder_off.are_valid_inds()) > 0
        if is_there_valid_inds:
            valid_inds += valid_finder_off.inds[valid_finder_off.are_valid_inds()].tolist()
            valid_bds += valid_finder_off.bds[valid_finder_off.are_valid_inds()].tolist()
            valid_fits += valid_finder_off.fits[valid_finder_off.are_valid_inds()].tolist()
            valid_infos += valid_finder_off.infos[valid_finder_off.are_valid_inds()].tolist()

            n_evaluated_valid_inds += int(sum(valid_finder_off.are_valid_inds()))

        run_timeout_flg = is_running_timeout(timer=timer, label=qd_cfg.QD_RUN_TIME_LABEL)

    off.inds = np.array(valid_inds[:n_targeted_valid_inds])
    off.bds = np.array(valid_bds[:n_targeted_valid_inds])
    off.fits = np.array(valid_fits[:n_targeted_valid_inds])
    off.infos = np.array(valid_infos[:n_targeted_valid_inds])

    return additionnal_n_evals, run_timeout_flg


def re_evaluate_until_pop_is_full_of_valid_inds(pop, pop_size, genotype_len, sigma_mut, evaluate_fn, multiproc_pool, timer):
    n_targeted_valid_inds = len(pop)
    n_evaluated_valid_inds = int(sum(pop.are_valid_inds()))

    valid_inds, valid_bds, valid_fits, valid_infos = [], [], [], []
    additionnal_n_evals = 0
    run_timeout_flg = False

    is_there_valid_inds = n_evaluated_valid_inds > 0
    if is_there_valid_inds:
        valid_inds += pop.inds[pop.are_valid_inds()].tolist()
        valid_bds += pop.bds[pop.are_valid_inds()].tolist()
        valid_fits += pop.fits[pop.are_valid_inds()].tolist()
        valid_infos += pop.infos[pop.are_valid_inds()].tolist()

    while n_evaluated_valid_inds < n_targeted_valid_inds and not run_timeout_flg:
        #print('(init) n_evaluated_valid_inds=', n_evaluated_valid_inds)
        valid_finder_pop = Population(len_pop=pop_size, len_genotype=genotype_len, sigma_mut=sigma_mut)
        valid_finder_pop.init_inds()
        valid_finder_pop.evaluate(evaluate_fn=evaluate_fn, multiproc_pool=multiproc_pool)
        additionnal_n_evals += len(valid_finder_pop)

        is_there_valid_inds = sum(valid_finder_pop.are_valid_inds()) > 0
        if is_there_valid_inds:
            valid_inds += valid_finder_pop.inds[valid_finder_pop.are_valid_inds()].tolist()
            valid_bds += valid_finder_pop.bds[valid_finder_pop.are_valid_inds()].tolist()
            valid_fits += valid_finder_pop.fits[valid_finder_pop.are_valid_inds()].tolist()
            valid_infos += valid_finder_pop.infos[valid_finder_pop.are_valid_inds()].tolist()

            n_evaluated_valid_inds += int(sum(valid_finder_pop.are_valid_inds()))

        run_timeout_flg = is_running_timeout(timer=timer, label=qd_cfg.QD_RUN_TIME_LABEL)

    pop.inds = np.array(valid_inds[:n_targeted_valid_inds])
    pop.bds = np.array(valid_bds[:n_targeted_valid_inds])
    pop.fits = np.array(valid_fits[:n_targeted_valid_inds])
    pop.infos = np.array(valid_infos[:n_targeted_valid_inds])

    return additionnal_n_evals, run_timeout_flg


def initialize_evo_process_from_scratch(
        pop,
        evaluate_fn,
        multiproc_pool,
        timer,
        include_invalid_inds,
        pop_size,
        genotype_len,
        sigma_mut,
        is_novelty_required,
        archive,
):
    pop.init_inds()
    pop.evaluate(evaluate_fn=evaluate_fn, multiproc_pool=multiproc_pool)
    n_evals = n_evals_including_invalid = len(pop)
    run_timeout_flg = is_running_timeout(timer=timer, label=qd_cfg.QD_RUN_TIME_LABEL)

    if not include_invalid_inds:
        additionnal_n_evals, run_timeout_flg = re_evaluate_until_pop_is_full_of_valid_inds(
            pop=pop,
            pop_size=pop_size,
            genotype_len=genotype_len,
            sigma_mut=sigma_mut,
            evaluate_fn=evaluate_fn,
            multiproc_pool=multiproc_pool,
            timer=timer
        )
        n_evals_including_invalid += additionnal_n_evals

        if run_timeout_flg:
            print(f'Timeout during pop init. End of init routine then exit.')

    if is_novelty_required:
        pop.compute_novelty(archive=archive)

    return pop, n_evals, n_evals_including_invalid, run_timeout_flg


def initialize_pop_and_archive(
    pop_size,
    genotype_len,
    sigma_mut,
    archive_type,
    archive_kwargs,
    outcome_archive_kwargs,
    evaluate_fn,
    is_novelty_required,
    progression_monitoring,
    stats_tracker,
    multiproc_pool,
    timer,
    timer_label,
    include_invalid_inds,
    **kwargs,
):

    pop = Population(len_pop=pop_size, len_genotype=genotype_len, sigma_mut=sigma_mut)
    archive = init_archive(archive_type, archive_kwargs)
    outcome_archive = OutcomeArchive(**outcome_archive_kwargs)

    pop, n_evals, n_evals_including_invalid, run_timeout_flg = initialize_evo_process_from_scratch(
        pop=pop,
        evaluate_fn=evaluate_fn,
        multiproc_pool=multiproc_pool,
        timer=timer,
        include_invalid_inds=include_invalid_inds,
        pop_size=pop_size,
        genotype_len=genotype_len,
        sigma_mut=sigma_mut,
        is_novelty_required=is_novelty_required,
        archive=archive,
    )

    archive.fill(pop=pop)
    outcome_archive.update(pop)

    progression_monitoring.update(
        pop=pop, outcome_archive=outcome_archive, n_evals=n_evals, n_evals_including_invalid=n_evals_including_invalid
    )

    stats_tracker.update(
        pop=pop,
        outcome_archive=outcome_archive,
        curr_n_evals=progression_monitoring.n_eval,
        curr_n_evals_including_invalid=progression_monitoring.n_eval_including_invalid,
        timer=timer,
        timer_label=timer_label
    )

    return pop, archive, outcome_archive, progression_monitoring, stats_tracker, run_timeout_flg


def fill_archive_routine(archive, off):
    archive.fill(pop=off)
    archive.manage_archive_size()
    return


def replace_pop(pop, replace_pop_strat, ref_pop_inds=None, **kwargs):

    # Selection & replacement
    if replace_pop_strat == qd_cfg.ReplacePopulationStrategy.RANDOM:
        pop.replace_random_sample(src_pop=ref_pop_inds)

    elif replace_pop_strat == qd_cfg.ReplacePopulationStrategy.NOVELTY_BASED:
        pop.replace_novelty_based(src_pop=ref_pop_inds)

    elif replace_pop_strat == qd_cfg.ReplacePopulationStrategy.RESET_FROM_SCRATCH:
        pop.init_inds()

    else:
        raise AttributeError


def update_novelties(pop, off, archive, ref_pop, pop_size):
    ref_pop.compute_novelty(archive=archive)
    pop.novs = ref_pop.novs[:pop_size]
    off.novs = ref_pop.novs[pop_size:]
    assert len(pop.novs) == len(pop.inds)
    assert len(off.novs) == len(off.inds)


def init_fixed_attr_dict(kwargs):
    return {
        'pop_size': kwargs['pop_size'],
        'genotype_len': kwargs['genotype_len'],
        'sigma_mut': kwargs['sigma_mut'],
        'archive_type': kwargs['archive_type'],
        'qd_method': kwargs['qd_method'],
        'archive_kwargs': kwargs['archive_kwargs'],
        'outcome_archive_kwargs': kwargs['outcome_archive_kwargs'],
        'evaluate_fn': kwargs['evaluation_function'],
        'is_novelty_required': kwargs['is_novelty_required'],
        'nb_offsprings_to_generate': kwargs['nb_offsprings_to_generate'],
        'mut_strat': kwargs['mut_strat'],
        'prob_cx': kwargs['prob_cx'],
        'robot_name': kwargs['robot_name'],
        'stabilized_obj_pose': kwargs['stabilized_obj_pose'],
        'parallelize': kwargs['parallelize'],
        'select_off_strat': kwargs['select_off_strat'],
        'replace_pop_strat': kwargs['replace_pop_strat'],
        'include_invalid_inds': kwargs['include_invalid_inds'],
    }


