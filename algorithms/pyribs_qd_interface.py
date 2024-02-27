

from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from algorithms.archives.elite_structured_archive import EliteStructuredArchive

import numpy as np

from algorithms.stats.stats_tracker import StatsTracker
from utils.progression_monitoring import ProgressionMonitoring
from utils.evo_proc_routines import init_fixed_attr_dict
from utils.io_run_data import dump_archive_success_routine, export_running_data_routine
from utils.timer import Timer, is_running_timeout

from algorithms.population import Population
from algorithms.archives.outcome_archive import OutcomeArchive

import configs.qd_config as qd_cfg


def get_cma_mae_pyribs_learning_rate(qd_method):
    if qd_method in qd_cfg.CMA_MAE_QD_METHODS:
        return qd_cfg.CMA_MAE_PREDEFINED_ALPHA
    elif qd_method in qd_cfg.CMA_ME_QD_METHODS:
        return qd_cfg.CMA_ME_PREDEFINED_ALPHA
    elif qd_method in qd_cfg.CMA_ES_QD_METHODS:
        return qd_cfg.CMA_ES_PREDEFINED_ALPHA
    else:
        raise NotImplementedError()


def init_archive_pyribs(solution_dim, qd_method, archive_kwargs):
    lr = get_cma_mae_pyribs_learning_rate(qd_method)

    # Trick to get compute the grid cell similarly to the core qd pipeline
    dummy_structured_archive = EliteStructuredArchive(**archive_kwargs)
    dummy_sa_dims = dummy_structured_archive.get_dims()

    archive_kwargs = {
        'solution_dim': solution_dim,
        'dims': (dummy_sa_dims['n_bins_x'], dummy_sa_dims['n_bins_y'], dummy_sa_dims['n_bins_z']),
        'ranges': [
            (dummy_sa_dims['min_x'], dummy_sa_dims['max_x']),
            (dummy_sa_dims['min_y'], dummy_sa_dims['max_y']),
            (dummy_sa_dims['min_z'], dummy_sa_dims['max_z'])
        ],
        'learning_rate': lr,
        'threshold_min': -1.0  # fitness > 0 => defining f_thresh < f_min is suggested by cma_mae authors
    }
    return GridArchive(**archive_kwargs)


def evaluate_pyribs(self, evaluate_fn, multiproc_pool):
    assert self._inds is not None

    if multiproc_pool is not None:
        evaluation_pop = list(multiproc_pool.map(evaluate_fn, self._inds))
    else:
        evaluation_pop = list(map(evaluate_fn, self._inds))

    self._bds, self._fits, self._infos = map(np.array, zip(*evaluation_pop))


def evaluate_and_update_pop_pyribs(evaluate_fn, multiproc_pool, pop):
    b_descriptors, is_scs_fitnesses, infos = evaluate_grasp_pyribs(
        evaluate_fn=evaluate_fn, multiproc_pool=multiproc_pool, inds=pop.inds
    )
    pop.update_individuals(
        fitnesses=is_scs_fitnesses,
        b_descriptors=b_descriptors,
        infos=infos,
    )

    objective_batch = is_scs_fitnesses
    measure_batch = b_descriptors
    infos_batch = infos

    return objective_batch, measure_batch, infos_batch


def evaluate_grasp_pyribs(evaluate_fn, multiproc_pool, inds):
    assert inds is not None

    if multiproc_pool is not None:
        evaluation_pop = list(multiproc_pool.map(evaluate_fn, inds))
    else:
        evaluation_pop = list(map(evaluate_fn, inds))

    b_descriptors, is_scs_fitnesses, infos = map(np.array, zip(*evaluation_pop))

    objective_batch = is_scs_fitnesses
    measure_batch = b_descriptors
    infos_batch = infos

    return measure_batch, objective_batch, infos_batch


def re_evaluate_until_pop_is_full_of_valid_inds_pyribs(pop, scheduler, timer, objective_batch, measure_batch, **kwargs):
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

        # Warning: each subsample affects the next one ; pyribs prevent several successive calls
        scheduler.tell(objective_batch, measure_batch)

        solution_batch = scheduler.ask()
        solution_batch = solution_batch.clip(min=-qd_cfg.GENOTYPE_MAX_VAL, max=qd_cfg.GENOTYPE_MAX_VAL)
        valid_finder_pop = Population(
            len_pop=len(solution_batch),
            len_genotype=kwargs['genotype_len'],
            sigma_mut=kwargs['sigma_mut'],
            inds=solution_batch
        )

        objective_batch, measure_batch, infos_batch = evaluate_and_update_pop_pyribs(
            evaluate_fn=kwargs['evaluation_function'], multiproc_pool=kwargs['multiproc_pool'], pop=valid_finder_pop,
        )

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


def run_qd_pyribs(**kwargs):
    timer = Timer()
    timer.start(label=qd_cfg.QD_RUN_TIME_LABEL)
    stats_tracker = StatsTracker(
        qd_method=kwargs['qd_method'],
        targeted_object=kwargs['scene_details']['targeted_object'],
        robot_name_str=kwargs['robot_name_str']
    )
    progression_monitoring = ProgressionMonitoring(n_budget_rollouts=kwargs['n_budget_rollouts'])

    fixed_attr_dict = init_fixed_attr_dict(kwargs)

    outcome_archive = OutcomeArchive(**fixed_attr_dict['outcome_archive_kwargs'])

    archive = init_archive_pyribs(
        solution_dim=fixed_attr_dict['genotype_len'],
        qd_method=fixed_attr_dict['qd_method'],
        archive_kwargs=fixed_attr_dict['archive_kwargs']
    )
    random_init_solution = np.random.uniform(
        low=-qd_cfg.GENOTYPE_MAX_VAL,
        high=qd_cfg.GENOTYPE_MAX_VAL,
        size=fixed_attr_dict['genotype_len']
    )

    # ---------------------------- EVOLUTIONARY PROCESS INITIALIZATION ------------------------------- #

    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=random_init_solution,  # initial solution (must be a single ind)
            sigma0=0.5,
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=qd_cfg.CMA_MAE_EMITTER_BATCH_SIZE,
        ) for _ in range(qd_cfg.CMA_MAE_N_EMITTERS)
    ]
    scheduler = Scheduler(archive, emitters)

    run_timeout_flg = is_running_timeout(timer=timer, label=qd_cfg.QD_RUN_TIME_LABEL)

    # ---------------------------------------- BEGIN EVOLUTION --------------------------------------- #
    gen = 0
    do_iterate_gen = True if not run_timeout_flg else False

    while do_iterate_gen:
        gen += 1
        # -------------------------------------- GENERATE OFFPRING ----------------------------------- #

        solution_batch = scheduler.ask()
        solution_batch = solution_batch.clip(min=-qd_cfg.GENOTYPE_MAX_VAL, max=qd_cfg.GENOTYPE_MAX_VAL)
        pop = Population(
            len_pop=len(solution_batch),
            len_genotype=kwargs['genotype_len'],
            sigma_mut=kwargs['sigma_mut'],
            inds=solution_batch
        )

        # ------------------------------------------ EVALUATE -------------------------------------- #

        objective_batch, measure_batch, infos_batch = evaluate_and_update_pop_pyribs(
            evaluate_fn=kwargs['evaluation_function'], multiproc_pool=kwargs['multiproc_pool'], pop=pop,
        )
        n_evals = n_evals_including_invalid = len(pop)

        # ------------------------ RE-EVALUATE IF NECESSARY ------------------------ #
        if not kwargs['include_invalid_inds']:
            additionnal_n_evals, run_timeout_flg = re_evaluate_until_pop_is_full_of_valid_inds_pyribs(
                pop=pop,
                scheduler=scheduler,
                timer=timer,
                objective_batch=objective_batch,
                measure_batch=measure_batch,
                **kwargs
            )
            n_evals_including_invalid += additionnal_n_evals

            if run_timeout_flg:
                do_iterate_gen = False
                print(f'Timeout during offspring evals. End of running.')
                continue  # end of the evolutionary process

        outcome_archive.update(pop)

        # ---------------------------------- UPDATE ROLLING VARIABLES ------------------------------ #

        progression_monitoring.update(
            pop=pop, outcome_archive=outcome_archive, n_evals=n_evals,
            n_evals_including_invalid=n_evals_including_invalid
        )

        # ----------------------------------------- MEASURE ---------------------------------------- #

        stats_tracker.update(
            pop=pop, outcome_archive=outcome_archive, curr_n_evals=progression_monitoring.n_eval,
            curr_n_evals_including_invalid=progression_monitoring.n_eval_including_invalid,
            timer=timer, timer_label=qd_cfg.QD_RUN_TIME_LABEL)

        # --------------------------------- SUCCESS ARCHIVE DUMPING -------------------------------- #

        do_dump_scs_archive = qd_cfg.DUMP_SCS_ARCHIVE_ON_THE_FLY and gen % qd_cfg.N_GEN_FREQ_DUMP_SCS_ARCHIVE == 0
        if do_dump_scs_archive:
            dump_archive_success_routine(
                timer=timer,
                timer_label=qd_cfg.QD_RUN_TIME_LABEL,
                run_name=kwargs['run_name'],
                curr_neval=progression_monitoring.n_eval,
                outcome_archive=outcome_archive,
            )

        scheduler.tell(objective_batch, measure_batch)

        # --------------------------------- CHECK EVALUATION BUDGET -------------------------------- #

        if progression_monitoring.n_eval > kwargs['n_budget_rollouts']:
            print(f'curr_n_eval={progression_monitoring.n_eval} > n_budget_rollouts={kwargs["n_budget_rollouts"]} | '
                  f'End of running.')
            do_iterate_gen = False
            continue  # end of the evolutionary process

        pass  # end of generation

    print('\n\nEnd of running.')

    print('\nLast dump of success archive...')
    dump_archive_success_routine(
        timer=timer,
        timer_label=qd_cfg.QD_RUN_TIME_LABEL,
        run_name=kwargs['run_name'],
        curr_neval=progression_monitoring.n_eval,
        outcome_archive=outcome_archive,
        is_last_call=True,
    )
    print(f'\nLatest success archive has been successfully dumped to: {kwargs["run_name"]}')

    timer.stop(label=qd_cfg.QD_RUN_TIME_LABEL)

    success_archive_len = outcome_archive.get_n_successful_cells()
    run_infos = {
        f'Number of triggered outcome cells (outcome_archive len, out of {outcome_archive.max_size})':
            len(outcome_archive),
        'Number of successful individuals (outcome_archive get_n_successful_cells)': success_archive_len,
        'Number of evaluations (progression_monitoring.n_eval)': progression_monitoring.n_eval,
        'Number of evaluations including invalid individuals (progression_monitoring.n_eval_including_invalid)':
            progression_monitoring.n_eval_including_invalid,
        'elapsed_time': timer.get_all(format='h:m:s'),
        'Number of computed generations (gen)': gen,
        'first_saved_ind_gen': stats_tracker.first_saved_ind_gen,
        'first_saved_ind_n_evals': stats_tracker.first_saved_ind_n_evals,
        'outcome_archive_dims': outcome_archive.get_dims(),
        'is running timeout': run_timeout_flg,
    }

    export_running_data_routine(
        stats_tracker=stats_tracker,
        run_name=kwargs['run_name'],
        run_details=kwargs['run_details'],
        run_infos=run_infos
    )

    return success_archive_len

