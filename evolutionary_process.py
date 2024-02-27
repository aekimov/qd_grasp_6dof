
from algorithms.stats.stats_tracker import StatsTracker
from utils.progression_monitoring import ProgressionMonitoring
from utils.evo_proc_routines import mutate_offspring, select_offspring_routine, init_fixed_attr_dict, \
    initialize_pop_and_archive, update_novelties, fill_archive_routine, replace_pop, \
    re_evaluate_until_off_is_full_of_valid_inds
from utils.io_run_data import dump_archive_success_routine, export_running_data_routine
from utils.timer import Timer, is_running_timeout


from algorithms.pyribs_qd_interface import run_qd_pyribs

import configs.qd_config as qd_cfg
import configs.exec_config as ex_cfg


def run_qd_core(**kwargs):
    timer = Timer()
    timer.start(label=qd_cfg.QD_RUN_TIME_LABEL)
    stats_tracker = StatsTracker(
        qd_method=kwargs['qd_method'],
        targeted_object=kwargs['scene_details']['targeted_object'],
        robot_name_str=kwargs['robot_name_str']
    )
    progression_monitoring = ProgressionMonitoring(n_budget_rollouts=kwargs['n_budget_rollouts'])
    ref_pop = None

    fixed_attr_dict = init_fixed_attr_dict(kwargs)

    # ---------------------------- EVOLUTIONARY PROCESS INITIALIZATION ------------------------------- #
    pop, archive, outcome_archive, progression_monitoring, stats_tracker, run_timeout_flg = \
        initialize_pop_and_archive(
            progression_monitoring=progression_monitoring,
            stats_tracker=stats_tracker,
            multiproc_pool=kwargs['multiproc_pool'],
            timer=timer,
            timer_label=qd_cfg.QD_RUN_TIME_LABEL,
            **fixed_attr_dict
        )

    # ---------------------------------------- BEGIN EVOLUTION --------------------------------------- #
    gen = 0
    do_iterate_gen = True if not run_timeout_flg else False

    while do_iterate_gen:
        gen += 1

        # -------------------------------------- GENERATE OFFPRING ----------------------------------- #

        off = select_offspring_routine(
            pop=pop,
            archive=archive,
            **fixed_attr_dict
        )

        # ------------------------------------------ MUTATE ---------------------------------------- #

        mutate_offspring(
            off=off,
            mut_strat=kwargs['mut_strat'],
        )

        # ------------------------------------------ EVALUATE -------------------------------------- #

        off.evaluate(evaluate_fn=kwargs['evaluation_function'], multiproc_pool=kwargs['multiproc_pool'])
        n_evals = n_evals_including_invalid = len(off)

        # ------------------------ RE-EVALUATE IF NECESSARY ------------------------ #
        if not kwargs['include_invalid_inds']:
            additionnal_n_evals, run_timeout_flg = re_evaluate_until_off_is_full_of_valid_inds(
                pop=pop, off=off, archive=archive, fixed_attr_dict=fixed_attr_dict, timer=timer, **kwargs
            )
            n_evals_including_invalid += additionnal_n_evals

            if run_timeout_flg:
                do_iterate_gen = False
                print(f'Timeout during offspring evals. End of running.')
                continue  # end of the evolutionary process

        outcome_archive.update(off)

        # ---------------------------------- UPDATE ROLLING VARIABLES ------------------------------ #

        progression_monitoring.update(
            pop=off, outcome_archive=outcome_archive, n_evals=n_evals,
            n_evals_including_invalid=n_evals_including_invalid
        )

        # ----------------------------------- UPDATE REFERENCE POP --------------------------------- #

        if kwargs['is_pop_based'] and kwargs['is_novelty_required']:
            ref_pop = pop + off

        # --------------------------------------- UPDATE NOVELTY ----------------------------------- #

        if kwargs['is_novelty_required']:
            update_novelties(pop=pop, off=off, archive=archive, ref_pop=ref_pop, pop_size=kwargs['pop_size'])

        # --------------------------------------- UPDATE ARCHIVE ----------------------------------- #

        fill_archive_routine(archive=archive, off=off)

        # -------------------------------- NEXT GENERATION PARENTS --------------------------------- #

        if kwargs['is_pop_based']:
            replace_pop(
                pop=pop,
                ref_pop_inds=ref_pop,
                **fixed_attr_dict
            )

        # ----------------------------------------- MEASURE ---------------------------------------- #

        stats_tracker.update(
            pop=off, outcome_archive=outcome_archive, curr_n_evals=progression_monitoring.n_eval,
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

        # --------------------------------- CHECK EVALUATION BUDGET -------------------------------- #

        if progression_monitoring.n_eval > kwargs['n_budget_rollouts']:
            print(f'curr_n_eval={progression_monitoring.n_eval} > n_budget_rollouts={kwargs["n_budget_rollouts"]} | '
                  f'End of running.')
            do_iterate_gen = False
            continue  # end of the evolutionary process

        # ---------------------------------- CHECK RUN TIME BUDGET --------------------------------- #

        if is_running_timeout(timer=timer, label=qd_cfg.QD_RUN_TIME_LABEL):
            elapsed_time = timer.get_on_the_fly_time(qd_cfg.QD_RUN_TIME_LABEL)
            print(f'elapsed_time={elapsed_time} > ex_cfg.MAX_RUN_TIME_IN_S={ex_cfg.MAX_RUN_TIME_IN_S} | '
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


def run_qd(**kwargs):

    if kwargs['qd_method'] in qd_cfg.PYRIBS_QD_METHODS:
        success_archive_len = run_qd_pyribs(**kwargs)
    else:
        success_archive_len = run_qd_core(**kwargs)

    return success_archive_len



