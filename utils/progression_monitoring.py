
import tqdm


class ProgressionMonitoring:
    def __init__(self, n_budget_rollouts):

        self._n_eval = 0  # number of evaluation
        
        self._t_success_archive = None  # tqdm verbose bar : number of successful individuals
        self._t_outcome_archive = None  # tqdm verbose bar : number of successful individuals
        self._n_eval_including_invalid_tqdm = None
        self._n_eval_tqdm = None  # tqdm ticks bar : number of evaluations

        self._init_tqdm_bars(n_budget_rollouts=n_budget_rollouts)

    @property
    def n_eval(self):
        return self._n_eval

    @property
    def n_eval_including_invalid(self):
        return self._n_eval_including_invalid_tqdm.n

    def _init_tqdm_bars(self, n_budget_rollouts):
        self._t_success_archive = tqdm.tqdm(
            total=float('inf'),
            leave=False,
            desc='Success archive size',
            bar_format='{desc}: {n_fmt}'
        )

        self._t_outcome_archive = tqdm.tqdm(
            total=float('inf'),
            leave=False,
            desc='Outcome archive size',
            bar_format='{desc}: {n_fmt}'
        )
        self._n_eval_including_invalid_tqdm = tqdm.tqdm(
            total=float('inf'),
            leave=False,
            desc='Number of evaluations (including invalid inds)',
            bar_format='{desc}: {n_fmt}'
        )
        self._n_eval_tqdm = tqdm.tqdm(
            range(n_budget_rollouts),
            ascii=True,
            desc='Number of evaluations'
        )

    def _update_verbose_bars(self, n_success=None, outcome_archive_len=None, n_evals_including_invalid2add=None):
        """Update tqdm attributes with the given values (values added to total count)"""
        
        if n_success is not None:
            self._t_success_archive.n = n_success
            self._t_success_archive.refresh()

        if outcome_archive_len is not None:
            self._t_outcome_archive.n = outcome_archive_len
            self._t_outcome_archive.refresh()

        if n_evals_including_invalid2add is not None:
            self._n_eval_including_invalid_tqdm.n += n_evals_including_invalid2add
            self._n_eval_including_invalid_tqdm.refresh()

    def update(self, pop, outcome_archive, n_evals, n_evals_including_invalid):

        n_eval2add = n_evals
        n_evals_including_invalid2add = n_evals_including_invalid

        self._n_eval += n_eval2add

        self._update_verbose_bars(
            n_success=outcome_archive.get_n_successful_cells(),
            outcome_archive_len=len(outcome_archive),
            n_evals_including_invalid2add=n_evals_including_invalid2add,
        )

        self._n_eval_tqdm.update(n_eval2add)


