
import pdb

import os
import numpy as np

from .elite_structured_archive import EliteStructuredArchive
from configs.qd_config import FillArchiveStrategy
import configs.eval_config as eval_cfg


class OutcomeArchive(EliteStructuredArchive):
    def __init__(self, search_space_bb, fill_archive_strat=FillArchiveStrategy.STRUCTURED_ELITES):
        super().__init__(search_space_bb=search_space_bb, fill_archive_strat=fill_archive_strat)

        self._it_export_success = 0
        self._at_least_one_success = False

    def get_n_successful_cells(self):
        return int(np.sum([self._map_infos[key][eval_cfg.IS_SUCCESS_KEY_ID] for key in self._map_infos]))

    def update(self, pop):
        self._add_inds(pop=pop)

    def _add_inds(self, pop):
        self.fill_elites(pop)

        if not self._at_least_one_success:
            # avoid useless for loops when checking for scs data (e.g. during export)
            self._at_least_one_success = np.sum(
                [self._map_infos[key][eval_cfg.IS_SUCCESS_KEY_ID] for key in self._map_infos]
            ) > 0

    def export(self, run_name, curr_neval, elapsed_time, verbose=False, only_scs=True):

        if self._at_least_one_success:
            inds, bds, fits, infos = self.get_successful_inds_data()
        else:
            inds, bds, fits, infos = np.array([]), np.array([]), np.array([]), np.array([])

        export_target_name = 'success_archives' if only_scs else 'outcome_archives'
        export_archive_path = str(run_name) + '/' + export_target_name
        if not os.path.isdir(export_archive_path):
            os.mkdir(export_archive_path)

        saving = {
            "inds": inds,
            "behavior_descriptors": bds,
            "fitnesses": fits,
            'infos': infos,
            'infos_keys': eval_cfg.INFO_KEYS,
            "nevals": curr_neval,
            "elapsed_time": elapsed_time,
        }

        it_export = self._it_export_success if only_scs else self._it_export
        np.savez_compressed(file=export_archive_path + f'/individuals_{it_export}', **saving)

        if verbose:
            print(f'{export_target_name} n°{it_export} has been successfully dumped at {export_archive_path}.')

        if only_scs:
            self._it_export_success += 1
        else:
            self._it_export += 1


