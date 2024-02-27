import pdb

import numpy as np

from .unstructured_archive import UnstructuredArchive


class NoveltyArchive(UnstructuredArchive):

    def __init__(self, archive_limit_size, archive_limit_strat, pop_size, fill_archive_strat, search_space_bb):

        super().__init__(
            archive_limit_size=archive_limit_size,
            archive_limit_strat=archive_limit_strat,
            pop_size=pop_size,
            fill_archive_strat=fill_archive_strat,
        )

    def fill_novelty_based(self, pop):
        """Fill the archive with the most novel individuals."""
        novelties = pop.novs
        max_novelties_idx = np.argsort(-novelties)[:self._nb2add_per_update]
        self._inds += [pop.inds[ind_id] for ind_id in max_novelties_idx]
        self._bds += [pop.bds[ind_id] for ind_id in max_novelties_idx]



