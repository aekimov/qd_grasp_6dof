

from .archive import Archive
from configs.qd_config import FillArchiveStrategy


class DummyArchive(Archive):
    """Used for archive-less methods, for code consistency. This archive has no effect, but avoid errors while calling
    archive specific methods."""

    def __init__(self, search_space_bb, fill_archive_strat=FillArchiveStrategy.NONE):
        super().__init__(fill_archive_strat=fill_archive_strat)

    def manage_archive_size(self):
        pass

