
import numpy as np
import random

from utils.novelty_computation import assess_novelties
import configs.qd_config as qd_cfg
import configs.eval_config as eval_cfg


class Population:
    def __init__(self, len_pop, len_genotype, sigma_mut, inds=None):

        self._inds = None
        self._bds = None
        self._fits = None
        self._novs = None
        self._infos = None

        self._len_pop = None
        self._len_genotype = None
        self._sigma_mut = None

        self._init_attrs(len_pop=len_pop, len_genotype=len_genotype, sigma_mut=sigma_mut, inds=inds)

    @property
    def inds(self):
        return self._inds

    @inds.setter
    def inds(self, inds):
        self._inds = inds

    @property
    def bds(self):
        return self._bds

    @bds.setter
    def bds(self, bds):
        self._bds = bds

    @property
    def fits(self):
        return self._fits

    @fits.setter
    def fits(self, fits):
        self._fits = fits

    @property
    def novs(self):
        return self._novs

    @novs.setter
    def novs(self, novs):
        self._novs = novs

    @property
    def infos(self):
        return self._infos

    @infos.setter
    def infos(self, infos):
        self._infos = infos

    @property
    def len_pop(self):
        return self._len_pop

    @len_pop.setter
    def len_pop(self, len_pop):
        self._len_pop = len_pop

    @property
    def sigma_mut(self):
        return self._sigma_mut

    def __len__(self):
        return self._len_pop

    def __add__(self, other):
        assert self._sigma_mut == other.sigma_mut
        out_pop = Population(
            len_pop=self._len_pop + len(other),
            len_genotype=self._len_genotype,
            sigma_mut=self._sigma_mut,
        )
        out_pop.inds = np.concatenate((self._inds, other.inds), axis=0)
        out_pop.bds = np.concatenate((self._bds, other.bds), axis=0)
        out_pop.fits = np.concatenate((self._fits, other.fits), axis=0)
        out_pop.infos = np.concatenate((self._infos, other.infos), axis=0)
        out_pop.novs = None

        return out_pop

    def get_successul_inds(self):
        return self._inds[self._infos[:, eval_cfg.IS_SUCCESS_KEY_ID]]

    def are_valid_inds(self):
        return self._infos[:, eval_cfg.IS_VALID_KEY_ID].astype(bool)

    def _init_attrs(self, len_pop, len_genotype, sigma_mut, inds):
        self._len_pop = len_pop
        self._len_genotype = len_genotype
        self._sigma_mut = sigma_mut

        if inds is not None:
            # Initialize the pop inds with the provided genome. It automatically creates a copy of the inds in a new
            # matrix: the offspring can safely be mutated without modifying the parents.
            self._inds = np.array(inds)

    def init_inds(self):
        self._inds = np.random.uniform(
            low=-qd_cfg.GENOTYPE_MAX_VAL,
            high=qd_cfg.GENOTYPE_MAX_VAL,
            size=(self._len_pop, self._len_genotype)
        )

    def set_inds(self, inds):
        self._inds = inds.copy()
        self._bds = None
        self._fits = None
        self._novs = None
        self._infos = None

        self._len_pop = len(inds)
        # note: _len_pop must be updated to take the right number of evalution into account
        # note: _len_genotype and _sigma_mut remains constant.

    def evaluate(self, evaluate_fn, multiproc_pool):
        assert self._inds is not None

        if multiproc_pool is not None:
            evaluation_pop = list(multiproc_pool.map(evaluate_fn, self._inds))
        else:
            evaluation_pop = list(map(evaluate_fn, self._inds))

        self._bds, self._fits, self._infos = map(np.array, zip(*evaluation_pop))

    def compute_novelty(self, archive):
        self._novs = assess_novelties(
            pop=self,
            archive=archive,
        )

    def mutate_gauss(self):
        self._inds += np.random.normal(loc=0, scale=self._sigma_mut, size=self._inds.shape)
        self._inds = self._inds.clip(min=-qd_cfg.GENOTYPE_MAX_VAL, max=qd_cfg.GENOTYPE_MAX_VAL)

    def random_sample(self, n_sample):
        id_selected = random.sample(range(len(self._inds)), n_sample)
        selected_inds = self._inds[id_selected]
        return selected_inds

    def replace_random_sample(self, src_pop):
        id_selected = random.sample(range(len(src_pop)), self._len_pop)
        self._inds = src_pop.inds[id_selected]
        self._bds = src_pop.bds[id_selected]
        self._fits = src_pop.fits[id_selected]
        self._infos = src_pop.infos[id_selected]
        self._novs = None

    def replace_novelty_based(self, src_pop):

        id_sorted_nov = np.argsort(src_pop.novs)[::-1]
        id_selected = id_sorted_nov[:self._len_pop]

        self._inds = src_pop.inds[id_selected]
        self._bds = src_pop.bds[id_selected]
        self._fits = src_pop.fits[id_selected]
        self._infos = src_pop.infos[id_selected]
        self._novs = None

    def update_individuals(self, fitnesses, b_descriptors, infos):
        """Update individuals with the given fitnesses, bds and infos."""
        self._bds, self._fits, self._infos = b_descriptors, fitnesses, infos



