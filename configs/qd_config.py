
from enum import Enum


########################################################################################################################
#  CONSTANTS
########################################################################################################################


# -------------------------------------------------------------------------------------------------------------------- #
# GENOME
# -------------------------------------------------------------------------------------------------------------------- #

GENOTYPE_MAX_VAL = 1.
FIXED_INTERVAL_GENOME = [-GENOTYPE_MAX_VAL, GENOTYPE_MAX_VAL]


# -------------------------------------------------------------------------------------------------------------------- #
# GENOME LENGTH
# -------------------------------------------------------------------------------------------------------------------- #

# Standard direct search
GENOTYPE_LEN_6DOF_POSE = 6

# Approach variants commons
GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT = 4  # [d, nu, ksi, omega]

# Approach variants vector-based
GENOTYPE_LEN_CONTACT_POINT_FINDER = GENOTYPE_LEN_6DOF_POSE

# Approach variants voxel-based
GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF = 3  # [x, y, z]

# Approach variants list-based
GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST = 1  # [contact_points_list_id]

# Antipodal variants commons
GENOTYPE_LEN_HAND_POSITION_RELATIVE_TO_CONTACT_ENTRAXE = 1  # [tau]


# -------------------------------------------------------------------------------------------------------------------- #
# OFFSPRING
# -------------------------------------------------------------------------------------------------------------------- #

OFFSPRING_NB_COEFF = 1.  # number of offsprings generated (coeff of pop length)


# -------------------------------------------------------------------------------------------------------------------- #
# ARCHIVE
# -------------------------------------------------------------------------------------------------------------------- #

ARCHIVE_LIMIT_SIZE = 25000
ARCHIVE_DECREMENTAL_RATIO = 0.9  # if archive size is bigger than thresh, cut down archive by this ratio


# -------------------------------------------------------------------------------------------------------------------- #
# RUNNING DATA SAVING
# -------------------------------------------------------------------------------------------------------------------- #

DUMP_SCS_ARCHIVE_ON_THE_FLY = False  # Useful for backup if the run is interrupted ; not crucial for fast exec
N_GEN_FREQ_DUMP_SCS_ARCHIVE = 10


# -------------------------------------------------------------------------------------------------------------------- #
# TIMER
# -------------------------------------------------------------------------------------------------------------------- #

QD_RUN_TIME_LABEL = 'run_qd'


########################################################################################################################
#  STRATEGIES
########################################################################################################################


# -------------------------------------------------------------------------------------------------------------------- #
#  ARCHIVE MANAGEMENT STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #

class ArchiveType(Enum):
    ELITE_STRUCTURED = 1
    NOVELTY = 2
    NONE = 3


# -------------------------------------------------------------------------------------------------------------------- #
#  ARCHIVE MANAGEMENT STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #


class FillArchiveStrategy(Enum):
    NOVELTY_BASED = 1
    STRUCTURED_ELITES = 2
    NONE = 3


# -------------------------------------------------------------------------------------------------------------------- #
#  POPULATION MANAGEMENT STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #

class ReplacePopulationStrategy(Enum):
    RANDOM = 1
    NOVELTY_BASED = 2
    RESET_FROM_SCRATCH = 3


# -------------------------------------------------------------------------------------------------------------------- #
#  MUTATION STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #


class MutationStrategy(Enum):
    GAUSS = 1
    NONE = 2

# -------------------------------------------------------------------------------------------------------------------- #
#  SELECT OFFSPRING STRATEGIES
# -------------------------------------------------------------------------------------------------------------------- #


class SelectOffspringStrategy(Enum):
    RANDOM_FROM_POP = 1
    RANDOM_FROM_ARCHIVE = 2
    FITNESS_FROM_ARCHIVE = 3
    SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE = 4
    SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE_WITH_DUPLICATES = 5


# -------------------------------------------------------------------------------------------------------------------- #
# METHODS
# -------------------------------------------------------------------------------------------------------------------- #

class SupportedMethod(Enum):
    APPROACH_RAND_SAMPLE_CPF = 1
    ANTIPODAL_RAND_SAMPLE_CPF = 2
    RAND_SAMPLE_CONTACT_LIST = 3

    APPROACH_ME_SCS_CPF = 4
    ANTIPODAL_ME_SCS_CPF = 5
    ME_SCS_CONTACT_CPF = 6

    APPROACH_ME_RAND_CPF = 7
    ME_RAND_CONTACT_CPF = 8

    APPROACH_CMA_MAE_CPF = 9
    CMA_MAE_CONTACT_CPF = 10


# -------------------------------------------------------------------------------------------------------------------- #
# ARCHIVE LIMIT STRATEGY
# -------------------------------------------------------------------------------------------------------------------- #


class ArchiveLimitStrategy(Enum):
    RANDOM = 1


# -------------------------------------------------------------------------------------------------------------------- #
# SEARCH SPACE REPRESENTATION
# -------------------------------------------------------------------------------------------------------------------- #

class SearchSpaceRepresentation(Enum):
    APPROACH_STRATEGY_CONTACT_POINT_FINDER = 1
    ANTIPODAL_STRATEGY_CONTACT_POINT_FINDER = 2
    RANDOM_SAMPLE_CONTACT_STRATEGY_LIST = 3
    CONTACT_STRATEGY_CONTACT_POINT_FINDER = 4


########################################################################################################################
#  METHOD SETUPS
########################################################################################################################


# Input argument strings
ALGO_ARG_TO_METHOD = {
    'approach_rand_sample': SupportedMethod.APPROACH_RAND_SAMPLE_CPF,
    'antipodal_rand_sample': SupportedMethod.ANTIPODAL_RAND_SAMPLE_CPF,
    'rand_sample_contact_list': SupportedMethod.RAND_SAMPLE_CONTACT_LIST,

    'approach_me_scs': SupportedMethod.APPROACH_ME_SCS_CPF,
    'antipodal_me_scs': SupportedMethod.ANTIPODAL_ME_SCS_CPF,
    'me_scs_contact': SupportedMethod.ME_SCS_CONTACT_CPF,

    'approach_me_rand': SupportedMethod.APPROACH_ME_RAND_CPF,
    'me_rand_contact': SupportedMethod.ME_RAND_CONTACT_CPF,

    'approach_cma_mae': SupportedMethod.APPROACH_CMA_MAE_CPF,
    'cma_mae_contact': SupportedMethod.CMA_MAE_CONTACT_CPF,
}


# Configs associated with each method

QD_METHODS_CONFIGS = {
    SupportedMethod.APPROACH_RAND_SAMPLE_CPF: {
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
    SupportedMethod.ANTIPODAL_RAND_SAMPLE_CPF: {
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'search_representation': SearchSpaceRepresentation.ANTIPODAL_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_HAND_POSITION_RELATIVE_TO_CONTACT_ENTRAXE,
    },
    SupportedMethod.RAND_SAMPLE_CONTACT_LIST: {
        'archive_type': ArchiveType.NONE,
        'fill_archive_strat': FillArchiveStrategy.NONE,
        'mutation_strat': MutationStrategy.NONE,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_POP,
        'replace_pop_strat': ReplacePopulationStrategy.RESET_FROM_SCRATCH,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': True,
        'search_representation': SearchSpaceRepresentation.RANDOM_SAMPLE_CONTACT_STRATEGY_LIST,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_LIST + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
    SupportedMethod.APPROACH_ME_SCS_CPF: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
    SupportedMethod.ANTIPODAL_ME_SCS_CPF: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'search_representation': SearchSpaceRepresentation.ANTIPODAL_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_HAND_POSITION_RELATIVE_TO_CONTACT_ENTRAXE,
    },
    SupportedMethod.ME_SCS_CONTACT_CPF: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'search_representation': SearchSpaceRepresentation.CONTACT_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
    SupportedMethod.APPROACH_ME_RAND_CPF: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
    SupportedMethod.ME_RAND_CONTACT_CPF: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.RANDOM_FROM_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'search_representation': SearchSpaceRepresentation.CONTACT_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
    SupportedMethod.APPROACH_CMA_MAE_CPF: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'search_representation': SearchSpaceRepresentation.APPROACH_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
    SupportedMethod.CMA_MAE_CONTACT_CPF: {
        'archive_type': ArchiveType.ELITE_STRUCTURED,
        'fill_archive_strat': FillArchiveStrategy.STRUCTURED_ELITES,
        'mutation_strat': MutationStrategy.GAUSS,
        'select_off_strat': SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE,
        'replace_pop_strat': None,
        'archive_limit_strat': ArchiveLimitStrategy.RANDOM,
        'is_novelty_required': False,
        'is_pop_based': False,
        'search_representation': SearchSpaceRepresentation.CONTACT_STRATEGY_CONTACT_POINT_FINDER,
        'genotype_len': GENOTYPE_LEN_CONTACT_POINT_FINDER_CPF + GENOTYPE_LEN_6DOF_POSE_RELATIVE_TO_CONTACT_POINT,
    },
}

VXL_BASED_ALGORITHMS = [
    SupportedMethod.APPROACH_RAND_SAMPLE_CPF,
    SupportedMethod.ANTIPODAL_RAND_SAMPLE_CPF,
    SupportedMethod.APPROACH_ME_SCS_CPF,
    SupportedMethod.ANTIPODAL_ME_SCS_CPF,
    SupportedMethod.ME_SCS_CONTACT_CPF,
    SupportedMethod.APPROACH_ME_RAND_CPF,
    SupportedMethod.ME_RAND_CONTACT_CPF,
    SupportedMethod.APPROACH_CMA_MAE_CPF,
    SupportedMethod.CMA_MAE_CONTACT_CPF,
]

ANTIPODAL_BASED_ALGORITHMS = [
    SupportedMethod.ANTIPODAL_RAND_SAMPLE_CPF,
    SupportedMethod.ANTIPODAL_ME_SCS_CPF,
]

# -------------------------------------------------------------------------------------------------------------------- #
# PYRIBS INTERFACE
# -------------------------------------------------------------------------------------------------------------------- #

# Pyribs qd methods
CMA_ME_QD_METHODS = []
CMA_ES_QD_METHODS = []
CMA_MAE_QD_METHODS = [SupportedMethod.APPROACH_CMA_MAE_CPF, SupportedMethod.CMA_MAE_CONTACT_CPF]

PYRIBS_QD_METHODS = CMA_ME_QD_METHODS + CMA_ES_QD_METHODS + CMA_MAE_QD_METHODS

# Pyribs learning rates
CMA_MAE_PREDEFINED_ALPHA = 0.01
CMA_ME_PREDEFINED_ALPHA = 1.0
CMA_ES_PREDEFINED_ALPHA = 0.0

# Pyribs hyperparameters (similar to cma_mae paper)
CMA_MAE_EMITTER_BATCH_SIZE = 36
CMA_MAE_N_EMITTERS = 15
CMA_MAE_POP_SIZE = CMA_MAE_EMITTER_BATCH_SIZE * CMA_MAE_N_EMITTERS














