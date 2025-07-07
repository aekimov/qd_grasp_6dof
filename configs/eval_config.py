
import numpy as np
import environments.src.env_constants as env_consts


# -------------------------------------------------------------------------------------------------------------------- #
# ENVIRONMENT EVALUATION CONFIGS
# -------------------------------------------------------------------------------------------------------------------- #

ENV_EVAL_CONFIGS = {
    env_consts.SimulatedRobot.PANDA_2_FINGERS: {
        'with_synergy': False,
        'n_synergies': None,

        'with_init_joint_state_in_genome': False,
        'n_init_joint_states': None,
    },
    env_consts.SimulatedRobot.ROBOTIQ_2_FINGERS: {
        'with_synergy': False,
        'n_synergies': None,

        'with_init_joint_state_in_genome': False,
        'n_init_joint_states': None,
    },
    env_consts.SimulatedRobot.ALLEGRO_HAND: {
        'with_synergy': True,
        'n_synergies': 4,

        'with_init_joint_state_in_genome': False,
        'n_init_joint_states': None,
    },
    env_consts.SimulatedRobot.BARRETT_HAND_280: {
        'with_synergy': False,
        'n_synergies': None,

        'with_init_joint_state_in_genome': True,
        'n_init_joint_states': 2,
    },
    env_consts.SimulatedRobot.SHADOW_HAND: {
        'with_synergy': False,
        'n_synergies': None,

        'with_init_joint_state_in_genome': False,
        'n_init_joint_states': None,
    },
}


# -------------------------------------------------------------------------------------------------------------------- #
# INFO KEYS
# -------------------------------------------------------------------------------------------------------------------- #

OUTCOME_INFO_KEYS = ['is_success', 'is_valid', 'is_obj_touched', 'is_robust_grasp']
OUTCOME_6DOF_KEYS = ['xyz_pose_x', 'xyz_pose_y', 'xyz_pose_z', 'quat_1', 'quat_2', 'quat_3', 'quat_4']

INFO_KEYS = OUTCOME_INFO_KEYS + OUTCOME_6DOF_KEYS
INFO_LEN = len(INFO_KEYS)

IS_SUCCESS_KEY_ID = 0
IS_VALID_KEY_ID = 1
IS_OBJ_TOUCHED_KEY_ID = 2
IS_ROBUST_GRASP_KEY_ID = 3

XYZ_POSE_X_KEY_ID = 4
XYZ_POSE_Y_KEY_ID = 5
XYZ_POSE_Z_KEY_ID = 6

XYZ_QUATERNION_1_KEY_ID = 7
XYZ_QUATERNION_2_KEY_ID = 8
XYZ_QUATERNION_3_KEY_ID = 9
XYZ_QUATERNION_4_KEY_ID = 10


# -------------------------------------------------------------------------------------------------------------------- #
# ORIENTATION LIMITS
# -------------------------------------------------------------------------------------------------------------------- #

BOUND_EULER_ORIENT = np.pi
FIXED_INTERVAL_EULER = [-BOUND_EULER_ORIENT, BOUND_EULER_ORIENT]


# -------------------------------------------------------------------------------------------------------------------- #
# BULLET CONSTANTS
# -------------------------------------------------------------------------------------------------------------------- #

# Visualization
BULLET_DEFAULT_DISPLAY_FLG = False  # whether to display steps

# Frictions
ROLLING_FRICTION_OBJ_DEFAULT_VALUE = 0.01
SPINNING_FRICTION_OBJ_DEFAULT_VALUE = 0.1


# -------------------------------------------------------------------------------------------------------------------- #
# OBJECT LOAD MODE
# -------------------------------------------------------------------------------------------------------------------- #

LOAD_OBJECT_WITH_FIXED_BASE = False


# -------------------------------------------------------------------------------------------------------------------- #
# DOMAIN RANDOMIZATION BASED FITNESS
# -------------------------------------------------------------------------------------------------------------------- #

MIN_DOMAIN_RANDOMIZATION_FITNESS_VALUE = 0.
DOMAIN_RANDOMIZATION_N_NOISY_TRIALS = 100

DOMAIN_RANDOMIZATION_OBJECT_POS_VARIANCE_IN_M = 0.005
DOMAIN_RANDOMIZATION_OBJECT_ORIENT_EULER_VARIANCE_IN_DEG = 30.
DOMAIN_RANDOMIZATION_OBJECT_ORIENT_EULER_VARIANCE_IN_RAD = \
    DOMAIN_RANDOMIZATION_OBJECT_ORIENT_EULER_VARIANCE_IN_DEG * np.pi / 180

DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MIN_VALUE = 0.01
DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MAX_VALUE = 0.04
DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MIN_VALUE = 0.1
DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MAX_VALUE = 0.4


# -------------------------------------------------------------------------------------------------------------------- #
# METHODS LIMITATIONS PER ROBOTS
# -------------------------------------------------------------------------------------------------------------------- #

SUPPORTED_ROBOTS_FOR_ANTIPODAL_BASED_METHODS = [env_consts.SimulatedRobot.PANDA_2_FINGERS]



