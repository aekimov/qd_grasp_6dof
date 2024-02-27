

from enum import Enum


# ====================================================================================== #
#                                   SUPPORTED ROBOTS
# ====================================================================================== #


SimulatedRobot = Enum(
    'SimulatedRobot',
    [
        'PANDA_2_FINGERS',
        'ALLEGRO_HAND',
        'BARRETT_HAND_280',
        'SHADOW_HAND'
     ]
)

INPUT_ARG_ROBOT2ROBOT_TYPE_NAME = {
    'panda_2f': SimulatedRobot.PANDA_2_FINGERS,
    'allegro': SimulatedRobot.ALLEGRO_HAND,
    'bh280': SimulatedRobot.BARRETT_HAND_280,
    'shadow': SimulatedRobot.SHADOW_HAND,
}


# ====================================================================================== #
#                                   RELATIVE PATHS
# ====================================================================================== #

GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS = '3d_models/robots'



# ====================================================================================== #
#                                 GRASPING EVALUATION
# ====================================================================================== #

SHAKING_PARAMETERS = {
    'n_shake': 1,
    'max_n_step': 1000,
    't_cmd_stable': 68,  # t_0.05
    'perturbated_joint_ids': [2, 3]
}




