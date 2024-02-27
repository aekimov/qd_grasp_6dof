
import numpy as np


"""
- 6DoF hand control joints:

j_id=0 | 'translationX'
j_id=1 | 'translationY'
j_id=2 | 'translationZ'
j_id=3 | 'rotationX'
j_id=4 | 'rotationY'
j_id=5 | 'rotationZ'


- Barrett hand joints:

j_id=6 | 'hand_joint' # fixed
j_id=7 | 'bh_base_joint' # fixed

j_id=8 | 'bh_j32_joint' # finger 3 (central) palm-to-proximal closure : controlled
j_id=9 | 'bh_j33_joint' # finger 3 (central) proximal-to-distal closure : controlled

j_id=10 | 'bh_j11_joint' # finger 1 palm-to-proximal rotation : controlled
j_id=11 | 'bh_j12_joint' # finger 1 palm-to-proximal closure : controlled
j_id=12 | 'bh_j13_joint' # finger 1 proximal-to-distal closure : controlled

j_id=13 | 'bh_j21_joint' # finger 2 palm-to-proximal rotation : controlled
j_id=14 | 'bh_j22_joint' # finger 2 palm-to-proximal closure : controlled
j_id=15 | 'bh_j23_joint' # finger 2 proximal-to-distal closure : controlled

"""

GRIPPER_6DOF_INFOS = {
    0: {'axis': 'x', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2},  # rouge
    1: {'axis': 'y', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2},  # vert
    2: {'axis': 'z', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82,  'velocity_gain': 64, 'joint_target_val': 0.2},  # bleu

    3: {'axis': 'x', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    4: {'axis': 'y', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    5: {'axis': 'z', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
}

LIST_ID_GRIPPER_6DOF = list(GRIPPER_6DOF_INFOS.keys())

# Ids associated with the gripper's bodies
LIST_ID_GRIPPER_FINGERS = [8, 9, 10, 11, 12, 13, 14, 15]

# Ids to trigger for closing the gripper
LIST_ID_GRIPPER_FINGERS_ACTUATORS_FINGER_1 = [11, 12]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_FINGER_2 = [14, 15]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_FINGER_3 = [8, 9]


LIST_ID_GRIPPER_FINGERS_ACTUATORS = LIST_ID_GRIPPER_FINGERS_ACTUATORS_FINGER_1 + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_FINGER_2 + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_FINGER_3

GRIPPER_PARAMETERS = {
    'max_n_step_close_grip': 100,
    'max_velocity_maintain_6dof': 10,
    'force_maintain_6dof': 100
}


# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #

# Manually force joint poses to initialize them at a different pose that the associated upper bound value
DEFAULT_JOINT_STATES = {}


# ---------------------------------------------- #
#                    3D Models
# ---------------------------------------------- #

BARRETT_HAND_280_GRIP_RELATIVE_PATH_URDF = "barrett_hand_280.urdf"


# ---------------------------------------------- #
#                   KEY MEASURES
# ---------------------------------------------- #
WRIST_PALM_OFFSET = 0.075
EXTREMITY_WIRST = 0.075 + 0.095
EXTREMITY_PALM = 0.095
HALF_PALM_DEPTH = 0.008


MAX_HAND_STANDOFF_BARRETT_HAND_280 = EXTREMITY_PALM

POSE_RELATIVE_TO_CONTACT_POINT_D_MIN = HALF_PALM_DEPTH
POSE_RELATIVE_TO_CONTACT_POINT_D_MAX = HALF_PALM_DEPTH + MAX_HAND_STANDOFF_BARRETT_HAND_280


# ---------------------------------------------- #
#         INITIAL JOINT STATES IN GENOME
# ---------------------------------------------- #

N_INIT_JOINT_STATE_GENES = 2

J_ID_FINGER_1_PALM_TO_PROXIMAL_ROTATION = 10
J_ID_FINGER_2_PALM_TO_PROXIMAL_ROTATION = 13

FINGER_1_PALM_TO_PROXIMAL_ROTATION_MIN_VAL = 3.1416
FINGER_1_PALM_TO_PROXIMAL_ROTATION_MAX_VAL = 0

FINGER_2_PALM_TO_PROXIMAL_ROTATION_MIN_VAL = 3.1416
FINGER_2_PALM_TO_PROXIMAL_ROTATION_MAX_VAL = 0

J_ID_INIT_JOINT_STATE_BOUNDARIES = {
    J_ID_FINGER_1_PALM_TO_PROXIMAL_ROTATION: {
        'min': FINGER_1_PALM_TO_PROXIMAL_ROTATION_MIN_VAL,
        'max': FINGER_1_PALM_TO_PROXIMAL_ROTATION_MAX_VAL,
    },
    J_ID_FINGER_2_PALM_TO_PROXIMAL_ROTATION: {
        'min': FINGER_2_PALM_TO_PROXIMAL_ROTATION_MIN_VAL,
        'max': FINGER_2_PALM_TO_PROXIMAL_ROTATION_MAX_VAL,
    },
}

J_ID_FROM_ORDERED_INIT_JOINT_STATE_GENES = {
    0: J_ID_FINGER_1_PALM_TO_PROXIMAL_ROTATION,
    1: J_ID_FINGER_2_PALM_TO_PROXIMAL_ROTATION,
}




