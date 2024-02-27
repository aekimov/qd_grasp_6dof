
import numpy as np


"""
- 6DoF hand control joints:

j_id=0 | 'translationX'
j_id=1 | 'translationY'
j_id=2 | 'translationZ'
j_id=3 | 'rotationX'
j_id=4 | 'rotationY'
j_id=5 | 'rotationZ'

- Allegro hand joints:

j_id=6 | 'joint_00' # index basis rotation : should not be controlled !
j_id=7 | 'joint_01' # index closure : controlled # todo : modify min/ max values <----
j_id=8 | 'joint_02' # index closure : controlled # todo : modify min/ max values <----
j_id=9 | 'joint_03' # index closure : controlled # todo : modify min/ max values <----
j_id=10 | 'joint_03_tip' # fixed

j_id=11 | 'joint_04' # middle basis rotation : : should not be controlled !
j_id=12 | 'joint_05' # middle closure : controlled # todo : modify min/ max values <----
j_id=13 | 'joint_06' # middle closure : controlled # todo : modify min/ max values <----
j_id=14 | 'joint_07' # middle closure : controlled # todo : modify min/ max values <----
j_id=15 | 'joint_07_tip' # fixed

j_id=16 | 'joint_08' # last basis rotation : : should not be controlled !
j_id=17 | 'joint_09' # last closure : controlled # todo : modify min/ max values <----
j_id=18 | 'joint_10' # last closure : controlled # todo : modify min/ max values <----
j_id=19 | 'joint_11' # last closure : controlled # todo : modify min/ max values <----
j_id=20 | 'joint_11_tip' # fixed

j_id=21 | 'joint_12' # thumb basis rotation : : should not be controlled !
j_id=22 | 'joint_13' # thumb closure : controlled # todo : modify min/ max values <----
j_id=23 | 'joint_14' # thumb closure : controlled # todo : modify min/ max values <----
j_id=24 | 'joint_15' # thumb closure : controlled # todo : modify min/ max values <----
j_id=25 | 'joint_15_tip' # fixed

"""


GRIPPER_6DOF_INFOS = {
    0: {'axis': 'x', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2}, # rouge
    1: {'axis': 'y', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2},  # vert
    2: {'axis': 'z', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82,  'velocity_gain': 64, 'joint_target_val': 0.2},  # bleu

    3: {'axis': 'x', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    4: {'axis': 'y', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    5: {'axis': 'z', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
}

LIST_ID_GRIPPER_6DOF = list(GRIPPER_6DOF_INFOS.keys())

# Ids associated with the gripper's bodies
LIST_ID_GRIPPER_FINGERS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# Ids to trigger for closing the gripper
LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB = [23, 24]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX = [7, 8, 9]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID = [12, 13, 14]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_LAST = [17, 18, 19]

LIST_ID_GRIPPER_FINGERS_ACTUATORS = LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_LAST




GRIPPER_PARAMETERS = {
    'max_n_step_close_grip': 100,
    'max_velocity_maintain_6dof': 10,
    'force_maintain_6dof': 100
}


# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #

J_ID_METACARPAL_INDEX_ROTATION = 6
METACARPAL_INDEX_ROTATION_INIT_VALUE = 0.0
J_ID_METACARPAL_INDEX_CLOSURE = 7
METACARPAL_INDEX_CLOSURE_INIT_VALUE = 0.0
J_ID_PROXIMAL_INDEX_CLOSURE = 8
PROXIMAL_INDEX_CLOSURE_INIT_VALUE = 0.0
J_ID_DISTAL_INDEX_CLOSURE = 9
DISTAL_INDEX_CLOSURE_INIT_VALUE = 0.0

J_ID_METACARPAL_MIDDLE_ROTATION = 11
METACARPAL_MIDDLE_ROTATION_INIT_VALUE = 0.0
J_ID_METACARPAL_MIDDLE_CLOSURE = 12
METACARPAL_MIDDLE_CLOSURE_INIT_VALUE = 0.0
J_ID_PROXIMAL_MIDDLE_CLOSURE = 13
PROXIMAL_MIDDLE_CLOSURE_INIT_VALUE = 0.0
J_ID_DISTAL_MIDDLE_CLOSURE = 14
DISTAL_MIDDLE_CLOSURE_INIT_VALUE = 0.0

J_ID_METACARPAL_LAST_ROTATION = 16
METACARPAL_LAST_ROTATION_INIT_VALUE = 0.0
J_ID_METACARPAL_LAST_CLOSURE = 17
METACARPAL_LAST_CLOSURE_INIT_VALUE = 0.0
J_ID_PROXIMAL_LAST_CLOSURE = 18
PROXIMAL_LAST_CLOSURE_INIT_VALUE = 0.0
J_ID_DISTAL_LAST_CLOSURE = 19
DISTAL_LAST_CLOSURE_INIT_VALUE = 0.0


J_ID_THUMB_OPPOSITION = 21
OPPOSED_THUMB_INIT_VALUE = 1.396
J_ID_METACARPAL_THUMB_ROTATION = 22
METACARPAL_THUMB_ROTATION_INIT_VALUE = 0.0
J_ID_PROXIMAL_THUMB_CLOSURE = 23
PROXIMAL_THUMB_CLOSURE_INIT_VALUE = 0.0
J_ID_DISTAL_THUMB_CLOSURE = 24
DISTAL_THUMB_CLOSURE_INIT_VALUE = 0.0

# Manually force joint poses to initialize them at a different pose that the associated upper bound value
DEFAULT_JOINT_STATES = {
    J_ID_METACARPAL_INDEX_CLOSURE: METACARPAL_INDEX_CLOSURE_INIT_VALUE,
    J_ID_PROXIMAL_INDEX_CLOSURE: PROXIMAL_INDEX_CLOSURE_INIT_VALUE,
    J_ID_DISTAL_INDEX_CLOSURE: DISTAL_INDEX_CLOSURE_INIT_VALUE,

    J_ID_METACARPAL_MIDDLE_ROTATION: METACARPAL_MIDDLE_ROTATION_INIT_VALUE,
    J_ID_METACARPAL_MIDDLE_CLOSURE: METACARPAL_MIDDLE_CLOSURE_INIT_VALUE,
    J_ID_PROXIMAL_MIDDLE_CLOSURE: PROXIMAL_MIDDLE_CLOSURE_INIT_VALUE,
    J_ID_DISTAL_MIDDLE_CLOSURE: DISTAL_MIDDLE_CLOSURE_INIT_VALUE,

    J_ID_METACARPAL_LAST_ROTATION: METACARPAL_LAST_ROTATION_INIT_VALUE,
    J_ID_METACARPAL_LAST_CLOSURE: METACARPAL_LAST_CLOSURE_INIT_VALUE,
    J_ID_PROXIMAL_LAST_CLOSURE: PROXIMAL_LAST_CLOSURE_INIT_VALUE,
    J_ID_DISTAL_LAST_CLOSURE: DISTAL_LAST_CLOSURE_INIT_VALUE,

    J_ID_THUMB_OPPOSITION: OPPOSED_THUMB_INIT_VALUE,
    J_ID_METACARPAL_THUMB_ROTATION: METACARPAL_THUMB_ROTATION_INIT_VALUE,
    J_ID_PROXIMAL_THUMB_CLOSURE: PROXIMAL_THUMB_CLOSURE_INIT_VALUE,
    J_ID_DISTAL_THUMB_CLOSURE: DISTAL_THUMB_CLOSURE_INIT_VALUE,
}


# ---------------------------------------------- #
#                    3D Models
# ---------------------------------------------- #

ALLEGRO_HAND_GRIP_RELATIVE_PATH_URDF = "allegro_hand.urdf"


# ---------------------------------------------- #
#                    Synergies
# ---------------------------------------------- #

# [0, 1, 2, 3]
FIXED_INTERVAL_SYNERGIES = [0, 3]

SYNERGIES_ID_TO_STR = {
    0: "thumb_index",
    1: "thumb_mid",
    2: "thumb_index_mid",
    3: "all",
}

SYNERGIES_STR_TO_J_ID_GRIP_FINGERS_ACTUATED = {
    "thumb_index": LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB +
                   LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX,

    "thumb_mid": LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB +
                 LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID,

    "thumb_index_mid": LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB +
                       LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX +
                       LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID,
    "all": LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB +
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX +
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID +
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_LAST,
}



# ---------------------------------------------- #
#                   KEY MEASURES
# ---------------------------------------------- #

WRIST_PALM_OFFSET = 0.  # CODE DE MATHILDE : "PALM_WIRST"
EXTREMITY_WIRST = 0.2477
EXTREMITY_PALM = EXTREMITY_WIRST - 0.2477/3
HALF_PALM_DEPTH = 0.012


MAX_HAND_STANDOFF_ALLEGRO = EXTREMITY_PALM

POSE_RELATIVE_TO_CONTACT_POINT_D_MIN = WRIST_PALM_OFFSET + HALF_PALM_DEPTH
POSE_RELATIVE_TO_CONTACT_POINT_D_MAX = WRIST_PALM_OFFSET + HALF_PALM_DEPTH + MAX_HAND_STANDOFF_ALLEGRO
