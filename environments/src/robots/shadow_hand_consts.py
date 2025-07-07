
import numpy as np

"""

joint_state_direct (0, b'translationX', 1, 7, 6, 1, 1e-05, 0.0, -1000.0, 1000.0, 20.0, 1.0, b'base_link2', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
joint_state_direct (1, b'translationY', 1, 8, 7, 1, 0.0, 0.0, -1000.0, 1000.0, 10000.0, 1.0, b'base_link3', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
joint_state_direct (2, b'translationZ', 1, 9, 8, 1, 0.0, 0.0, -100.0, 100.0, 20.0, 0.2, b'base_link4', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1)
joint_state_direct (3, b'rotationX', 0, 10, 9, 1, 0.0, 0.0, -100.0, 100.0, 20.0, 0.2, b'base_link5', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 2)
joint_state_direct (4, b'rotationY', 0, 11, 10, 1, 0.0, 0.0, -100.0, 100.0, 20.0, 0.2, b'base_link6', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 3)
joint_state_direct (5, b'rotationZ', 0, 12, 11, 1, 0.0, 0.0, -100.0, 100.0, 20.0, 0.2, b'base_link7', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 4)
joint_state_direct (6, b'offset', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'forearm', (0.0, 0.0, 0.0), (0.0, 0.0, -0.32), (0.0, 0.0, 0.0, 1.0), 5)

joint_state_direct (7, b'WRJ2', 4, -1, -1, 0, 0.1, 0.01, -0.5235987755982988, 0.17453292519943295, 5.0, 1.0, b'wrist', (0.0, 0.0, 0.0), (0.0, -0.01, -0.787), (0.0, 0.0, 0.0, 1.0), 6)
joint_state_direct (8, b'WRJ1', 4, -1, -1, 0, 0.1, 0.01, -0.7853981633974483, 0.6108652381980153, 5.0, 1.0, b'palm', (0.0, 0.0, 0.0), (0.0, 0.0, 0.034), (0.0, 0.0, 0.0, 1.0), 7)

joint_state_direct (9, b'FFJ4', 0, 13, 12, 1, 0.1, 0.01, -0.4363323129985824, 0.4363323129985824, 100.0, 1.0, b'ffknuckle', (0.0, -1.0, 0.0), (0.033, 0.0, 0.06), (0.0, 0.0, 0.0, 1.0), 8)
joint_state_direct (10, b'FFJ3', 0, 14, 13, 1, 0.1, 0.01, 0.0, 1.57079632679, 100.0, 1.0, b'ffproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 9)
joint_state_direct (11, b'FFJ2', 0, 15, 14, 1, 0.1, 0.01, 0.0, 1.57079632679, 100.0, 1.0, b'ffmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 10)
joint_state_direct (12, b'FFJ1', 0, 16, 15, 1, 0.1, 0.01, 0.0, 1.57079632679, 100.0, 1.0, b'ffdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 11)
joint_state_direct (13, b'FFtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'fftip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.013), (0.0, 0.0, 0.0, 1.0), 12)

joint_state_direct (14, b'MFJ4', 0, 17, 16, 1, 0.1, 0.01, -0.4363323129985824, 0.4363323129985824, 100.0, 1.0, b'mfknuckle', (0.0, -1.0, 0.0), (0.011, 0.0, 0.064), (0.0, 0.0, 0.0, 1.0), 8)
joint_state_direct (15, b'MFJ3', 0, 18, 17, 1, 0.1, 0.01, 0.0, 1.57079632679, 100.0, 1.0, b'mfproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 14)
joint_state_direct (16, b'MFJ2', 0, 19, 18, 1, 0.1, 0.01, 0.0, 1.57079632679, 100.0, 1.0, b'mfmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 15)
joint_state_direct (17, b'MFJ1', 0, 20, 19, 1, 0.1, 0.01, 0.0, 1.57079632679, 100.0, 1.0, b'mfdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 16)
joint_state_direct (18, b'MFtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'mftip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.013), (0.0, 0.0, 0.0, 1.0), 17)

joint_state_direct (19, b'RFJ4', 0, 21, 20, 1, 0.1, 0.01, -0.4363323129985824, 0.4363323129985824, 100.0, 1.0, b'rfknuckle', (0.0, 1.0, 0.0), (-0.011, 0.0, 0.06), (0.0, 0.0, 0.0, 1.0), 8)
joint_state_direct (20, b'RFJ3', 0, 22, 21, 1, 0.0, 0.0, 0.0, 1.57079632679, 100.0, 1.0, b'rfproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 19)
joint_state_direct (21, b'RFJ2', 0, 23, 22, 1, 0.0, 0.0, 0.0, 1.57079632679, 100.0, 1.0, b'rfmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 20)
joint_state_direct (22, b'RFJ1', 0, 24, 23, 1, 0.0, 0.0, 0.0, 1.57079632679, 100.0, 1.0, b'rfdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 21)
joint_state_direct (23, b'RFtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rftip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.013), (0.0, 0.0, 0.0, 1.0), 22)

joint_state_direct (24, b'LFJ5', 0, 25, 24, 1, 0.0, 0.0, 0.0, 0.6981317007977318, 100.0, 1.0, b'lfmetacarpal', (0.573576436, 0.0, 0.819152044), (-0.033, 0.0, -0.014290000000000004), (0.0, 0.0, 0.0, 1.0), 8)
joint_state_direct (25, b'LFJ4', 0, 26, 25, 1, 0.0, 0.0, -0.4363323129985824, 0.4363323129985824, 100.0, 1.0, b'lfknuckle', (0.0, 1.0, 0.0), (0.0, 0.0, 0.02579), (0.0, 0.0, 0.0, 1.0), 24)
joint_state_direct (26, b'LFJ3', 0, 27, 26, 1, 0.0, 0.0, 0.0, 1.57079632679, 100.0, 1.0, b'lfproximal', (1.0, 0.0, 0.0), (0.0, 0.0, -0.06579), (0.0, 0.0, 0.0, 1.0), 25)
joint_state_direct (27, b'LFJ2', 0, 28, 27, 1, 0.0, 0.0, 0.0, 1.57079632679, 100.0, 1.0, b'lfmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 26)
joint_state_direct (28, b'LFJ1', 0, 29, 28, 1, 0.0, 0.0, 0.0, 1.57079632679, 100.0, 1.0, b'lfdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 27)

joint_state_direct (29, b'THJ5', 0, 30, 29, 1, 0.0, 0.0, 0.0, 1.309, 100.0, 1.0, b'thbase', (0.0, 0.0, -1.0), (0.034, -0.0085, -0.006000000000000002), (0.0, -0.38268343236488267, 0.0, 0.9238795325113726), 8)
joint_state_direct (30, b'THJ4', 0, 31, 30, 1, 0.0, 0.0, 0.0, 1.309, 100.0, 1.0, b'thproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 29)
joint_state_direct (31, b'THJ3', 0, 32, 31, 1, 0.0, 0.0, -0.2618, 0.2618, 100.0, 1.0, b'thhub', (1.0, 0.0, 0.0), (0.0, 0.0, 0.019), (0.0, 0.0, 0.0, 1.0), 30)
joint_state_direct (32, b'THJ2', 0, 33, 32, 1, 0.0, 0.0, -0.5237, 0.5237, 100.0, 1.0, b'thmiddle', (0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 31)
joint_state_direct (33, b'THJ1', 0, 34, 33, 1, 0.0, 0.0, 0.0, 1.571, 100.0, 1.0, b'thdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.016), (0.0, 0.0, 0.7071067812590626, 0.7071067811140325), 32)

"""


GRIPPER_6DOF_INFOS = {
    0: {'axis': 'x', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2}, # rouge
    1: {'axis': 'y', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2},  # vert
    2: {'axis': 'z', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82,  'velocity_gain': 64, 'joint_target_val': 0.2},  # bleu

    3: {'axis': 'x', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    4: {'axis': 'y', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    5: {'axis': 'z', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
}


# Ids associated with the gripper's bodies
LIST_ID_GRIPPER_FINGERS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

# Ids to trigger for closing the gripper
LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX = [10, 11, 12, 13]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID = [15, 16, 17, 18]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_RING = [20, 21, 22, 23]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_LAST = [26, 27, 28, 29]

LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_OPPOSITE = [29, 31, 32, 33]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_PALM_GRASP = [29, 30, 31, 32, 33]
LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB = LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_PALM_GRASP

# Adduction/Abduction joints (J4 joints)
LIST_ID_GRIPPER_FINGERS_ACTUATORS_ADDUCTION = [9, 14, 19, 25]  # FFJ4, MFJ4, RFJ4, LFJ4

LIST_ID_GRIPPER_FINGERS_ACTUATORS = LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_RING + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_LAST + \
                                    LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB
                                    # LIST_ID_GRIPPER_FINGERS_ACTUATORS_ADDUCTION # Add adduction/abduction joints

GRIPPER_PARAMETERS = {
    'max_n_step_close_grip': 100,
    'max_velocity_maintain_6dof': 10,
    'force_maintain_6dof': 1000
}


# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #

# Hand base

J_ID_FOREARM_WIRST = 7  # WRJ2
J_ID_FOREARM_WIRST_INIT_VALUE = 0.0

J_ID_WIRST_PALM = 8  # WRJ1
J_ID_WIRST_PALM_INIT_VALUE = 0.0


# Thumb
J_ID_PALM_THBASE_ROTATION = 29  # THJ5
PALM_THBASE_ROTATION_INIT_VALUE = 0

J_ID_THBASE_THPROXIMAL_ROTATION = 30  # THJ4
THBASE_THPROXIMAL_ROTATION_INIT_VALUE = 0.8

J_ID_THPROXIMAL_THHUB_ROTATION = 31  # THJ3
THPROXIMAL_THHUB_ROTATION_INIT_VALUE = 0

J_ID_THHUB_THMIDDLE_ROTATION = 32  # THJ2
THHUB_THMIDDLE_ROTATION_INIT_VALUE = 0

J_ID_THMIDDLE_THDISTAL_ROTATION = 33  # THJ1
THMIDDLE_THDISTAL_ROTATION_INIT_VALUE = 0

# Index

J_ID_PALM_FFNUCKLE_ROTATION = 9  # FFJ4
PALM_FFNUCKLE_ROTATION_INIT_VALUE = 0

J_ID_FFNUCKLE_FFPROXIMAL_ROTATION = 10  # FFJ3
FFNUCKLE_FFPROXIMAL_ROTATION_INIT_VALUE = 0

J_ID_FFPROXIMAL_FFMIDDLE_ROTATION = 11  # FFJ2
FFPROXIMAL_FFMIDDLE_ROTATION_INIT_VALUE = 0

J_ID_FFMIDLE_FFDISTAL_ROTATION = 12  # FFJ1
FFMIDLE_FFDISTAL_ROTATION_INIT_VALUE = 0

J_ID_FFDISTAL_FFTIP_ROTATION = 13  # FFtip
FFDISTAL_FFTIP_ROTATION_INIT_VALUE = 0

# Last

J_ID_PALM_LFMETACARPAL_ROTATION = 24  # LFJ5
PALM_LFMETACARPAL_ROTATION_INIT_VALUE = 0

J_ID_LFMETACARPAL_LFNUCKLE_ROTATION = 25  # LFJ4
LFMETACARPAL_LFNUCKLE_ROTATION_INIT_VALUE = 0

J_ID_LFNUCKLE_LFPROXIMAL_ROTATION = 26  # LFJ3
LFNUCKLE_LFPROXIMAL_ROTATION_INIT_VALUE = 0

J_ID_LFPROXIMAL_LFMIDDLE_ROTATION = 27  # LFJ2
LFPROXIMAL_LFMIDDLE_ROTATION_INIT_VALUE = 0

J_ID_LFMIDDLE_LFDISTAL_ROTATION = 28  # LFJ1
LFMIDDLE_LFDISTAL_ROTATION_INIT_VALUE = 0

# Middle

J_ID_PALM_MFNUCKLE_ROTATION = 14  # MFJ4
PALM_MFNUCKLE_ROTATION_INIT_VALUE = 0

J_ID_MFNUCKLE_MFPROXIMAL_ROTATION = 15  # MFJ3
MFNUCKLE_MFPROXIMAL_ROTATION_INIT_VALUE = 0

J_ID_MFPROXIMAL_MFMIDDLE_ROTATION = 16  # MFJ2
MFPROXIMAL_MFMIDDLE_ROTATION_INIT_VALUE = 0

J_ID_MFMIDLE_MFDISTAL_ROTATION = 17   # MFJ1
MFMIDLE_MFDISTAL_ROTATION_INIT_VALUE = 0

J_ID_MFDISTAL_MFTIP_ROTATION = 18   # FFtip
MFDISTAL_MFTIP_ROTATION_INIT_VALUE = 0

# Ring

J_ID_PALM_RFNUCKLE_ROTATION = 19  # RFJ4
PALM_RFNUCKLE_ROTATION_INIT_VALUE = 0

J_ID_RFNUCKLE_RFPROXIMAL_ROTATION = 20  # RFFJ3
RFNUCKLE_RFPROXIMAL_ROTATION_INIT_VALUE = 0

J_ID_RFPROXIMAL_RFMIDDLE_ROTATION = 21  # RFJ2
RFPROXIMAL_RFMIDDLE_ROTATION_INIT_VALUE = 0

J_ID_RFMIDLE_RFDISTAL_ROTATION = 22  # RFJ1
RFMIDLE_RFDISTAL_ROTATION_INIT_VALUE = 0

J_ID_RFDISTAL_RFTIP_ROTATION = 23  # RFtip
RFDISTAL_RFTIP_ROTATION_INIT_VALUE = 0

# Add initial values for adduction/abduction joints
J_ID_PALM_FFKNUCKLE_ADDUCTION = 9  # FFJ4
PALM_FFKNUCKLE_ADDUCTION_INIT_VALUE = 0 #-0.3

J_ID_PALM_MFKNUCKLE_ADDUCTION = 14  # MFJ4
PALM_MFKNUCKLE_ADDUCTION_INIT_VALUE = 0 #-0.15

J_ID_PALM_RFKNUCKLE_ADDUCTION = 19  # RFJ4
PALM_RFKNUCKLE_ADDUCTION_INIT_VALUE = 0 #-0.15

J_ID_PALM_LFKNUCKLE_ADDUCTION = 25  # LFJ4
PALM_LFKNUCKLE_ADDUCTION_INIT_VALUE = 0 #-0.3

DEFAULT_JOINT_STATES = {

    # Basis
    J_ID_FOREARM_WIRST: J_ID_FOREARM_WIRST_INIT_VALUE,
    J_ID_WIRST_PALM: J_ID_WIRST_PALM_INIT_VALUE,

    # Thumb
    J_ID_PALM_THBASE_ROTATION: PALM_THBASE_ROTATION_INIT_VALUE,
    J_ID_THBASE_THPROXIMAL_ROTATION: THBASE_THPROXIMAL_ROTATION_INIT_VALUE,
    J_ID_THPROXIMAL_THHUB_ROTATION: THPROXIMAL_THHUB_ROTATION_INIT_VALUE,
    J_ID_THHUB_THMIDDLE_ROTATION: THHUB_THMIDDLE_ROTATION_INIT_VALUE,
    J_ID_THMIDDLE_THDISTAL_ROTATION: THMIDDLE_THDISTAL_ROTATION_INIT_VALUE,

    # Index
    J_ID_PALM_FFNUCKLE_ROTATION: PALM_FFNUCKLE_ROTATION_INIT_VALUE,
    J_ID_FFNUCKLE_FFPROXIMAL_ROTATION: FFNUCKLE_FFPROXIMAL_ROTATION_INIT_VALUE,
    J_ID_FFPROXIMAL_FFMIDDLE_ROTATION: FFPROXIMAL_FFMIDDLE_ROTATION_INIT_VALUE,
    J_ID_FFMIDLE_FFDISTAL_ROTATION: FFMIDLE_FFDISTAL_ROTATION_INIT_VALUE,
    J_ID_FFDISTAL_FFTIP_ROTATION: FFDISTAL_FFTIP_ROTATION_INIT_VALUE,

    # Last
    J_ID_PALM_LFMETACARPAL_ROTATION: PALM_LFMETACARPAL_ROTATION_INIT_VALUE,
    J_ID_LFMETACARPAL_LFNUCKLE_ROTATION: LFMETACARPAL_LFNUCKLE_ROTATION_INIT_VALUE,
    J_ID_LFNUCKLE_LFPROXIMAL_ROTATION: LFNUCKLE_LFPROXIMAL_ROTATION_INIT_VALUE,
    J_ID_LFPROXIMAL_LFMIDDLE_ROTATION: LFPROXIMAL_LFMIDDLE_ROTATION_INIT_VALUE,
    J_ID_LFMIDDLE_LFDISTAL_ROTATION: LFMIDDLE_LFDISTAL_ROTATION_INIT_VALUE,

    # Middle
    J_ID_PALM_MFNUCKLE_ROTATION: PALM_MFNUCKLE_ROTATION_INIT_VALUE,
    J_ID_MFNUCKLE_MFPROXIMAL_ROTATION: MFNUCKLE_MFPROXIMAL_ROTATION_INIT_VALUE,
    J_ID_MFPROXIMAL_MFMIDDLE_ROTATION: MFPROXIMAL_MFMIDDLE_ROTATION_INIT_VALUE,
    J_ID_MFMIDLE_MFDISTAL_ROTATION: MFMIDLE_MFDISTAL_ROTATION_INIT_VALUE,
    J_ID_MFDISTAL_MFTIP_ROTATION: MFDISTAL_MFTIP_ROTATION_INIT_VALUE,

    # Ring
    J_ID_PALM_RFNUCKLE_ROTATION: PALM_RFNUCKLE_ROTATION_INIT_VALUE,
    J_ID_RFNUCKLE_RFPROXIMAL_ROTATION: RFNUCKLE_RFPROXIMAL_ROTATION_INIT_VALUE,
    J_ID_RFPROXIMAL_RFMIDDLE_ROTATION: RFPROXIMAL_RFMIDDLE_ROTATION_INIT_VALUE,
    J_ID_RFMIDLE_RFDISTAL_ROTATION: RFMIDLE_RFDISTAL_ROTATION_INIT_VALUE,
    J_ID_RFDISTAL_RFTIP_ROTATION: RFDISTAL_RFTIP_ROTATION_INIT_VALUE,
    
    # Adduction joints
    J_ID_PALM_FFKNUCKLE_ADDUCTION: PALM_FFKNUCKLE_ADDUCTION_INIT_VALUE,
    J_ID_PALM_MFKNUCKLE_ADDUCTION: PALM_MFKNUCKLE_ADDUCTION_INIT_VALUE,
    J_ID_PALM_RFKNUCKLE_ADDUCTION: PALM_RFKNUCKLE_ADDUCTION_INIT_VALUE,
    J_ID_PALM_LFKNUCKLE_ADDUCTION: PALM_LFKNUCKLE_ADDUCTION_INIT_VALUE,
}

# ---------------------------------------------- #
#                    3D Models
# ---------------------------------------------- #

SHADOW_HAND_GRIP_RELATIVE_PATH_URDF = "shadow_hand.urdf"


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
    "thumb_index": #[J_ID_PALM_FFKNUCKLE_ADDUCTION] +  # FFJ4 for index adduction
                   LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX +
                #    LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_OPPOSITE +
                   LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_PALM_GRASP,

    "thumb_mid": #[J_ID_PALM_MFKNUCKLE_ADDUCTION] +  # MFJ4 for middle adduction
                 LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_OPPOSITE +
                 LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID,

    "thumb_index_mid": #[J_ID_PALM_FFKNUCKLE_ADDUCTION, J_ID_PALM_MFKNUCKLE_ADDUCTION] +  # FFJ4 and MFJ4 for adduction
                       LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_OPPOSITE +
                       LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX +
                       LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID,

    "all": #LIST_ID_GRIPPER_FINGERS_ACTUATORS_ADDUCTION +  # All J4 joints
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_THUMB_FOR_PALM_GRASP +
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_INDEX +
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_MID +
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_RING +
           LIST_ID_GRIPPER_FINGERS_ACTUATORS_LAST
}

# ---------------------------------------------- #
#           JOINT LOCKING CONFIGURATIONS
# ---------------------------------------------- #

# Define which joints correspond to each level
DISTAL_JOINTS = [
    J_ID_FFMIDLE_FFDISTAL_ROTATION,     # FFJ1 - Index distal
    J_ID_MFMIDLE_MFDISTAL_ROTATION,     # MFJ1 - Middle distal
    J_ID_RFMIDLE_RFDISTAL_ROTATION,     # RFJ1 - Ring distal
    J_ID_LFMIDDLE_LFDISTAL_ROTATION,    # LFJ1 - Little distal
    J_ID_THMIDDLE_THDISTAL_ROTATION,    # THJ1 - Thumb distal
]

MIDDLE_JOINTS = [
    J_ID_FFPROXIMAL_FFMIDDLE_ROTATION,  # FFJ2 - Index middle
    J_ID_MFPROXIMAL_MFMIDDLE_ROTATION,  # MFJ2 - Middle middle
    J_ID_RFPROXIMAL_RFMIDDLE_ROTATION,  # RFJ2 - Ring middle
    J_ID_LFPROXIMAL_LFMIDDLE_ROTATION,  # LFJ2 - Little middle
    J_ID_THHUB_THMIDDLE_ROTATION,       # THJ2 - Thumb middle
]

PROXIMAL_JOINTS = [
    J_ID_FFNUCKLE_FFPROXIMAL_ROTATION,  # FFJ3 - Index proximal
    J_ID_MFNUCKLE_MFPROXIMAL_ROTATION,  # MFJ3 - Middle proximal
    J_ID_RFNUCKLE_RFPROXIMAL_ROTATION,  # RFJ3 - Ring proximal
    J_ID_LFNUCKLE_LFPROXIMAL_ROTATION,  # LFJ3 - Little proximal
    J_ID_THPROXIMAL_THHUB_ROTATION,     # THJ3 - Thumb proximal
]

# Function to filter actuators based on joint lock configuration
def filter_actuators_by_joint_lock(actuator_list, joint_lock='none'):
    """
    Filter out locked joints from the actuator list based on joint_lock configuration.
    
    Args:
        actuator_list: List of joint IDs to potentially actuate
        joint_lock: One of 'none', 'lock_distal_only', 'lock_middle_only', 'lock_proximal_only'
    
    Returns:
        Filtered list of joint IDs that should be actuated
    """
    if joint_lock == 'none':
        return actuator_list
    
    locked_joints = []
    if joint_lock == 'lock_distal_only':
        locked_joints = DISTAL_JOINTS
    elif joint_lock == 'lock_middle_only':
        locked_joints = MIDDLE_JOINTS
    elif joint_lock == 'lock_proximal_only':
        locked_joints = PROXIMAL_JOINTS
    
    # Return only joints that are not in the locked list
    return [j_id for j_id in actuator_list if j_id not in locked_joints]

# ---------------------------------------------- #
#                   KEY MEASURES
# ---------------------------------------------- #

WRIST_PALM_OFFSET = 0.0  # CODE DE MATHILDE : "PALM_WIRST"
EXTREMITY_WIRST = 0.0
EXTREMITY_PALM = 0.145
HALF_PALM_DEPTH = 0.01


MAX_HAND_STANDOFF_ALLEGRO = EXTREMITY_PALM

POSE_RELATIVE_TO_CONTACT_POINT_D_MIN = WRIST_PALM_OFFSET + HALF_PALM_DEPTH
POSE_RELATIVE_TO_CONTACT_POINT_D_MAX = WRIST_PALM_OFFSET + HALF_PALM_DEPTH + MAX_HAND_STANDOFF_ALLEGRO

