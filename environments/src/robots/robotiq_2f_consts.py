
import numpy as np


GRIPPER_6DOF_INFOS = {
    0: {'axis': 'x', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2}, # rouge
    1: {'axis': 'y', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2},  # vert
    2: {'axis': 'z', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82,  'velocity_gain': 64, 'joint_target_val': 0.2},  # bleu

    3: {'axis': 'x', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    4: {'axis': 'y', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    5: {'axis': 'z', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
}

LIST_ID_GRIPPER_6DOF = list(GRIPPER_6DOF_INFOS.keys())

# WITH OUTER KNUCKLES
# 6 (finger_joint (on aurait pu l'appeler left_outer_knuckle_joint ) et 12 (right_outer_knuckle_joint) : base des doigts
# 8 (left_inner_knuckle_joint) et 10 (right_inner_knuckle_joint) : intermédiaire des doigts
# 9 (left_inner_finger_joint) 11 (right_inner_finger_joint) : bout des doigts
#LIST_ID_GRIPPER_FINGERS = [6, 7, 8, 9, 10, 11, 12, 13]  # ids associated with the gripper's bodies
#LIST_ID_GRIPPER_FINGERS_ACTUATORS = [6, 7, 8, 9, 10, 11, 12, 13] #[6, 8, 9, 10, 11, 12]  # ids to trigger for closing the gripper

# WITHOUT OUTER KNUCKLES
# 6 (left_inner_knuckle_joint) et 8 (right_inner_knuckle_joint) : intermédiaire des doigts
# 7 (left_inner_finger_joint) 9 (right_inner_finger_joint) : bout des doigts

LIST_ID_GRIPPER_FINGERS = [6, 7, 8, 9]  # ids associated with the gripper's bodies
LIST_ID_GRIPPER_FINGERS_ACTUATORS = [6, 7, 8, 9] #[6, 8, 9, 10, 11, 12]  # ids to trigger for closing the gripper


GRIPPER_PARAMETERS = {
    'max_n_step_close_grip': 100,
    'max_velocity_maintain_6dof': 10,
    'force_maintain_6dof': 100
}


# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #

# Manually force joint poses to initialize them at a different pose that the associated upper bound value
DEFAULT_JOINT_STATES = {}
"""
DEFAULT_JOINT_STATES = {
    10: 0.04,
    11: 0.04
}
"""

# ---------------------------------------------- #
#                    3D Models
# ---------------------------------------------- #

ROBOTIQ_2_FINGERS_85_GRIP_RELATIVE_PATH_URDF = "robotiq_85_gripper.urdf"


WRIST_PALM_OFFSET = 0.088  # CODE DE MATHILDE : "PALM_WIRST"
EXTREMITY_PALM = (149.3-88)*0.001
HALF_PALM_DEPTH = 0.0001


MAX_HAND_STANDOFF_ROBOTIQ = EXTREMITY_PALM

POSE_RELATIVE_TO_CONTACT_POINT_D_MIN = 0 #WRIST_PALM_OFFSET + HALF_PALM_DEPTH
POSE_RELATIVE_TO_CONTACT_POINT_D_MAX = MAX_HAND_STANDOFF_ROBOTIQ #WRIST_PALM_OFFSET + HALF_PALM_DEPTH + MAX_HAND_STANDOFF_ROBOTIQ




