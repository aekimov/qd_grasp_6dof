
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

LIST_ID_GRIPPER_FINGERS = [6, 7]
LIST_ID_GRIPPER_FINGERS_ACTUATORS = [6, 7]

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


# ---------------------------------------------- #
#                    3D Models
# ---------------------------------------------- #

PANDA_2_FINGERS_GRIP_RELATIVE_PATH_URDF = "panda_gripper.urdf"


# ---------------------------------------------- #
#                   KEY MEASURES
# ---------------------------------------------- #

WRIST_PALM_OFFSET = 0.066
MAX_HAND_STANDOFF_PANDA = 0.046
HALF_PALM_DEPTH = 0.


POSE_RELATIVE_TO_CONTACT_POINT_D_MIN = 0.
POSE_RELATIVE_TO_CONTACT_POINT_D_MAX = MAX_HAND_STANDOFF_PANDA



