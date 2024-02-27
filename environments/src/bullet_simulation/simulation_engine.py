
import os
import numpy as np
from pathlib import Path

from environments.src.search_space_bb_processor import get_search_space_bb, get_search_space_bb_side
from environments.src.bullet_simulation.entities.bullet_sim_object import BulletSimObject

DEFAULT_ROBOT_POSE = [0, 0, 0]
DEFAULT_ROBOT_ORIENT = [0, 0, 0, 1]


class SimulationEngine:
    def __init__(
            self,
            robot_urdf_path,
            object_name,
            bullet_client,
            list_id_gripper_fingers,
            list_id_gripper_fingers_actuated,
            gripper_6dof_infos,
            gripper_parameters,
            gripper_default_joint_states,
            max_standoff_gripper,
            wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max,
    ):
        self.init_state_p_file_root = None  # local save of bullet sim config for quick reinitialization
        self.bullet_sim_obj = None  # manage bullet simulation object to grasp
        self._robot_id = None
        self._search_space_bb = None
        self._search_space_bb_side = None
        self._fingers_joint_infos = None
        self._gripper_6dof_infos = None
        self._gripper_parameters = None
        self._gripper_default_joint_states = None
        self._list_id_gripper_fingers = None
        self._list_id_gripper_fingers_actuated = None
        self._max_standoff_gripper = None
        self._wrist_palm_offset_gripper = None
        self._half_palm_depth_offset_gripper = None
        self._pose_relative_to_contact_point_d_min = None
        self._pose_relative_to_contact_point_d_max = None

        self._init_attributes(
            robot_urdf_path=robot_urdf_path,
            object_name=object_name,
            bullet_client=bullet_client,
            list_id_gripper_fingers=list_id_gripper_fingers,
            list_id_gripper_fingers_actuated=list_id_gripper_fingers_actuated,
            gripper_6dof_infos=gripper_6dof_infos,
            gripper_parameters=gripper_parameters,
            gripper_default_joint_states=gripper_default_joint_states,
            max_standoff_gripper=max_standoff_gripper,
            wrist_palm_offset_gripper=wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper=half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min=pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max=pose_relative_to_contact_point_d_max,
        )

    @property
    def obj_id(self):
        return self.bullet_sim_obj.obj_id

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def search_space_bb(self):
        return self._search_space_bb

    @property
    def search_space_bb_side(self):
        return self._search_space_bb_side

    @property
    def search_space_bb_object(self):
        return self.bullet_sim_obj.search_space_bb_object

    @property
    def fingers_joint_infos(self):
        return self._fingers_joint_infos

    @property
    def gripper_6dof_infos(self):
        return self._gripper_6dof_infos

    @property
    def gripper_default_joint_states(self):
        return self._gripper_default_joint_states

    @property
    def max_standoff_gripper(self):
        return self._max_standoff_gripper

    @property
    def wrist_palm_offset_gripper(self):
        return self._wrist_palm_offset_gripper

    @property
    def half_palm_depth_offset_gripper(self):
        return self._half_palm_depth_offset_gripper

    @property
    def pose_relative_to_contact_point_d_min(self):
        return self._pose_relative_to_contact_point_d_min

    @property
    def pose_relative_to_contact_point_d_max(self):
        return self._pose_relative_to_contact_point_d_max

    @property
    def gripper_parameters(self):
        return self._gripper_parameters

    @property
    def path2obj_point_cloud(self):
        return self.bullet_sim_obj.path2obj_point_cloud

    @property
    def list_of_points_for_each_triangle_obj_mesh(self):
        return self.bullet_sim_obj.list_of_points_for_each_triangle_obj_mesh

    @property
    def object_normals_to_triangles(self):
        return self.bullet_sim_obj.object_normals_to_triangles

    @property
    def obj_mesh_vertice_points(self):
        return self.bullet_sim_obj.obj_mesh_vertice_points

    @property
    def list_id_gripper_fingers(self):
        return self._list_id_gripper_fingers

    @property
    def list_id_gripper_fingers_actuated(self):
        return self._list_id_gripper_fingers_actuated

    @property
    def uniform_obj_contact_points(self):
        return self.bullet_sim_obj.uniform_obj_contact_points

    @property
    def k_tree_uniform_contact_points(self):
        return self.bullet_sim_obj.k_tree_uniform_contact_points

    def _init_attributes(
            self,
            robot_urdf_path,
            object_name,
            bullet_client,
            list_id_gripper_fingers,
            list_id_gripper_fingers_actuated,
            gripper_6dof_infos,
            gripper_parameters,
            gripper_default_joint_states,
            max_standoff_gripper,
            wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max,
    ):
        self._gripper_6dof_infos = gripper_6dof_infos
        self._gripper_parameters = gripper_parameters
        self._gripper_default_joint_states = gripper_default_joint_states
        self._list_id_gripper_fingers = list_id_gripper_fingers
        self._list_id_gripper_fingers_actuated = list_id_gripper_fingers_actuated
        self._max_standoff_gripper = max_standoff_gripper
        self._wrist_palm_offset_gripper = wrist_palm_offset_gripper
        self._half_palm_depth_offset_gripper = half_palm_depth_offset_gripper
        self._pose_relative_to_contact_point_d_min = pose_relative_to_contact_point_d_min
        self._pose_relative_to_contact_point_d_max = pose_relative_to_contact_point_d_max

        if max_standoff_gripper is None or wrist_palm_offset_gripper is None or half_palm_depth_offset_gripper is None:
            raise NotImplementedError(
                f'Both max_standoff_gripper and wrist_palm_offset_gripper must be defined for the given gripper.'
            )

        bullet_client.resetSimulation()
        bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        object_kwargs = {
            'bullet_client': bullet_client,
            'name': object_name,
        }
        self.bullet_sim_obj = BulletSimObject(**object_kwargs)

        self._robot_id = bullet_client.loadURDF(
            fileName=robot_urdf_path,
            basePosition=np.array(DEFAULT_ROBOT_POSE) + [0, 0, 0.8],
            baseOrientation=DEFAULT_ROBOT_ORIENT,
            useFixedBase=True
        )
        self._search_space_bb = get_search_space_bb(obj_id=self.obj_id, robot_id=self._robot_id)
        self._search_space_bb_side = get_search_space_bb_side(self._search_space_bb)

        self._fingers_joint_infos = self._init_fingers_joint_infos(
            bullet_client=bullet_client, list_id_gripper_fingers=list_id_gripper_fingers
        )

        self._reset_robot_fingers(bullet_client=bullet_client)

    def load_state_from_local_save(self, bullet_client):
        assert self.init_state_p_file_root, 'bullet tmp file not properly set'
        file2load = self.init_state_p_file_root
        bullet_client.restoreState(fileName=file2load)

    def init_local_sim_save(self, bullet_client):

        save_folder_root = os.getcwd() + '/tmp'

        self.init_state_p_file_root = save_folder_root + "/init_state.bullet"

        Path(save_folder_root).mkdir(exist_ok=True)

        # Make sure each worker (cpu core) has its own local save
        init_state_p_local_pid_file = self.init_state_p_file_root
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f'dir_path = {dir_path}')
        print('trying to save = ', init_state_p_local_pid_file)
        bullet_client.saveBullet(init_state_p_local_pid_file)
        print(f'init_state_p_local_pid_file={init_state_p_local_pid_file} successfully saved.')

        # Create a unique reference file that is erased each time. Usefull to rely on a unique file if necessary (i.e.
        # in plot_trajectory, that parallelize re-evaluation with Pool and display on a single cpu the results)
        init_state_p_local_pid_file = self.init_state_p_file_root
        bullet_client.saveBullet(init_state_p_local_pid_file)

    def _init_fingers_joint_infos(self, bullet_client, list_id_gripper_fingers):
        assert self._robot_id is not None
        fingers_joint_infos = {}
        for i_grip in list_id_gripper_fingers:
            j_infos = bullet_client.getJointInfo(self._robot_id, i_grip)
            j_low_lim, j_up_lim = j_infos[8], j_infos[9]
            joint_max_force, joint_max_velocity = j_infos[10], j_infos[11]
            fingers_joint_infos[i_grip] = {
                'low_lim': j_low_lim,
                'up_lim': j_up_lim,
                'max_force': joint_max_force,
                'max_vel': joint_max_velocity,
            }
        return fingers_joint_infos

    def _reset_robot_fingers(self, bullet_client):
        assert self.robot_id is not None
        assert self._fingers_joint_infos is not None
        assert self._list_id_gripper_fingers_actuated is not None

        # By default, set at max pose w.r.t. urdf file
        for i_grip in self._list_id_gripper_fingers_actuated:
            bullet_client.resetJointState(
                self.robot_id,
                jointIndex=i_grip,
                targetValue=self._fingers_joint_infos[i_grip]['up_lim']
            )

        # Hand fix joints poses is necessary
        for i_grip in self._gripper_default_joint_states:
            bullet_client.resetJointState(
                self.robot_id,
                jointIndex=i_grip,
                targetValue=self._gripper_default_joint_states[i_grip]
            )

    def set_6dof_pose_gripper(
            self, bullet_client, start_pos_robot_xyz, start_orient_robot_rpy, start_orient_robot_quat
    ):
        if start_orient_robot_rpy is not None:
            start_orient_robot_quaternion = bullet_client.getQuaternionFromEuler(start_orient_robot_rpy)
        elif start_orient_robot_quat is not None:
            start_orient_robot_quaternion = start_orient_robot_quat
        else:
            raise AttributeError('start_orient_robot_rpy or start_orient_robot_quat must be != None to set pose.')

        bullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.robot_id,
            posObj=start_pos_robot_xyz,
            ornObj=start_orient_robot_quaternion
        )

    def are_fingers_touching_object(self, bullet_client):

        for finger_id in self._list_id_gripper_fingers:
            contact_pts = bullet_client.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=finger_id
            )
            if self.is_there_contacts(contact_pts):
                return True
        return False

    def is_grasping_candidate(self, bullet_client):
        return self.are_fingers_touching_object(bullet_client=bullet_client)

    def is_there_contacts(self, contacts):
        return len(contacts) != 0

    def is_grasping(self, bullet_client):
        contacts = bullet_client.getContactPoints(self._robot_id)
        return self.is_there_contacts(contacts)

    def reset(self, bullet_client):
        self._reset_robot_fingers(bullet_client=bullet_client)
        self.bullet_sim_obj.reset_object_pose(bullet_client=bullet_client)

    def is_there_overlapping(self, bullet_client):
        contacts_robot = bullet_client.getContactPoints(bodyA=self._robot_id)
        id_arg_contact_dist = 8
        if self.is_there_contacts(contacts_robot):
            for i in range(len(contacts_robot)):
                contact_distance = contacts_robot[i][id_arg_contact_dist]
                if contact_distance < 0:
                    is_there_overlap = True
                    return is_there_overlap
        is_there_overlap = False
        return is_there_overlap

    def is_there_object_robot_overlapping(self, bullet_client):
        contacts = bullet_client.getContactPoints(bodyA=self.obj_id, bodyB=self.robot_id)
        is_there_overlap = len(contacts) != 0
        return is_there_overlap

    def set_robot_joint_states(self, bullet_client, joint_ids_to_states):
        assert self.robot_id is not None

        # Hand fix joints poses is necessary
        for i_grip in joint_ids_to_states:
            bullet_client.resetJointState(
                self.robot_id,
                jointIndex=i_grip,
                targetValue=joint_ids_to_states[i_grip]
            )
