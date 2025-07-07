

import time
import numpy as np

from pybullet_utils.bullet_client import BulletClient
import pybullet as p

import environments.src.env_constants as env_consts
import configs.eval_config as eval_cfg

from environments.src.bullet_simulation.simulation_rendering import SimulationRendering
from environments.src.bullet_simulation.simulation_engine import SimulationEngine

import configs.exec_config as exec_cfg


class RobotGrasping:
    def __init__(
            self,
            robot_urdf_path,
            list_id_gripper_fingers,
            list_id_gripper_fingers_actuated,
            gripper_6dof_infos,
            gripper_parameters,
            gripper_default_joint_states,
            object_name,
            max_standoff_gripper,
            wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max,
            display=eval_cfg.BULLET_DEFAULT_DISPLAY_FLG,
            debug=False,
            remove_gripper=False,
            **kwargs
            ):

        self._bullet_client = None  # bullet physics client
        self.physics_client_id = None  # bullet physics client id
        self.sim_render = None  # manage simulation rendering
        self.sim_engine = None  # manage simulation engine

        self._debug = debug
        self.debug_i_debug_bodies = []  # for debugging purpose

        self._init_attributes(
            display=display,
            object_name=object_name,
            robot_urdf_path=robot_urdf_path,
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
            remove_gripper=remove_gripper,
        )


    @property
    def robot_id(self):
        return self.sim_engine.robot_id

    @property
    def obj_id(self):
        return self.sim_engine.obj_id

    @property
    def bullet_client(self):
        return self._bullet_client

    @property
    def debug(self):
        return self._debug

    @property
    def search_space_bb(self):
        return self.sim_engine.search_space_bb

    @property
    def search_space_bb_object(self):
        return self.sim_engine.search_space_bb_object

    @property
    def fingers_joint_infos(self):
        return self.sim_engine.fingers_joint_infos

    @property
    def gripper_6dof_infos(self):
        return self.sim_engine.gripper_6dof_infos

    @property
    def list_id_gripper_fingers(self):
        return self.sim_engine.list_id_gripper_fingers

    @property
    def list_id_gripper_fingers_actuated(self):
        return self.sim_engine.list_id_gripper_fingers_actuated

    @property
    def max_standoff_gripper(self):
        return self.sim_engine.max_standoff_gripper

    @property
    def wrist_palm_offset_gripper(self):
        return self.sim_engine.wrist_palm_offset_gripper

    @property
    def half_palm_depth_offset_gripper(self):
        return self.sim_engine.half_palm_depth_offset_gripper

    @property
    def pose_relative_to_contact_point_d_min(self):
        return self.sim_engine.pose_relative_to_contact_point_d_min

    @property
    def pose_relative_to_contact_point_d_max(self):
        return self.sim_engine.pose_relative_to_contact_point_d_max

    @property
    def path2obj_point_cloud(self):
        return self.sim_engine.path2obj_point_cloud

    @property
    def list_of_points_for_each_triangle_obj_mesh(self):
        return self.sim_engine.list_of_points_for_each_triangle_obj_mesh

    @property
    def object_normals_to_triangles(self):
        return self.sim_engine.object_normals_to_triangles

    @property
    def obj_mesh_vertice_points(self):
        return self.sim_engine.obj_mesh_vertice_points

    def _init_attributes(
            self,
            display,
            object_name,
            robot_urdf_path,
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
            remove_gripper,
    ):

        self._init_bullet_physics_client(display=display)

        sim_engine_kwargs = {
            'robot_urdf_path': robot_urdf_path,
            'object_name': object_name,
            'bullet_client': self._bullet_client,
            'list_id_gripper_fingers': list_id_gripper_fingers,
            'list_id_gripper_fingers_actuated': list_id_gripper_fingers_actuated,
            'gripper_6dof_infos': gripper_6dof_infos,
            'gripper_parameters': gripper_parameters,
            'gripper_default_joint_states': gripper_default_joint_states,
            'max_standoff_gripper': max_standoff_gripper,
            'wrist_palm_offset_gripper': wrist_palm_offset_gripper,
            'half_palm_depth_offset_gripper': half_palm_depth_offset_gripper,
            'pose_relative_to_contact_point_d_min': pose_relative_to_contact_point_d_min,
            'pose_relative_to_contact_point_d_max': pose_relative_to_contact_point_d_max,
        }
        self.sim_engine = SimulationEngine(**sim_engine_kwargs)

        self.sim_engine.reset(bullet_client=self._bullet_client)

        if remove_gripper:
            self._bullet_client.removeBody(self.robot_id)

        self.sim_engine.init_local_sim_save(bullet_client=self._bullet_client)

        self._init_rendering(
            display=display,
        )

    def _init_rendering(self, display):
        self.sim_render = SimulationRendering(bullet_client=self._bullet_client, display=display)

    def _init_bullet_physics_client(self, display):
        self._bullet_client = BulletClient(connection_mode=p.GUI if display else p.DIRECT)
        self.physics_client_id = self._bullet_client._client

    def reset(
            self,
    ):
        # load from local save the initialized scene for faster computation
        self.sim_engine.load_state_from_local_save(
            bullet_client=self._bullet_client,
        )

    def close(self):
        is_bullet_client_on = self.physics_client_id >= 0
        if is_bullet_client_on:
            self._bullet_client.disconnect()
            self.physics_client_id = -1

    def is_debug_mode(self):
        return self._debug

    def delete_debug_bodies(self):
        if len(self.debug_i_debug_bodies) == 0:
            return

        for body_id in self.debug_i_debug_bodies:
            self.bullet_client.removeBody(body_id)

        self.debug_i_debug_bodies = []

    def set_6dof_gripper_pose(self, gripper_6dof_pose):

        self.sim_engine.set_6dof_pose_gripper(
            bullet_client=self._bullet_client,
            start_pos_robot_xyz=gripper_6dof_pose['xyz'],
            start_orient_robot_rpy=gripper_6dof_pose['euler_rpy'],
            start_orient_robot_quat=gripper_6dof_pose['quaternions'],
        )

        self._bullet_client.stepSimulation()

    def _cvt_genome2synergy_label(self, synergy_label, debug=False):
        raise NotImplementedError('Must be overwritten in robot_grasping subclasses.')

    def _cvt_genome2init_joint_states(self, init_joint_state_genes):
        raise NotImplementedError('Must be overwritten in robot_grasping subclasses.')

    def close_gripper(self, robot_id, fingers_joint_infos, synergy_label=None):

        if synergy_label is not None:
            list_id_grip_fingers_actuated = self._cvt_genome2synergy_label(synergy_label)
            list_id_grip_fingers_actuated_non_actuated = list(
                set(self.list_id_gripper_fingers_actuated) - set(list_id_grip_fingers_actuated)
            )
        else:
            list_id_grip_fingers_actuated = self.list_id_gripper_fingers_actuated

        grip_params = self.sim_engine.gripper_parameters

        init_gripper_joint_poses = [0.] * len(self.gripper_6dof_infos)
        max_n_step = grip_params['max_n_step_close_grip']
        max_velocity_gripper = grip_params['max_velocity_maintain_6dof']
        force_gripper = grip_params['force_maintain_6dof']
        is_obj_touched = False

        for i_step in range(max_n_step):

            if not is_obj_touched:
                is_obj_touched = self.sim_engine.are_fingers_touching_object(bullet_client=self._bullet_client)

            if synergy_label is not None:
                # Force non-used fingers to init pose
                for i_finger_joint in list_id_grip_fingers_actuated_non_actuated:
                    self._bullet_client.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=i_finger_joint,
                        controlMode=self._bullet_client.POSITION_CONTROL,
                        targetPosition=self.sim_engine.gripper_default_joint_states[i_finger_joint],
                        maxVelocity=fingers_joint_infos[i_finger_joint]['max_vel'],
                        force=fingers_joint_infos[i_finger_joint]['max_force']
                    )

            # Close fingers
            for i_finger_joint in list_id_grip_fingers_actuated:
                # joint_info = fingers_joint_infos[i_finger_joint]
            
                # if joint_info['low_lim'] < joint_info['up_lim']:
                #     target = joint_info['low_lim']  # Normal case
                # else:
                #     target = joint_info['up_lim']  # Reversed limits case
                
                self._bullet_client.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=i_finger_joint,
                    controlMode=self._bullet_client.POSITION_CONTROL,
                    targetPosition=fingers_joint_infos[i_finger_joint]['low_lim'],
                    maxVelocity=fingers_joint_infos[i_finger_joint]['max_vel'],
                    force=fingers_joint_infos[i_finger_joint]['max_force']
                )

            # Maintain the hand at the 6DoF pose
            for i_grip_joint, gripper_joint_infos in self.gripper_6dof_infos.items():
                joint_target_val = init_gripper_joint_poses[i_grip_joint]
                self._bullet_client.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=i_grip_joint,
                    controlMode=self._bullet_client.POSITION_CONTROL,
                    targetPosition=joint_target_val,
                    maxVelocity=max_velocity_gripper,
                    force=force_gripper
                )

            self._bullet_client.stepSimulation()
            if self.sim_render.display:
                time.sleep(exec_cfg.TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)

        if not is_obj_touched:
            is_obj_touched = self.sim_engine.are_fingers_touching_object(bullet_client=self._bullet_client)

        return is_obj_touched

    def is_grasping_candidate(self):
        return self.sim_engine.is_grasping_candidate(bullet_client=self._bullet_client)

    def apply_all_gripper_shaking(self, gripper_6dof_output_data):
        grip_6dof_infos = self.sim_engine.gripper_6dof_infos
        assert not gripper_6dof_output_data['is_overlap']
        assert gripper_6dof_output_data['is_obj_touched']

        are_all_shakes_successful = True
        for i_grip_joint in env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']:
            gripper_joint_infos = grip_6dof_infos[i_grip_joint]
            is_being_grasped, n_shake_success = self.apply_gripper_shaking(
                joint_index=i_grip_joint,
                gripper_joint_infos=gripper_joint_infos,
            )
            gripper_6dof_output_data['6dof_data'][i_grip_joint]['is_success'] = is_being_grasped
            at_least_one_failure = are_all_shakes_successful and not is_being_grasped
            if at_least_one_failure:
                are_all_shakes_successful = False
            gripper_6dof_output_data['6dof_data'][i_grip_joint]['n_shake_success'] = n_shake_success

        gripper_6dof_output_data['is_success'] = True
        gripper_6dof_output_data['is_robust_grasp'] = are_all_shakes_successful
        return gripper_6dof_output_data

    def apply_gripper_shaking(self, joint_index, gripper_joint_infos):
        n_shake = env_consts.SHAKING_PARAMETERS['n_shake']
        target_position = gripper_joint_infos['joint_target_val']
        cmd_jp_kwargs = self.build_cmd_jg_kwargs(
            joint_index=joint_index, gripper_joint_infos=gripper_joint_infos
        )

        is_being_grasped = True
        i_shake = 0

        target_j_poses = [target_position, -target_position, 0]
        while i_shake < n_shake:
            for j_pose in target_j_poses:
                self.command_joint_pose(target_position=j_pose, **cmd_jp_kwargs)
                if not self.sim_engine.is_grasping(self._bullet_client):
                    is_being_grasped = False
                    return is_being_grasped, i_shake

            i_shake += 1

        return is_being_grasped, i_shake

    def build_cmd_jg_kwargs(self, joint_index, gripper_joint_infos):
        max_velocity = gripper_joint_infos['max_vel']
        force = gripper_joint_infos['force']
        position_gain = gripper_joint_infos['position_gain']
        velocity_gain = gripper_joint_infos['velocity_gain']
        return {
            'joint_index': joint_index,
            'max_velocity': max_velocity,
            'force': force,
            'position_gain': position_gain,
            'velocity_gain': velocity_gain
        }

    def command_joint_pose(
            self,
            joint_index,
            target_position,
            max_velocity,
            force,
            position_gain,
            velocity_gain,
    ):
        for _ in range(env_consts.SHAKING_PARAMETERS['t_cmd_stable']):
            self._bullet_client.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=self._bullet_client.POSITION_CONTROL,
                targetPosition=target_position,
                maxVelocity=max_velocity,
                force=force,
                positionGain=position_gain,
                velocityGain=velocity_gain
            )

            self._bullet_client.stepSimulation()
            if self.sim_render.display:
                time.sleep(exec_cfg.TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)
        return

    def is_there_overlapping(self):
        return self.sim_engine.is_there_overlapping(bullet_client=self._bullet_client)

    def set_joint_states_from_genes(self, init_joint_state_genes):

        joint_ids_to_states = self._cvt_genome2init_joint_states(init_joint_state_genes)

        self.sim_engine.set_robot_joint_states(
            bullet_client=self._bullet_client, joint_ids_to_states=joint_ids_to_states
        )

    def add_noise_to_object_state(self):

        obj_pose_xyz, obj_orient_quat = self.bullet_client.getBasePositionAndOrientation(self.obj_id)

        noise2add_obj_pose_xyz = np.random.normal(
            loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_POS_VARIANCE_IN_M, size=3
        )
        noise2add_obj_orient_euler_rpy = np.random.normal(
            loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_ORIENT_EULER_VARIANCE_IN_RAD, size=3
        )
        noise2add_obj_orient_quat = self.bullet_client.getQuaternionFromEuler(noise2add_obj_orient_euler_rpy)

        noisy_obj_pose_xyz = np.array(obj_pose_xyz) + noise2add_obj_pose_xyz
        noisy_obj_orient_quat = np.array(obj_orient_quat) + noise2add_obj_orient_quat

        self.bullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.obj_id,
            posObj=noisy_obj_pose_xyz,
            ornObj=noisy_obj_orient_quat
        )

    def add_noise_to_friction_coefficients(self):

        noisy_rolling_friction = np.random.uniform(
            low=eval_cfg.DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MIN_VALUE,
            high=eval_cfg.DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MAX_VALUE
        )
        noisy_spinning_friction = np.random.uniform(
            low=eval_cfg.DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MIN_VALUE,
            high=eval_cfg.DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MAX_VALUE
        )

        self.bullet_client.changeDynamics(
            bodyUniqueId=self.obj_id, linkIndex=-1,
            rollingFriction=noisy_rolling_friction,
            spinningFriction=noisy_spinning_friction
        )



