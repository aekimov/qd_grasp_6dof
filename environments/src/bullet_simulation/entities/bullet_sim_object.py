import pdb

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors as Nearest

from environments.src.search_space_bb_processor import get_search_space_bb_object
from algorithms.evaluation.grasp_strategies_routines import convert_mesh_pose_to_inertial_frame

import configs.eval_config as eval_cfg

DEFAULT_OBJECT_POSE = [0, 0, 0]
DEFAULT_OBJECT_ORIENT = [0, 0, 0, 1]


class BulletSimObject:
    def __init__(self, bullet_client, name):

        self.frictions = None  # dict containing friction parameters
        self.object_name = None  # striped object name
        self.obj_id = None  # id associated with the object in the bullet simulation
        self._path2obj_point_cloud = None
        self._obj_mesh_vertice_points = None
        self._list_of_points_for_each_triangle_obj_mesh = None
        self._search_space_bb_object = None
        self._obj_ss_bb_diagonal_dist = None
        self._uniform_obj_contact_points = None
        self._k_tree_uniform_contact_points = None

        self._init_attributes(
            bullet_client=bullet_client,
            name=name
        )

    @property
    def path2obj_point_cloud(self):
        return self._path2obj_point_cloud

    @property
    def obj_mesh_vertice_points(self):
        return self._obj_mesh_vertice_points

    @property
    def object_normals_to_triangles(self):
        return self._object_normals_to_triangles

    @property
    def list_of_points_for_each_triangle_obj_mesh(self):
        return self._list_of_points_for_each_triangle_obj_mesh

    @property
    def search_space_bb_object(self):
        return self._search_space_bb_object

    @property
    def uniform_obj_contact_points(self):
        return self._uniform_obj_contact_points

    @property
    def k_tree_uniform_contact_points(self):
        return self._k_tree_uniform_contact_points

    def _init_attributes(self, bullet_client, name):

        if name is None:
            print('Warning : no given object.')
            return

        self.object_name = self._init_object_name(name)
        self._load_object_bullet(bullet_client=bullet_client)
        self.frictions = self._init_frictions(bullet_client=bullet_client)

        path2obj_point_cloud = self.extract_path2obj_contact_point_cloud_from_urdf(object_name=name)
        precise_vertices_point, mesh, triangles = self.import_point_cloud_from_obj(path2obj_point_cloud)

        path2obj_point_cloud_visual = self.extract_path2obj_visual_point_cloud_from_urdf(object_name=name)
        _, visual_mesh, _ = self.import_point_cloud_from_obj(path2obj_point_cloud_visual)

        if not eval_cfg.LOAD_OBJECT_WITH_FIXED_BASE:
            precise_vertices_point = self.shift_precise_vertices_point_based_on_inertia(precise_vertices_point)

        self._search_space_bb_object = get_search_space_bb_object(obj_id=self.obj_id)
        self._obj_ss_bb_diagonal_dist = np.linalg.norm(
            [self._search_space_bb_object.aabb_max, self._search_space_bb_object.aabb_min]
        )

        self._obj_mesh_vertice_points = precise_vertices_point
        self._list_of_points_for_each_triangle_obj_mesh = self.explicit_triangles(
            triangles=triangles,
            precise_vertices_point=precise_vertices_point,
        )
        self._object_normals_to_triangles = self.find_normal_to_triangles_in_the_object(mesh)

        n_point_mesh_sample = self._get_n_point_mesh_sample()
        print(f'n_point_mesh_sample={n_point_mesh_sample}')
        sampled_point_cloud = np.asarray(visual_mesh.sample_points_uniformly(n_point_mesh_sample).points)
        shifted_sampled_point_cloud = self.shift_precise_vertices_point_based_on_inertia(sampled_point_cloud)
        self._uniform_obj_contact_points = shifted_sampled_point_cloud

        self._k_tree_uniform_contact_points = Nearest(n_neighbors=1, metric='minkowski')
        self._k_tree_uniform_contact_points.fit(shifted_sampled_point_cloud)

    def _load_object_bullet(self, bullet_client):
        self.load_object(
            bullet_client=bullet_client,
            object_name=self.object_name
        )

    def _init_object_name(self, object_name):
        return object_name.strip() if object_name is not None else None

    def get_object_root_path(self, object_name):
        return Path(__file__).parent.parent.parent.parent/"3d_models/objects"/object_name

    def get_path_to_urdf(self, object_name):
        return self.get_object_root_path(object_name)/f"{object_name}.urdf"

    def load_object(self, bullet_client, object_name=None):
        assert isinstance(object_name, str)

        urdf = self.get_path_to_urdf(object_name)
        if not urdf.exists():
            raise ValueError(str(urdf) + " doesn't exist")

        try:
            obj_to_grab_id = bullet_client.loadURDF(
                fileName=str(urdf),
                basePosition=DEFAULT_OBJECT_POSE,
                baseOrientation=DEFAULT_OBJECT_ORIENT,
                useFixedBase=eval_cfg.LOAD_OBJECT_WITH_FIXED_BASE
            )
        except bullet_client.error as e:
            raise bullet_client.error(f"{e}: " + str(urdf))

        bullet_client.changeDynamics(
            bodyUniqueId=obj_to_grab_id,
            linkIndex=-1,
            spinningFriction=eval_cfg.SPINNING_FRICTION_OBJ_DEFAULT_VALUE,
            rollingFriction=eval_cfg.ROLLING_FRICTION_OBJ_DEFAULT_VALUE,
        )

        self.obj_id = obj_to_grab_id

    def _init_frictions(self, bullet_client):
        dynamicsInfo = bullet_client.getDynamicsInfo(self.obj_id, -1)  # save intial friction coefficient of the object
        frictions = {'lateral': dynamicsInfo[1], 'rolling': dynamicsInfo[6], 'spinning': dynamicsInfo[7]}
        return frictions

    def update_infos(self, bullet_client, info):
        is_obj_initialized = self.obj_id is not None
        if is_obj_initialized:
            obj_pose = bullet_client.getBasePositionAndOrientation(self.obj_id)
            obj_vel = bullet_client.getBaseVelocity(self.obj_id)
            info['object position'], info['object xyzw'] = obj_pose
            info['object linear velocity'], info['object angular velocity'] = obj_vel
        return info

    def reset_object_pose(self, bullet_client):

        bullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.obj_id,
            posObj=DEFAULT_OBJECT_POSE,
            ornObj=DEFAULT_OBJECT_ORIENT
        )

    def extract_path2obj_contact_point_cloud_from_urdf(self, object_name):
        path_to_urdf_obj = self.get_path_to_urdf(object_name)
        root_path_to_obj = str(self.get_object_root_path(object_name))

        path2obj_point_cloud = None
        tree = ET.parse(path_to_urdf_obj)
        obj_root = tree.getroot()[0]

        for element in obj_root:
            if 'collision' not in element.tag:
                continue

            for collision_element in element:

                if 'geometry' not in collision_element.tag:
                    continue

                mesh_element = collision_element[0]
                assert mesh_element.tag == 'mesh'
                path2obj_point_cloud = f"{root_path_to_obj}/{mesh_element.attrib['filename']}"

        assert path2obj_point_cloud is not None, \
            f'Invalid urdf structure : cannot extract contact path to point cloud from {path_to_urdf_obj}.'

        return path2obj_point_cloud

    def extract_path2obj_visual_point_cloud_from_urdf(self, object_name):
        path_to_urdf_obj = self.get_path_to_urdf(object_name)
        root_path_to_obj = str(self.get_object_root_path(object_name))

        path2obj_point_cloud = None
        tree = ET.parse(path_to_urdf_obj)
        obj_root = tree.getroot()[0]

        for element in obj_root:
            if 'visual' not in element.tag:
                continue

            for collision_element in element:

                if 'geometry' not in collision_element.tag:
                    continue

                mesh_element = collision_element[0]
                assert mesh_element.tag == 'mesh'
                path2obj_point_cloud = f"{root_path_to_obj}/{mesh_element.attrib['filename']}"

        assert path2obj_point_cloud is not None, \
            f'Invalid urdf structure : cannot extract contact path to point cloud from {path_to_urdf_obj}.'

        return path2obj_point_cloud

    def extract_inertial_pose_from_urdf(self, object_name):

        path_to_urdf_obj = self.get_path_to_urdf(object_name)
        root_path_to_obj = str(self.get_object_root_path(object_name))

        inertial_pose = None
        tree = ET.parse(path_to_urdf_obj)
        obj_root = tree.getroot()[0]

        for element in obj_root:
            if 'inertial' not in element.tag:
                continue

            for inertial_element in element:
                if 'origin' not in inertial_element.tag:
                    continue

                inertial_pose_str = inertial_element.attrib['xyz'].split(' ')
                inertial_pose = np.array([float(pose_str) for pose_str in inertial_pose_str])

        return inertial_pose

    def import_point_cloud_from_obj(self, path2obj_point_cloud):
        mesh = o3d.io.read_triangle_mesh(path2obj_point_cloud)
        array_vertices = np.asarray(mesh.vertices)
        array_triangle = np.asarray(mesh.triangles)
        return array_vertices, mesh, array_triangle

    def explicit_triangles(self, triangles, precise_vertices_point):
        list_of_points_in_a_triangle = []
        for t in range(len(triangles)):
            triangle_indexes = triangles[t]
            A_idx, B_idx, C_idx = triangle_indexes
            A_xyz = precise_vertices_point[A_idx].tolist()
            B_xyz = precise_vertices_point[B_idx].tolist()
            C_xyz = precise_vertices_point[C_idx].tolist()
            list_of_points_in_a_triangle.append([A_xyz, B_xyz, C_xyz])

        return list_of_points_in_a_triangle

    def find_normal_to_triangles_in_the_object(self, mesh):
        mesh.compute_vertex_normals()
        array_normals_to_triangles = np.asarray(mesh.triangle_normals)
        return array_normals_to_triangles

    def shift_precise_vertices_point_based_on_inertia(self, precise_vertices_point):

        inertial_pose = self.extract_inertial_pose_from_urdf(self.object_name)
        shifted_vertices_point = convert_mesh_pose_to_inertial_frame(
            obj_inertial_pose=inertial_pose, meshpose2cvt=precise_vertices_point
        )
        return shifted_vertices_point

    def _get_n_point_mesh_sample(self):
        # defined ratio : 0.2m => 2000 points
        n_point_mesh_sample = int(self._obj_ss_bb_diagonal_dist * 2000 / 0.2)
        return n_point_mesh_sample