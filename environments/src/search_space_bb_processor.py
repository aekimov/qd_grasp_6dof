
import numpy as np
import pybullet as p

from collections import namedtuple
import configs.qd_config as qd_cfg

SearchSpaceBB = namedtuple("SearchSpaceBB", "aabb_min aabb_max")


interval_genome = [-qd_cfg.GENOTYPE_MAX_VAL, qd_cfg.GENOTYPE_MAX_VAL]
interval_stop_euler = [-np.pi, np.pi]


def get_search_space_bb(obj_id, robot_id):
    object_plus_robot_bb = True
    if object_plus_robot_bb:
        return get_search_space_bb_object_plus_robot(obj_id=obj_id, robot_id=robot_id)
    else:
        return get_search_space_bb_object(obj_id=obj_id)


def get_search_space_bb_side(ss_bb):
    all_ssbb_sides = np.array(ss_bb.aabb_max) - np.array(ss_bb.aabb_min)
    ssbb_side = all_ssbb_sides.max()
    return ssbb_side


def get_search_space_bb_object_plus_robot(obj_id, robot_id):
    robot_aabb_min, robot_aabb_max = big_boundary_box(robot_id)
    obj_aabb_min, obj_aabb_max = big_boundary_box(obj_id)

    aabb_min_tuple = np.array([robot_aabb_min, obj_aabb_min]).sum(axis=0)
    aabb_max_tuple = np.array([robot_aabb_max, obj_aabb_max]).sum(axis=0)
    l_max = (np.array(aabb_max_tuple) - np.array(aabb_min_tuple)).max()  # <===
    bb_centroid = (np.array(obj_aabb_min) + np.array(obj_aabb_max)) / 2

    min_x, max_x = bb_centroid[0] - l_max / 2, bb_centroid[0] + l_max / 2
    min_y, max_y = bb_centroid[1] - l_max / 2, bb_centroid[1] + l_max / 2
    min_z, max_z = bb_centroid[2] - l_max / 2, bb_centroid[2] + l_max / 2
    aabb_min = [min_x, min_y, min_z]
    aabb_max = [max_x, max_y, max_z]
    ss_bb = SearchSpaceBB(aabb_min=aabb_min, aabb_max=aabb_max)
    return ss_bb


def get_search_space_bb_object(obj_id):
    aabb_min_tuple, aabb_max_tuple = p.getAABB(obj_id)
    aabb_min = list(aabb_min_tuple)
    aabb_max = list(aabb_max_tuple)
    ss_bb = SearchSpaceBB(aabb_min=aabb_min, aabb_max=aabb_max)
    return ss_bb


def big_boundary_box(body, draw=False):
    n_joints = p.getNumJoints(body)
    if n_joints > 0:
        all_min_tuple, all_max_tuple = [], []
        for j_id in range(n_joints):
            j_aabb_min_tuple, j_aabb_max_tuple = p.getAABB(body, linkIndex=j_id)
            all_min_tuple.append(j_aabb_min_tuple)
            all_max_tuple.append(j_aabb_max_tuple)

        aabb_max_tuple = np.array(all_max_tuple).max(axis=0)
        aabb_min_tuple = np.array(all_min_tuple).min(axis=0)
    else:
        aabb_min_tuple, aabb_max_tuple = p.getAABB(body)

    l_max = (np.array(aabb_max_tuple) - np.array(aabb_min_tuple)).max()
    bb_centroid = (np.array(aabb_max_tuple) + np.array(aabb_min_tuple)) / 2

    min_x, max_x = bb_centroid[0] - l_max / 2, bb_centroid[0] + l_max / 2
    min_y, max_y = bb_centroid[1] - l_max / 2, bb_centroid[1] + l_max / 2
    min_z, max_z = bb_centroid[2] - l_max / 2, bb_centroid[2] + l_max / 2
    aabb_min = [min_x, min_y, min_z]
    aabb_max = [max_x, max_y, max_z]

    return aabb_min, aabb_max


