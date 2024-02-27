import pdb

import numpy as np
from numpy import linalg as la
import astropy.coordinates
from transforms3d import euler
from pyquaternion import Quaternion
import math

import environments.src.env_constants as env_consts

EPSILON_NUMERICAL_STABILITY = 0.0000001

PATH_TO_GRASPING_STRATEGIES_3D_MODELS = "algorithms/evaluation/grasping_strategies_3d_models/"
PATH_TO_SMALL_RED_SPHERE_URDF = PATH_TO_GRASPING_STRATEGIES_3D_MODELS + "sphere.urdf"
PATH_TO_SMALL_YELLOW_SPHERE_URDF = PATH_TO_GRASPING_STRATEGIES_3D_MODELS + "sphere3.urdf"
PATH_TO_MEDIUM_MAGENTA_SPHERE_URDF = PATH_TO_GRASPING_STRATEGIES_3D_MODELS + "sphere4.urdf"
PATH_SMALL_GREY_SPHERE_URDF = PATH_TO_GRASPING_STRATEGIES_3D_MODELS + "contact_sphere9.urdf"
PATH_TO_CYLINDER_WITH_LINKS_URDF = PATH_TO_GRASPING_STRATEGIES_3D_MODELS + "cylinder_with_links.urdf"

DUMMY_QUATERNION = [0, 0, 0, 1]

ENTRAXE_DIST_PANDA_2F = 0.08


def display_mesh_point_from_array_debug(bullet_client, object_array_points, debug_obj=PATH_TO_SMALL_RED_SPHERE_URDF):
    for point_in_object_array_points in object_array_points:
        xyz = point_in_object_array_points
        sphere = bullet_client.loadURDF(debug_obj, xyz, DUMMY_QUATERNION)


def get_rotation_matrix_from_an_axis_and_an_angle(direction_vector, theta):
    """See: https://www.techno-science.net/glossaire-definition/Rotation-vectorielle.html"""
    ux, uy, uz = direction_vector

    P = np.array([
        [ux ** 2, ux * uy, ux * uz],
        [ux * uy, uy ** 2, uy * uz],
        [ux * uz, uy * uz, uz ** 2]
    ])
    I = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    Q = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    R = P + (I - P) * np.cos(theta) + Q * np.sin(theta)
    return R


def rotate_vector_from_axis_and_angle(rotation_axis, theta, vector_u):
    """Rodrigues-Euler formula. Apply a rotation on vector_u that is defined with the reference axis rotation_axis and
     the angle theta."""

    # Get the rotation matrix
    rotation_matrix = get_rotation_matrix_from_an_axis_and_an_angle(direction_vector=rotation_axis, theta=theta)

    # Apply theta rotation on vector_u around rotation_axis
    point_generator_prime_np_v1 = np.dot(rotation_matrix, vector_u)

    return point_generator_prime_np_v1


def generate_orthogonal_vector(ref_vector):
    """Generate a vector that is orthogonal to ref_vector.

    Assuming:
        u, v = ref_vector, orthogonal_vector
        u.v = 0 <=> u.x * v.x + u.y * v.y + u.z * v.z = 0

    By fixing x = 1 and y = 1 (and assuming u.z is not 0) we have:
        v.z = (- u.x * v.x - u.y * v.y) / u.z
            = (- u.x - u.y) / u.z


    """
    x, y = 1, 1
    z = (-ref_vector[0] - ref_vector[1]) / (ref_vector[2] + EPSILON_NUMERICAL_STABILITY)
    orthogonal_vector = np.array([x, y, z])
    return orthogonal_vector


def generate_conic_surface(bullet_client, contact_point, y_vector_normalized_B1, theta_rad, x_vector_B1_prime, debug):
    # step 3 : rotation of x of the base B1 of pi/4 around y of the base B1
    contact_point_np = np.array(contact_point)
    x_prime_prime_B1 = rotate_vector_from_axis_and_angle(
        rotation_axis=y_vector_normalized_B1, theta=theta_rad, vector_u=x_vector_B1_prime
    )

    if debug:
        bullet_client.addUserDebugLine(
            lineFromXYZ=contact_point_np,
            lineToXYZ=x_prime_prime_B1,
            lineColorRGB=[0, 255, 255],
            lineWidth=1
        )
    return x_prime_prime_B1


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def get_gripper_pose_relative_to_contact_normal(
        bullet_client,
        normal_at_contact_point,
        standoff_from_palm,
        contact_point,
        half_aperture_complementary,
        debug,
        ksi
):
    # Create a base B1 = (normal_at_contact_point_normalized, x_vector_normalized_B1, y_vector_normalized_B1),
    # where B1 is orthonormal, normal_at_contact_point_normalized is the normal to the surface at contact point, and
    # x_vector_normalized_B1 and y_vector_normalized_B1 are tangent vector to the surface at contact point.
    normal_at_contact_point_normalized = normalize_vector(normal_at_contact_point)

    x_vector_B1 = generate_orthogonal_vector(normal_at_contact_point_normalized)
    x_vector_normalized_B1 = normalize_vector(x_vector_B1)

    # creation of y_vector_normalized_B1
    y_vector_B1 = np.cross(x_vector_normalized_B1, normal_at_contact_point_normalized)
    y_vector_normalized_B1 = normalize_vector(y_vector_B1)

    # Apply standoff translation (search variable: d)
    # x -> x'
    x_vector_B1_prime = x_vector_normalized_B1 * standoff_from_palm

    # Rotate the vector from the normal based on the cone aperture (search variable: nu)
    # x' -> x''
    x_prime_prime_B1 = generate_conic_surface(
        bullet_client=bullet_client,
        contact_point=contact_point,
        y_vector_normalized_B1=y_vector_normalized_B1,
        theta_rad=half_aperture_complementary,
        x_vector_B1_prime=x_vector_B1_prime,
        debug=debug,
    )

    # Rotate the vector around the normal to position it on the cone surface (search variable: ksi)
    # x'' -> x'''
    x_prime_prime_prime_B1 = rotate_vector_from_axis_and_angle(
        rotation_axis=normal_at_contact_point_normalized,
        theta=ksi,
        vector_u=x_prime_prime_B1
    )
    gripper_pose_relative_to_contact_normal = x_prime_prime_prime_B1 + np.array(contact_point)

    return gripper_pose_relative_to_contact_normal


def get_normal_surface_point(
        bullet_client,
        contact_point,
        list_of_points_for_each_triangle_object_mesh,
        object_normals_to_triangles,
        debug
):
    normal_at_contact_point = calculate_normal_from_surface(
        bullet_client=bullet_client,
        list_of_points_for_each_triangle=list_of_points_for_each_triangle_object_mesh,
        contact_point=contact_point,
        object_normals_to_triangles=object_normals_to_triangles,
        debug=debug,
    )

    return normal_at_contact_point


def find_normal_to_vertices_in_the_object(mesh):
    mesh.compute_vertex_normals()
    array_normals_to_triangles = np.asarray(mesh.triangle_normals)
    return array_normals_to_triangles


def get_normal_to_surface_at_contact_point(object_normals_to_triangles, id_min_dist_triangle):
    return object_normals_to_triangles[id_min_dist_triangle]


def display_normal_to_contact_point_debug(bullet_client, normal_at_contact_point, sphere_closest_point, color_display):
    # The line can start from [0, 0, 0] as it is relative to parentObjectUniqueId=sphere_closest_point.
    bullet_client.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=normal_at_contact_point,
        parentObjectUniqueId=sphere_closest_point,
        parentLinkIndex=-1,
        lineColorRGB=color_display,
        lineWidth=0.3,
        lifeTime=4

    )


def calculate_normal_from_surface(bullet_client, list_of_points_for_each_triangle, contact_point, object_normals_to_triangles, debug=True):

    id_min_dist_triangle, sphere_closest_point = find_nearest_point_in_the_object(
        bullet_client=bullet_client,
        list_of_points_for_each_triangle=list_of_points_for_each_triangle,
        contact_point_array=contact_point,
        debug=debug
    )

    normal_at_contact_point = get_normal_to_surface_at_contact_point(object_normals_to_triangles, id_min_dist_triangle)

    if debug:
        display_normal_to_contact_point_debug(
            bullet_client=bullet_client,
            normal_at_contact_point=normal_at_contact_point,
            sphere_closest_point=sphere_closest_point,
            color_display=[0, 255, 0]
        )

    return normal_at_contact_point


def pointTriangleDistance(TRI, P):
    """Comes from: https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e"""

    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = np.sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1
    return dist, PP0


def get_point_to_triangle_dist(triangle_vertices_poses, point):
    dist, closest_point_in_triangle = pointTriangleDistance(TRI=triangle_vertices_poses, P=point)
    return dist, closest_point_in_triangle


def find_nearest_point_in_the_object(bullet_client, list_of_points_for_each_triangle, contact_point_array, debug=True):

    all_dists, all_closest_point_in_triangles = zip(*[
        get_point_to_triangle_dist(
            triangle_vertices_poses=np.array(list_of_points_for_each_triangle[i]),
            point=contact_point_array
        ) for i in range(len(list_of_points_for_each_triangle))
    ])
    min_dist, id_min_dist_triangle = np.min(all_dists), np.argmin(all_dists)
    closest_point_in_triangle_min_dist = all_closest_point_in_triangles[id_min_dist_triangle]
    closest_triangle = list_of_points_for_each_triangle[id_min_dist_triangle]

    if debug:
        display_closest_triangle_debug(bullet_client=bullet_client, closest_triangle=closest_triangle)

    sphere_closest_point = display_closest_point_in_triangle_debug(
        bullet_client=bullet_client, closest_point_in_triangle_min_dist=closest_point_in_triangle_min_dist) if debug \
        else None

    return id_min_dist_triangle, sphere_closest_point


def display_closest_point_in_triangle_debug(bullet_client, closest_point_in_triangle_min_dist):
    sphere_obj = bullet_client.loadURDF(
        PATH_TO_MEDIUM_MAGENTA_SPHERE_URDF, closest_point_in_triangle_min_dist, DUMMY_QUATERNION
    )
    return sphere_obj


def display_closest_triangle_debug(bullet_client, closest_triangle):
    bullet_client.loadURDF(PATH_TO_SMALL_YELLOW_SPHERE_URDF, closest_triangle[0], DUMMY_QUATERNION)
    bullet_client.loadURDF(PATH_TO_SMALL_YELLOW_SPHERE_URDF, closest_triangle[1], DUMMY_QUATERNION)
    bullet_client.loadURDF(PATH_TO_SMALL_YELLOW_SPHERE_URDF, closest_triangle[2], DUMMY_QUATERNION)
    bullet_client.addUserDebugLine(closest_triangle[0], closest_triangle[1])
    bullet_client.addUserDebugLine(closest_triangle[1], closest_triangle[2])
    bullet_client.addUserDebugLine(closest_triangle[2], closest_triangle[0])


def get_gripper_orient_relative_to_contact_normal(gripper_wrist_pose, gripper_tip_pose, omega_rad):

    # Get the vector corresponding to grip wrist to tip
    gripper_wrist2tip_cartesian = gripper_tip_pose - gripper_wrist_pose

    # Convert it to spherical to get compute the corresponding quaternion
    gripper_wrist2tip_spherical = astropy.coordinates.cartesian_to_spherical(
        x=gripper_wrist2tip_cartesian[0], y=gripper_wrist2tip_cartesian[1], z=gripper_wrist2tip_cartesian[2]
    )
    latitude = gripper_wrist2tip_spherical[1].rad
    longitude = gripper_wrist2tip_spherical[2].rad
    gripper_wrist2tip_quat = euler.euler2quat(np.pi + longitude, np.pi / 2 - latitude, np.pi, axes='sxyz')

    # Rotate hand quaternion by omega_rad around the X axis
    omega_rotation_quat = Quaternion(axis=[1, 0, 0], angle=omega_rad)  # omega_rad rotation around X
    rotated_gripper_orient_quat = omega_rotation_quat * gripper_wrist2tip_quat  # combine quaternion rotation

    # Return result as array
    gripper_wrist_orient_quat = rotated_gripper_orient_quat.elements
    return gripper_wrist_orient_quat


def get_gripper_pose_relatively_to_contact_point(
        bullet_client,
        wrist_palm_offset_gripper,
        hand_pose_from_contact_params,
        normal_at_contact_point,
        contact_point,
        debug
):

    half_aperture = hand_pose_from_contact_params['nu']
    standoff = hand_pose_from_contact_params['d']
    ksi = hand_pose_from_contact_params['ksi']
    wrist_rotation = hand_pose_from_contact_params['omega']

    # Vector orientation is defined with the complementary angle of the half aperture
    half_aperture_complementary = np.pi / 2 - half_aperture

    # The standoff search space is limited to the palm-hand distance, but the hand pose is defined at wrist pose
    standoff_from_palm = standoff + wrist_palm_offset_gripper

    # Get gripper position
    gripper_wrist_pose = get_gripper_pose_relative_to_contact_normal(
        bullet_client=bullet_client,
        normal_at_contact_point=normal_at_contact_point,
        standoff_from_palm=standoff_from_palm,
        contact_point=contact_point,
        half_aperture_complementary=half_aperture_complementary,
        debug=debug,
        ksi=ksi
    )

    # Get gripper orientation
    gripper_wrist_orient_quat = get_gripper_orient_relative_to_contact_normal(
        gripper_wrist_pose=gripper_wrist_pose, gripper_tip_pose=contact_point, omega_rad=wrist_rotation
    )

    gripper_6dof_pose = {
        'xyz': gripper_wrist_pose.tolist(),
        'euler_rpy': None,
        'quaternions': gripper_wrist_orient_quat.tolist()
    }
    return gripper_6dof_pose


def get_90_degres_rotation(omega, rot_axis):
    q_rot_xyzw = np.array((np.sin(omega / 2) * np.array(rot_axis)).tolist() + [np.cos(omega / 2)])
    return q_rot_xyzw


def get_quaternion_oriented_to_the_point(wrist_RO):
    c1 = np.array([0, 0, 0])
    center_axis_R0 = (wrist_RO-c1)
    center_axis_normalized_R0 = center_axis_R0/np.linalg.norm(center_axis_R0)

    reference_vec = np.array([0, 0, -1])
    if center_axis_normalized_R0[0] == 0 and center_axis_normalized_R0[1] == 0:
        # case in which the reference vector arbitrarily chosen is colinear to center_axis_normalized_R0
        reference_vec = np.array([0, -1, 0])

    normalVector = np.cross(reference_vec, center_axis_normalized_R0)
    normalVectorNormalised = normalVector/np.linalg.norm(normalVector)  # n = (u^v)/|u^v|

    if np.linalg.norm(normalVector) == 0:
        pdb.set_trace()

    cosTheta = np.dot(center_axis_normalized_R0, np.array([0, 0, -1])) #u.v = |u||v|cosTheta
    sinTheta = np.dot(normalVectorNormalised, normalVector)  #(u^v).n = |u||v|sinTheta
    theta = np.arccos(cosTheta)*np.sign(sinTheta)

    q_xyzw = np.array((np.sin(theta/2)*normalVectorNormalised).tolist() + [np.cos(theta/2)]) # quaternion representing theta rotation around normalVectorNormalised

    return q_xyzw


def set_prehensor_from_quaternion_pose_orientation(
    gripper_tip_pose,
    gripper_wrist_pose,
    omega,
    robot,
):

    wrist_RO = gripper_wrist_pose-gripper_tip_pose

    # quaternion to align the direction of the gripper to the point in the object
    q_alignment_xyzw = get_quaternion_oriented_to_the_point(wrist_RO=wrist_RO)
    q_alignment_wxyz = Quaternion(
        a=q_alignment_xyzw[3],
        b=q_alignment_xyzw[0],
        c=q_alignment_xyzw[1],
        d=q_alignment_xyzw[2]
    )

    if robot == env_consts.SimulatedRobot.ALLEGRO_HAND:
        q_rot_wxyz = Quaternion(axis=[0, 1, 0], angle=np.pi / 2)
        q_alignment_wxyz = q_alignment_wxyz * q_rot_wxyz
        q_rot2_wxyz = Quaternion(axis=[0, 1, 0], angle=np.pi)
        q_alignment_wxyz = q_alignment_wxyz * q_rot2_wxyz
        rot_axis = [1, 0, 0]

    elif robot == env_consts.SimulatedRobot.PANDA_2_FINGERS:
        q_rot_wxyz = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)
        q_alignment_wxyz = q_alignment_wxyz * q_rot_wxyz
        rot_axis = [0, 0, 1]

    elif robot == env_consts.SimulatedRobot.BARRETT_HAND_280:
        q_rot_wxyz = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)
        q_alignment_wxyz = q_alignment_wxyz * q_rot_wxyz
        rot_axis = [0, 0, 1]

    elif robot == env_consts.SimulatedRobot.SHADOW_HAND:
        q_rot_wxyz = Quaternion(axis=[0, 1, 0], angle=-np.pi / 2)
        q_alignment_wxyz = q_alignment_wxyz * q_rot_wxyz

        q_rot_wxyz = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
        q_alignment_wxyz = q_alignment_wxyz * q_rot_wxyz

        rot_axis = [0, 1, 0]

    else:
        raise NotImplementedError(f"Invalid robot ref for approach variant: {robot}")

    q_rotation_xyzw = get_90_degres_rotation(omega, rot_axis)
    q_rotation_wxyz = Quaternion(
        a=q_rotation_xyzw[3],
        b=q_rotation_xyzw[0],
        c=q_rotation_xyzw[1],
        d=q_rotation_xyzw[2],
    )
    combine_quat_wxyz = q_alignment_wxyz * q_rotation_wxyz
    combine_quat_xyzw = np.array([
        combine_quat_wxyz[1],
        combine_quat_wxyz[2],
        combine_quat_wxyz[3],
        combine_quat_wxyz[0],
    ])

    gripper_wrist_orient_quat = combine_quat_xyzw
    return gripper_wrist_orient_quat


def get_gripper_pose_relatively_to_contact_point_allegro_compatible_debug(
        bullet_client,
        wrist_palm_offset_gripper,
        hand_pose_from_contact_params,
        normal_at_contact_point,
        contact_point,
        debug,
        robot
):
    all_gripper_6dof_poses = []
    half_aperture = hand_pose_from_contact_params['nu']
    ksi_list = np.linspace(0, 2 * np.pi, 20)
    for ksi in ksi_list:
        half_aperture = half_aperture
        standoff =hand_pose_from_contact_params['d']
        wrist_rotation = hand_pose_from_contact_params['omega']

        # The standoff search space is limited to the palm-hand distance, but the hand pose is defined at wrist pose
        standoff_from_palm = standoff + wrist_palm_offset_gripper

        # Create a base B1 = (normal_at_contact_point_normalized, x_vector_normalized_B1, y_vector_normalized_B1),
        # where B1 is orthonormal, normal_at_contact_point_normalized is the normal to the surface at contact point, and
        # x_vector_normalized_B1 and y_vector_normalized_B1 are tangent vector to the surface at contact point.
        normal_at_contact_point_normalized = normalize_vector(normal_at_contact_point)

        x_vector_B1 = generate_orthogonal_vector(normal_at_contact_point_normalized)
        x_vector_normalized_B1 = normalize_vector(x_vector_B1)

        # creation of y_vector_normalized_B1
        y_vector_B1 = np.cross(x_vector_normalized_B1, normal_at_contact_point_normalized)
        y_vector_normalized_B1 = normalize_vector(y_vector_B1)

        # Apply standoff translation (search variable: d)
        # x -> x'
        x_vector_B1_prime = x_vector_normalized_B1 * standoff_from_palm

        nu_wrap_rad = np.pi / 2 - half_aperture  # nu_wrap_rad deduced form the con revolution angle
        x_prime_prime_B1 = generate_conic_surface(
            bullet_client=bullet_client,
            contact_point=contact_point,
            y_vector_normalized_B1=y_vector_normalized_B1,
            theta_rad=nu_wrap_rad,
            x_vector_B1_prime=x_vector_B1_prime,
            debug=False
        )

        x_prime_prime_prime_B1 = rotate_vector_from_axis_and_angle(
            rotation_axis=normal_at_contact_point_normalized,
            theta=ksi,
            vector_u=x_prime_prime_B1
        )
        position_prehenseur_inside_cone_world = x_prime_prime_prime_B1 + np.array(contact_point)
        point_xyz = position_prehenseur_inside_cone_world

        # step4 : revolution around the norma
        gripper_wrist_pose = point_xyz

        gripper_wrist_orient_quat = set_prehensor_from_quaternion_pose_orientation(
            gripper_tip_pose=contact_point,
            gripper_wrist_pose=point_xyz,
            omega=wrist_rotation,
            robot=robot  # rotation autour de l'entraxe
        )

        gripper_6dof_pose = {
            'xyz': gripper_wrist_pose.tolist(),
            'euler_rpy': None,
            'quaternions': gripper_wrist_orient_quat.tolist()
        }

        all_gripper_6dof_poses.append(gripper_6dof_pose)

    return all_gripper_6dof_poses


def get_gripper_pose_relatively_to_contact_point_allegro_compatible(
        bullet_client,
        wrist_palm_offset_gripper,
        hand_pose_from_contact_params,
        normal_at_contact_point,
        contact_point,
        debug,
        robot
):
    half_aperture = hand_pose_from_contact_params['nu']
    standoff = hand_pose_from_contact_params['d']
    ksi = hand_pose_from_contact_params['ksi']
    wrist_rotation = hand_pose_from_contact_params['omega']

    # The standoff search space is limited to the palm-hand distance, but the hand pose is defined at wrist pose
    standoff_from_palm = standoff + wrist_palm_offset_gripper

    # Create a base B1 = (normal_at_contact_point_normalized, x_vector_normalized_B1, y_vector_normalized_B1),
    # where B1 is orthonormal, normal_at_contact_point_normalized is the normal to the surface at contact point, and
    # x_vector_normalized_B1 and y_vector_normalized_B1 are tangent vector to the surface at contact point.
    normal_at_contact_point_normalized = normalize_vector(normal_at_contact_point)

    x_vector_B1 = generate_orthogonal_vector(normal_at_contact_point_normalized)
    x_vector_normalized_B1 = normalize_vector(x_vector_B1)

    # creation of y_vector_normalized_B1
    y_vector_B1 = np.cross(x_vector_normalized_B1, normal_at_contact_point_normalized)
    y_vector_normalized_B1 = normalize_vector(y_vector_B1)

    # Apply standoff translation (search variable: d)
    # x -> x'
    x_vector_B1_prime = x_vector_normalized_B1 * standoff_from_palm

    nu_wrap_rad = np.pi / 2 - half_aperture  # nu_wrap_rad deduced form the con revolution angle
    x_prime_prime_B1 = generate_conic_surface(
        bullet_client=bullet_client,
        contact_point=contact_point,
        y_vector_normalized_B1=y_vector_normalized_B1,
        theta_rad=nu_wrap_rad,
        x_vector_B1_prime=x_vector_B1_prime,
        debug=False
    )

    x_prime_prime_prime_B1 = rotate_vector_from_axis_and_angle(
        rotation_axis=normal_at_contact_point_normalized,
        theta=ksi,
        vector_u=x_prime_prime_B1
    )
    position_prehenseur_inside_cone_world = x_prime_prime_prime_B1 + np.array(contact_point)

    point_xyz = position_prehenseur_inside_cone_world

    # step4 : revolution around the norma
    gripper_wrist_pose = point_xyz

    gripper_wrist_orient_quat = set_prehensor_from_quaternion_pose_orientation(
        gripper_tip_pose=contact_point,
        gripper_wrist_pose=point_xyz,
        omega=wrist_rotation,
        robot=robot  # rotation autour de l'entraxe
    )

    gripper_6dof_pose = {
        'xyz': gripper_wrist_pose.tolist(),
        'euler_rpy': None,
        'quaternions': gripper_wrist_orient_quat.tolist()
    }

    return gripper_6dof_pose


def get_live_point(alpha, begin_xyz, end_xyz):
    """ Returns the 3d coordinate of the point live_point that verifies:
        live_point = alpha * vector_director + begin_xyz
    """
    if isinstance(begin_xyz, np.ndarray) and isinstance(end_xyz, np.ndarray):
        vector_director = end_xyz - begin_xyz
    else:
        vector_director = np.array(end_xyz) - np.array(begin_xyz)
    point_in_line = alpha * vector_director + begin_xyz
    return point_in_line, vector_director


def search_first_contact_point_clean(
        bullet_client, object_id, start_pos_robot_xyz, euler_robot,
):
    """Note: Make it works first with the this cylinder trick, then optimize it (avoid object creation)."""
    link_world_position_robot = start_pos_robot_xyz
    link_world_orientation = bullet_client.getQuaternionFromEuler(euler_robot)

    # Set approach direction cylinder
    cylinder_with_end = bullet_client.loadURDF(
        PATH_TO_CYLINDER_WITH_LINKS_URDF, link_world_position_robot, link_world_orientation
    )
    link_world_position_sphere_end = list(bullet_client.getLinkState(bodyUniqueId=cylinder_with_end, linkIndex=0)[0])

    # Sphere walk along the direction
    live_point, vector_director = None, None
    precision = np.linspace(0, 1, 300)
    is_contact_point_found = False
    for alpha in precision:
        live_point, vector_director = get_live_point(
            alpha=alpha, begin_xyz=link_world_position_robot, end_xyz=link_world_position_sphere_end
        )
        sphere = bullet_client.loadURDF(PATH_SMALL_GREY_SPHERE_URDF, live_point, link_world_orientation)

        bullet_client.stepSimulation()
        contact = bullet_client.getContactPoints(bodyA=sphere, bodyB=object_id)

        bullet_client.removeBody(sphere)

        if len(contact) != 0:
            is_contact_point_found = True
            break

    bullet_client.removeBody(cylinder_with_end)

    return is_contact_point_found, live_point, vector_director


def display_mesh_point_from_array_debug(bullet_client, object_array_points, debug_obj=PATH_TO_SMALL_RED_SPHERE_URDF):
    for point_in_object_array_points in object_array_points:
        xyz = point_in_object_array_points
        sphere = bullet_client.loadURDF(debug_obj, xyz, [0, 0, 0, 1])


def convert_mesh_pose_to_inertial_frame(obj_inertial_pose, meshpose2cvt):
    mesh2cvt_is_a_valid_vector = len(obj_inertial_pose) == meshpose2cvt.shape[0] \
        if len(meshpose2cvt.shape) == 1 else False
    mesh2cvt_is_a_valid_matrix = len(obj_inertial_pose) == meshpose2cvt.shape[1] \
        if len(meshpose2cvt.shape) == 2 else False
    assert mesh2cvt_is_a_valid_vector or mesh2cvt_is_a_valid_matrix
    meshpose2cvt -= obj_inertial_pose
    return meshpose2cvt


def search_opposite_contact(bullet_client, normal_at_contact_point, contact_point, object_id):
    normal_at_contact_point_in_contact_point_frame = normal_at_contact_point + contact_point

    radius_sphere = 0.005
    nstep = 100
    precision = np.linspace(1 + radius_sphere, 1.5, nstep)
    is_opposite_point_found = False

    object_init_pos, object_init_orient_quat = bullet_client.getBasePositionAndOrientation(object_id)

    live_point, vector_director = None, None
    for alpha in precision:
        live_point, vector_director = get_live_point(
            alpha=alpha, begin_xyz=normal_at_contact_point_in_contact_point_frame, end_xyz=contact_point
        )
        sphere = bullet_client.loadURDF(PATH_SMALL_GREY_SPHERE_URDF, live_point, [0, 0, 0, 1])
        bullet_client.stepSimulation()
        contact = bullet_client.getContactPoints(bodyA=sphere, bodyB=object_id)

        # object base is not fixed: position must be reset as the sphere makes the object move
        bullet_client.resetBasePositionAndOrientation(object_id, object_init_pos, object_init_orient_quat)

        is_sphere_out_of_the_mesh = len(contact) == 0
        if is_sphere_out_of_the_mesh:
            is_opposite_point_found = True
            bullet_client.removeBody(sphere)
            return is_opposite_point_found, live_point, vector_director.tolist()
        else:
            bullet_client.removeBody(sphere)

    return is_opposite_point_found, live_point, vector_director.tolist()


def angle_between_2_vectors(bullet_client, first_vector, second_vector, debug=True):
    norm_first_vec = la.norm(first_vector)
    norm_second_vec = la.norm(second_vector)
    gamma_rad = np.arccos((np.dot(first_vector, second_vector)) / (norm_first_vec * norm_second_vec))
    gamma_degrees = gamma_rad * 180 / 3.14

    if debug:
        # Display the two compared vectors in the world frame
        bullet_client.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=first_vector,
            lineColorRGB=[0, 0, 0]
        )
        bullet_client.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=second_vector,
            lineColorRGB=[0, 255, 255]
        )

    return gamma_degrees


# [-15, 15, 160, 190] best setup in "a billion ways to grasp" => Pi/6
def is_inside_antipodal_selectability_cone(gamma, requirement=[-15, 15, 160, 190]):
    return (requirement[0] < gamma < requirement[1]) or (requirement[2] < gamma < requirement[3])


def calculate_distance_3d(pointA, pointB):
    x_wn, y_wn, z_wn = pointA.tolist()[0], pointA.tolist()[1], pointA.tolist()[2]
    x_contact, y_contact, z_contact = pointB.tolist()[0], pointB.tolist()[1], pointB.tolist()[2]
    distance = math.sqrt((x_wn - x_contact) ** 2 + (y_wn - y_contact) ** 2 + (z_wn - z_contact) ** 2)
    return distance


def entreaxe_selectability(point_first, point_second, distance_max=ENTRAXE_DIST_PANDA_2F):
    is_entraxe_dist_valid = calculate_distance_3d(point_first, point_second) <= distance_max
    return is_entraxe_dist_valid


def entraxe_world_generation(point_in, point_out):
    Entraxe = (point_out - point_in)
    midle_entrax_word = Entraxe/2 + point_in
    return Entraxe, midle_entrax_word


def compute_new_gripper_pose(
        bullet_client, tau, center_axis_R0, gripper_distance_to_axis, center_axis_mid_point_world
):
    q_aroundxaxis_xyzw = get_rotation_around_xaxis(tau=tau)  # plot gripper axis around x
    q_aroundxaxis_wxyz = Quaternion(
        a=q_aroundxaxis_xyzw[3],
        b=q_aroundxaxis_xyzw[0],
        c=q_aroundxaxis_xyzw[1],
        d=q_aroundxaxis_xyzw[2]
    )

    x_axis = np.array([1, 0, 0])
    if are_vectors_colinear(x_axis, center_axis_R0):
        q_entraxe_world_wxyz = Quaternion(a=1, b=0, c=0, d=0)
    else:
        q_entraxe_world_xyzw = get_transform_entraxe_to_xaxis_world(
            bullet_client=bullet_client, center_axis_R0=center_axis_R0
        )
        q_entraxe_world_wxyz = Quaternion(
            a=q_entraxe_world_xyzw[3],
            b=q_entraxe_world_xyzw[0],
            c=q_entraxe_world_xyzw[1],
            d=q_entraxe_world_xyzw[2])

    combine_quat_wxyz = q_entraxe_world_wxyz * q_aroundxaxis_wxyz

    q_rot_wxyz = Quaternion(axis=[0, 0, 1], angle=np.pi/2)

    # /!\ quaternions convention are different for bullet and pyquaternion
    combine_quat2_wxyz = combine_quat_wxyz * q_rot_wxyz
    combine_quat2_xyzw = [combine_quat2_wxyz[1],
                          combine_quat2_wxyz[2],
                          combine_quat2_wxyz[3],
                          combine_quat2_wxyz[0]]

    z_axis_wxyz = np.array([0, 0, 0, 1])
    rotated_z_wxyz = rotate_with_quaternion(combine_quat2_wxyz, z_axis_wxyz)

    new_position = np.array(center_axis_mid_point_world) - gripper_distance_to_axis * rotated_z_wxyz[1: 4]
    new_position = new_position.tolist()

    return new_position, combine_quat2_xyzw


def get_rotation_around_xaxis(tau):
    q = np.array((np.sin(tau / 2) * np.array([1, 0, 0])).tolist() + [
        np.cos(tau / 2)])  # quaternion representing tau rotation around x axis
    return q


def get_transform_entraxe_to_xaxis_world(bullet_client, center_axis_R0):

    center_axis_normalized_R0 = center_axis_R0/np.linalg.norm(center_axis_R0)

    rotation_axis = np.array([1, 0, 0])
    if (rotation_axis == center_axis_normalized_R0).all():
        # case when self.x and center_axis_normalized_R0 are colinear => rotate around y
        rotation_axis = np.array([0, 1, 0])

    normalVector = np.cross(rotation_axis, center_axis_normalized_R0)
    normalVectorNormalised = normalVector/np.linalg.norm(normalVector)  # n = (u^v)/|u^v|

    cosTheta = np.dot(center_axis_normalized_R0, np.array([1, 0, 0])) #u.v = |u||v|cosTheta
    sinTheta = np.dot(normalVectorNormalised, normalVector) #(u^v).n = |u||v|sinTheta
    theta = np.arccos(cosTheta)*np.sign(sinTheta)

    q = np.array((np.sin(theta/2)*normalVectorNormalised).tolist() + [np.cos(theta/2)]) # quaternion representing theta rotation around normalVectorNormalised

    return q


def are_vectors_colinear(U, V):
    n = U.size
    normU = normV = scalUV = 0.0
    for i in range(n):
        normU = normU + U[i] ** 2
        normV = normV + V[i] ** 2
        scalUV = scalUV + U[i] * V[i]
    normU = np.sqrt(normU)
    normV = np.sqrt(normV)
    test = np.abs(scalUV) - normU * normV
    return np.abs(test) < EPSILON_NUMERICAL_STABILITY


def rotate_with_quaternion(quaternion_wxyz, point_wxyz):
    quaternion_inv_wxyz = Quaternion(
        a=quaternion_wxyz[0],
        b=-quaternion_wxyz[1],
        c=-quaternion_wxyz[2],
        d=-quaternion_wxyz[3])
    hamilton = quaternion_wxyz * point_wxyz
    rotated_point_wxyz = hamilton*quaternion_inv_wxyz
    rotated_point_wxyz = np.array(
        [rotated_point_wxyz[0], rotated_point_wxyz[1], rotated_point_wxyz[2], rotated_point_wxyz[3]]
    )
    return rotated_point_wxyz

