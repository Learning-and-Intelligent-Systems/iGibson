import collections
import os
import random
import numpy as np

import pybullet as p
import scipy
import yaml
from packaging import version
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transforms3d import quaternions

scipy_version = version.parse(scipy.version.version)

# File I/O related


def parse_config(config):

    """
    Parse iGibson config file / object
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data


def parse_str_config(config):
    """
    Parse string config
    """
    return yaml.safe_load(config)


def dump_config(config):
    """
    Converts YML config into a string
    """
    return yaml.dump(config)


# Geometry related


def rotate_vector_3d(v, r, p, y, cck=True):
    """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
    if scipy_version >= version.parse("1.4"):
        local_to_global = R.from_euler("xyz", [r, p, y]).as_matrix()
    else:
        local_to_global = R.from_euler("xyz", [r, p, y]).as_dcm()
    if cck:
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v)
    else:
        return np.dot(local_to_global, v)


def get_transform_from_xyz_rpy(xyz, rpy):
    """
    Returns a homogeneous transformation matrix (numpy array 4x4)
    for the given translation and rotation in roll,pitch,yaw
    xyz = Array of the translation
    rpy = Array with roll, pitch, yaw rotations
    """
    if scipy_version >= version.parse("1.4"):
        rotation = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
    else:
        rotation = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_dcm()
    transformation = np.eye(4)
    transformation[0:3, 0:3] = rotation
    transformation[0:3, 3] = xyz
    return transformation


def get_rpy_from_transform(transform):
    """
    Returns the roll, pitch, yaw angles (Euler) for a given rotation or
    homogeneous transformation matrix
    transformation = Array with the rotation (3x3) or full transformation (4x4)
    """
    rpy = R.from_dcm(transform[0:3, 0:3]).as_euler("xyz")
    return rpy


def rotate_vector_2d(v, yaw):
    """Rotates 2d vector by yaw counterclockwise"""
    if scipy_version >= version.parse("1.4"):
        local_to_global = R.from_euler("z", yaw).as_matrix()
    else:
        local_to_global = R.from_euler("z", yaw).as_dcm()
    global_to_local = local_to_global.T
    global_to_local = global_to_local[:2, :2]
    if len(v.shape) == 1:
        return np.dot(global_to_local, v)
    elif len(v.shape) == 2:
        return np.dot(global_to_local, v.T).T
    else:
        print("Incorrect input shape for rotate_vector_2d", v.shape)
        return v


def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def quatFromXYZW(xyzw, seq):
    """Convert quaternion from XYZW (pybullet convention) to arbitrary sequence."""
    assert (
        len(seq) == 4 and "x" in seq and "y" in seq and "z" in seq and "w" in seq
    ), "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = ["xyzw".index(axis) for axis in seq]
    return xyzw[inds]


def quatToXYZW(orn, seq):
    """Convert quaternion from arbitrary sequence to XYZW (pybullet convention)."""
    assert (
        len(seq) == 4 and "x" in seq and "y" in seq and "z" in seq and "w" in seq
    ), "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = [seq.index(axis) for axis in "xyzw"]
    return orn[inds]


def quatXYZWFromRotMat(rot_mat):
    """Convert quaternion from rotation matrix"""
    quatWXYZ = quaternions.mat2quat(rot_mat)
    quatXYZW = quatToXYZW(quatWXYZ, "wxyz")
    return quatXYZW


# Represents a rotation by q1, followed by q0
def multQuatLists(q0, q1):
    """Multiply two quaternions that are represented as lists."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    return [
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ]


def normalizeListVec(v):
    """Normalizes a vector list."""
    length = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    if length <= 0:
        length = 1
    v = [val / np.sqrt(length) for val in v]
    return v


# Quat(wxyz)
def quat_pos_to_mat(pos, quat):
    """Convert position and quaternion to transformation matrix"""
    r_w, r_x, r_y, r_z = quat
    # print("quat", r_w, r_x, r_y, r_z)
    mat = np.eye(4)
    mat[:3, :3] = quaternions.quat2mat([r_w, r_x, r_y, r_z])
    mat[:3, -1] = pos
    # Return: roll, pitch, yaw
    return mat


def transform_texture(input_filename, output_filename, mixture_weight=0, mixture_color=(0, 0, 0)):
    img = np.array(Image.open(input_filename))
    img = img * (1 - mixture_weight) + np.array(list(mixture_color))[None, None, :] * mixture_weight
    img = img.astype(np.uint8)
    Image.fromarray(img).save(output_filename)


def brighten_texture(input_filename, output_filename, brightness=1):
    img = np.array(Image.open(input_filename))
    img = np.clip(img * brightness, 0, 255)
    img = img.astype(np.uint8)
    Image.fromarray(img).save(output_filename)


def restoreState(*args, **kwargs):
    """Restore to a given pybullet state, with a mitigation for a known sleep state restore bug.
    When the pybullet state is restored, the object's wake zone (the volume around the object where
    if any other object enters, the object should be waken up) does not get reset correctly,
    causing weird bugs around asleep objects. This function mitigates the issue by forcing the
    sleep code to update each object's wake zone.
    """
    p.restoreState(*args, **kwargs)
    for body_id in range(p.getNumBodies()):
        p.resetBasePositionAndOrientation(
            body_id, *p.getBasePositionAndOrientation(body_id), physicsClientId=kwargs.get("physicsClientId", 0)
        )
    return p.restoreState(*args, **kwargs)


def let_user_pick(options, print_intro=True, random_selection=False):
    if print_intro and not random_selection:
        print("Please choose:")
    for idx, element in enumerate(options):
        print("{}) {}".format(idx + 1, element))
    if not random_selection:
        i = input("Enter number: ")
    else:
        i = random.choice(range(len(options))) + 1
    try:
        if 0 < int(i) <= len(options):
            return int(i)
    except:
        pass
    return None

# To dynamically modify wbm3_modifiable_full_obs.yaml file
def modify_config_file(config_file, task_name, scene_id, randomize_init_state, seed):
    # Read in the file
    with open(config_file, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('task_placeholder', task_name)
    filedata = filedata.replace('scene_placeholder', scene_id)
    filedata = filedata.replace('online_sampling_placeholder', str(randomize_init_state))
    #print(filedata)
    
    config_file = config_file + "_tmp_" + task_name + str(scene_id) + str(seed)

    # Write the file out again
    with open(config_file, 'w') as file:
        file.write(filedata)
    
    return config_file