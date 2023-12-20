import os
import time

import numpy as np
import pybullet as p

import igibson
from igibson.external.pybullet_tools.utils import (
    CIRCULAR_LIMITS,
    MAX_DISTANCE,
    MAX_DISTANCE_GRASP,
    PI,
    birrt,
    circular_difference,
    direct_path,
    get_base_difference_fn,
    get_base_distance_fn,
    get_joint_positions,
    pairwise_collision,
)
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.fetch_gripper_robot import FetchGripper
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config


def plan_base_motion_fg(
    robot: FetchGripper,
    end_conf,
    base_limits,
    obstacles=[],
    direct=False,
    weights=1 * np.ones(3),
    resolutions=0.05 * np.ones(3),
    max_distance=MAX_DISTANCE,
    override_sample_fn=None,
    rng=np.random.default_rng(23),
    **kwargs
):
    def sample_fn():
        x, y = (rng.random() * (base_limits[1] - base_limits[0])) + base_limits[0]
        theta = (rng.random() * (CIRCULAR_LIMITS[1] - CIRCULAR_LIMITS[0])) + CIRCULAR_LIMITS[0]
        return (x, y, theta)

    if override_sample_fn is not None:
        sample_fn = override_sample_fn

    difference_fn = get_base_difference_fn()
    distance_fn = get_base_distance_fn(weights=weights)

    # body_ids = []
    # for part in ["base_link", "torso_lift_link", "gripper_link"]:   # these are not all, but hopefully it'll be enough and will be a bit faster
    #     body_ids.append(robot.parts[part].body_id)
    body_ids = [robot.body_id]

    if robot.object_in_hand is not None:
        body_ids.append(robot.object_in_hand)

    def extend_fn(q1, q2):
        target_theta = np.arctan2(q2[1] - q1[1], q2[0] - q1[0])

        n1 = int(np.abs(circular_difference(target_theta, q1[2]) / resolutions[2])) + 1
        n3 = int(np.abs(circular_difference(q2[2], target_theta) / resolutions[2])) + 1
        steps2 = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n2 = int(np.max(steps2)) + 1

        for i in range(n1):
            q = (i / (n1)) * np.array(difference_fn((q1[0], q1[1], target_theta), q1)) + np.array(q1)
            q = tuple(q)
            yield q

        for i in range(n2):
            q = (i / (n2)) * np.array(
                difference_fn((q2[0], q2[1], target_theta), (q1[0], q1[1], target_theta))
            ) + np.array((q1[0], q1[1], target_theta))
            q = tuple(q)
            yield q

        for i in range(n3):
            q = (i / (n3)) * np.array(difference_fn(q2, (q2[0], q2[1], target_theta))) + np.array(
                (q2[0], q2[1], target_theta)
            )
            q = tuple(q)
            yield q

    def collision_fn(q):
        # TODO: update this function
        # set_base_values(body, q)
        robot.set_position_orientation([q[0], q[1], robot.initial_z_offset], p.getQuaternionFromEuler([0, 0, q[2]]))
        return any(
            pairwise_collision(body_id, obs, max_distance=max_distance) for obs in obstacles for body_id in body_ids
        )

    pos = robot.get_position()
    yaw = p.getEulerFromQuaternion(robot.get_orientation())[2]
    start_conf = [pos[0], pos[1], yaw]
    if collision_fn(start_conf):
        print("Warning: initial configuration is in collision")
        return None
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)


def dry_run_base_plan(robot: FetchGripper, plan):
    for (x, y, yaw) in plan:
        robot.set_position_orientation([x, y, robot.initial_z_offset], p.getQuaternionFromEuler([0, 0, yaw]))
        time.sleep(0.05)


def dry_run_arm_plan(robot: FetchGripper, plan):
    for (x, y, z, roll, pitch, yaw) in plan:
        robot.parts["right_hand"].set_position_orientation([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
        time.sleep(0.01)


def plan_gripper_motion_fg(
    robot: FetchGripper,
    obj_in_hand,
    end_conf,
    joint_limits,
    obstacles=[],
    direct=False,
    weights=None,
    resolutions=None,
    max_distance=MAX_DISTANCE_GRASP,
    rng=np.random.default_rng(23),
    **kwargs
):
    
    def get_joint_mask(joints):
        mask = np.zeros(len(robot.joint_ids), dtype=np.bool)
        mask[joints] = True
        return mask

    # pos = robot.parts["right_hand"].get_position()
    # orn = robot.parts["right_hand"].get_orientation()
    # rpy = p.getEulerFromQuaternion(orn)
    # start_conf = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]
    start_conf = get_joint_positions(robot.robot_ids[0], robot.joint_ids)

    manipulation_mask = get_joint_mask(robot.manipulation_joints)
    assert np.allclose(np.array(end_conf)[~manipulation_mask], np.array(start_conf)[~manipulation_mask]), "Cannot plan joints that are not manipulation related"
    revolute_mask = get_joint_mask(robot.revolute_joints)
    continuous_mask = get_joint_mask(robot.continuous_joints)
    prismatic_mask = get_joint_mask(robot.prismatic_joints)

    if weights is None:
        weights = np.ones(len(robot.joint_ids))
        weights[continuous_mask | revolute_mask] = 5    # these are angles

    if resolutions is None:
        resolutions = 0.02 * np.ones(len(robot.joint_ids))

    def sample_fn():
        q = np.array(start_conf)
        upper = np.array(joint_limits[1])
        lower = np.array(joint_limits[0])
        q[manipulation_mask] = rng.random(size=manipulation_mask.sum()) * (upper - lower) + lower
        return tuple(q)

    def get_joint_difference_fn():
        def fn(q2, q1):
            q2 = np.array(q2)
            q1 = np.array(q1)

            diff = np.zeros_like(q2)
            diff[revolute_mask] = q2[revolute_mask] - q1[revolute_mask]
            diff[continuous_mask] = circular_difference(q2[continuous_mask], q1[continuous_mask])
            diff[prismatic_mask] = q2[prismatic_mask] - q2[prismatic_mask]
            return tuple(diff)

        return fn


    def get_joint_distance_fn(weights):
        difference_fn = get_joint_difference_fn()

        def fn(q1, q2):
            difference = np.array(difference_fn(q2, q1))
            return np.sqrt(np.dot(weights, difference * difference))

        return fn

    difference_fn = get_joint_difference_fn()
    distance_fn = get_joint_distance_fn(weights=weights)

    def extend_fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n = int(np.max(steps)) + 1

        for i in range(n):
            q = (i / float(n)) * np.array(difference_fn(q2, q1)) + np.array(q1)
            q = tuple(q)
            yield q

    def collision_fn(q):
        # TODO: update this function
        # set_base_values(body, q)

        # This internally moves held objects
        robot.set_joint_positions(q)

        collision = any(
            pairwise_collision((robot.body_id, (robot.parts["gripper_link"].body_part_index,
                                robot.parts["l_gripper_finger_link"].body_part_index,
                                robot.parts["r_gripper_finger_link"].body_part_index)),
                obs, max_distance=max_distance) for obs in obstacles
        )

        if obj_in_hand is not None:
            if type(obj_in_hand.body_id) == list:
                obj_in_hand_body_id = obj_in_hand.body_id[0]
            else:
                obj_in_hand_body_id = obj_in_hand.body_id
            collision = collision or any(
                pairwise_collision(obj_in_hand_body_id, obs, max_distance=max_distance) for obs in obstacles if obs != 0
            )

        return collision

    if collision_fn(start_conf):
        print("Warning: initial configuration is in collision")
        return None
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)


