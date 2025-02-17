import argparse
import logging
import time
from enum import IntEnum

import gym.spaces
import numpy as np
import pybullet as p
import scipy

from igibson import object_states
from igibson.envs.behavior_env import BehaviorEnv
from igibson.external.pybullet_tools.utils import CIRCULAR_LIMITS, get_base_difference_fn
from igibson.object_states.on_floor import RoomFloor
from igibson.object_states.utils import sample_kinematics, continuous_param_kinematics
from igibson.objects.articulated_object import URDFObject
from igibson.robots.behavior_robot import BRBody, BREye, BRHand
from igibson.utils.behavior_robot_planning_utils import dry_run_base_plan, plan_base_motion_br, plan_hand_motion_br
# from igibson.external.pybullet_tools.utils import (
#     CIRCULAR_LIMITS,
#     MAX_DISTANCE,
#     PI,
#     birrt,
#     circular_difference,
#     direct_path,
#     get_base_difference_fn,
#     get_base_distance_fn,
#     pairwise_collision,
# )

NUM_ACTIONS = 6
MAX_ACTION_CONT_PARAMS = 7


class ActionPrimitives(IntEnum):
    NAVIGATE_TO = 0
    GRASP = 1
    PLACE_ONTOP = 2
    PLACE_INSIDE = 3
    OPEN = 4
    CLOSE = 5


def get_aabb_volume(lo, hi):
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]


def detect_collision(bodyA, object_in_hand=None):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA or body_id == object_in_hand:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_robot_collision(robot):
    object_in_hand = robot.parts["right_hand"].object_in_hand
    return (
        detect_collision(robot.parts["body"].body_id)
        or detect_collision(robot.parts["left_hand"].body_id)
        or detect_collision(robot.parts["right_hand"].body_id, object_in_hand)
    )


class BehaviorTAMPEnv(BehaviorEnv):
    """
    iGibson Environment (OpenAI Gym interface).
    Similar to BehaviorMPEnv, except (1) all high-level
    actions are parameterized to make them deterministic
    and thus conducive to sample-based TAMP and (2) there are 
    more such high-level actions.
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        seed=0,
        action_filter="mobile_manipulation",
        use_motion_planning=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(BehaviorTAMPEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            action_filter=action_filter,
            seed=seed,
            automatic_reset=automatic_reset,
        )

        self.obj_in_hand = None
        self.use_motion_planning = use_motion_planning
        self.robots[0].initial_z_offset = 0.7

    def load_action_space(self):
        self.task_relevant_objects = [
            item
            for item in self.task.object_scope.values()
            if isinstance(item, URDFObject) or isinstance(item, RoomFloor)
        ]
        self.num_objects = len(self.task_relevant_objects)
        self.action_space = gym.spaces.Discrete(self.num_objects * NUM_ACTIONS)

    def get_body_ids(self, include_self=False):
        ids = []
        for object in self.scene.get_objects():
            if isinstance(object, URDFObject):
                ids.extend(object.body_ids)

        if include_self:
            ids.append(self.robots[0].parts["left_hand"].get_body_id())
            ids.append(self.robots[0].parts["body"].get_body_id())

        return ids

    def reset_and_release_hand(self):
        self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())
        for _ in range(100):
            self.robots[0].parts["right_hand"].set_close_fraction(0)
            self.robots[0].parts["right_hand"].trigger_fraction = 0
            p.stepSimulation()

    # def step(self, action, continuous_params):
    #     """
    #     :param action: an integer such that action % self.num_objects yields the object id, and 
    #     action // self.num_objects yields the correct action enum
    #     :param continuous_params: a numpy vector of length MAX_ACTION_CONT_PARAMS. This represents
    #     values derived from the continuous parameters of the action.
    #     """

    #     obj_list_id = int(action) % self.num_objects
    #     action_primitive = int(action) // self.num_objects
    #     obj = self.task_relevant_objects[obj_list_id]

    #     assert continuous_params.shape == (MAX_ACTION_CONT_PARAMS,)

    #     if not (isinstance(obj, BRBody) or isinstance(obj, BRHand) or isinstance(obj, BREye)):
    #         if action_primitive == ActionPrimitives.NAVIGATE_TO:
    #             if self.navigate_to_obj_pos(obj, continuous_params[0:2], use_motion_planning=self.use_motion_planning):
    #                 logging.debug("PRIMITIVE: navigate to {} success".format(obj.name))
    #             else:
    #                 logging.debug("PRIMITIVE: navigate to {} fail".format(obj.name))

    #         elif action_primitive == ActionPrimitives.GRASP:
    #             if self.obj_in_hand is None:
    #                 if isinstance(obj, URDFObject) and hasattr(obj, "states") and object_states.AABB in obj.states:
    #                     lo, hi = obj.states[object_states.AABB].get_value()
    #                     volume = get_aabb_volume(lo, hi)
    #                     if (
    #                         volume < 0.2 * 0.2 * 0.2 and not obj.main_body_is_fixed
    #                     ):  # say we can only grasp small objects
    #                         if (
    #                             np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position()))
    #                             < 2
    #                         ):
    #                             self.grasp_obj(obj, use_motion_planning=self.use_motion_planning)
    #                             logging.debug(
    #                                 "PRIMITIVE: grasp {} success, obj in hand {}".format(obj.name, self.obj_in_hand)
    #                             )
    #                         else:
    #                             logging.debug("PRIMITIVE: grasp {} fail, too far".format(obj.name))
    #                     else:
    #                         logging.debug("PRIMITIVE: grasp {} fail, too big or fixed".format(obj.name))

            # elif action_primitive == ActionPrimitives.PLACE_ONTOP:
            #     if self.obj_in_hand is not None and self.obj_in_hand != obj:
            #         logging.debug("PRIMITIVE:attempt to place {} ontop {}".format(self.obj_in_hand.name, obj.name))

            #         if isinstance(obj, URDFObject):
            #             if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
            #                 state = p.saveState()
            #                 result = sample_kinematics(
            #                     "onTop",
            #                     self.obj_in_hand,
            #                     obj,
            #                     True,
            #                     use_ray_casting_method=True,
            #                     max_trials=20,
            #                     skip_falling=True,
            #                 )
            #                 if result:
            #                     logging.debug(
            #                         "PRIMITIVE: place {} ontop {} success".format(self.obj_in_hand.name, obj.name)
            #                     )
            #                     pos = self.obj_in_hand.get_position()
            #                     orn = self.obj_in_hand.get_orientation()
            #                     self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
            #                 else:
            #                     p.removeState(state)
            #                     logging.debug(
            #                         "PRIMITIVE: place {} ontop {} fail, sampling fail".format(
            #                             self.obj_in_hand.name, obj.name
            #                         )
            #                     )

            #             else:
            #                 logging.debug(
            #                     "PRIMITIVE: place {} ontop {} fail, too far".format(self.obj_in_hand.name, obj.name)
            #                 )
            #         else:
            #             state = p.saveState()
            #             result = sample_kinematics(
            #                 "onFloor",
            #                 self.obj_in_hand,
            #                 obj,
            #                 True,
            #                 use_ray_casting_method=True,
            #                 max_trials=20,
            #                 skip_falling=True,
            #             )
            #             if result:
            #                 logging.debug(
            #                     "PRIMITIVE: place {} ontop {} success".format(self.obj_in_hand.name, obj.name)
            #                 )
            #                 pos = self.obj_in_hand.get_position()
            #                 orn = self.obj_in_hand.get_orientation()
            #                 self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
            #             else:
            #                 logging.debug(
            #                     "PRIMITIVE: place {} ontop {} fail, sampling fail".format(
            #                         self.obj_in_hand.name, obj.name
            #                     )
            #                 )
            #                 p.removeState(state)

            # elif action_primitive == ActionPrimitives.PLACE_INSIDE:
            #     if self.obj_in_hand is not None and self.obj_in_hand != obj and isinstance(obj, URDFObject):
            #         logging.debug("PRIMITIVE:attempt to place {} inside {}".format(self.obj_in_hand.name, obj.name))
            #         if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
            #             if (
            #                 hasattr(obj, "states")
            #                 and object_states.Open in obj.states
            #                 and obj.states[object_states.Open].get_value()
            #             ) or (hasattr(obj, "states") and not object_states.Open in obj.states):
            #                 state = p.saveState()
            #                 result = sample_kinematics(
            #                     "inside",
            #                     self.obj_in_hand,
            #                     obj,
            #                     True,
            #                     use_ray_casting_method=True,
            #                     max_trials=20,
            #                     skip_falling=True,
            #                 )
            #                 if result:
            #                     logging.debug(
            #                         "PRIMITIVE: place {} inside {} success".format(self.obj_in_hand.name, obj.name)
            #                     )
            #                     pos = self.obj_in_hand.get_position()
            #                     orn = self.obj_in_hand.get_orientation()
            #                     self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
            #                 else:
            #                     logging.debug(
            #                         "PRIMITIVE: place {} inside {} fail, sampling fail".format(
            #                             self.obj_in_hand.name, obj.name
            #                         )
            #                     )
            #                     p.removeState(state)
            #             else:
            #                 logging.debug(
            #                     "PRIMITIVE: place {} inside {} fail, need open not open".format(
            #                         self.obj_in_hand.name, obj.name
            #                     )
            #                 )
            #         else:
            #             logging.debug(
            #                 "PRIMITIVE: place {} inside {} fail, too far".format(self.obj_in_hand.name, obj.name)
            #             )
            # elif action_primitive == ActionPrimitives.OPEN:
            #     if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
            #         if hasattr(obj, "states") and object_states.Open in obj.states:
            #             obj.states[object_states.Open].set_value(True)
            #         else:
            #             logging.debug("PRIMITIVE open failed, cannot be opened")
            #     else:
            #         logging.debug("PRIMITIVE open failed, too far")

            # elif action_primitive == ActionPrimitives.CLOSE:
            #     if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
            #         if hasattr(obj, "states") and object_states.Open in obj.states:
            #             obj.states[object_states.Open].set_value(False)
            #         else:
            #             logging.debug("PRIMITIVE close failed, cannot be opened")
            #     else:
            #         logging.debug("PRIMITIVE close failed, too far")

        # state, reward, done, info = super(BehaviorTAMPEnv, self).step(np.zeros(17))
        # logging.debug("PRIMITIVE satisfied predicates:", info["satisfied_predicates"])
        # return state, reward, done, info

    def step_with_exec_info(self, action, continuous_params):
        """
        Same as the above step method, but returns an additional param (action_exec_status)
        that is True if the high-level action execution succeeded and False otherwise
        """
        obj_list_id = int(action) % self.num_objects
        action_primitive = int(action) // self.num_objects
        action_exec_status = False
        obj = self.task_relevant_objects[obj_list_id]

        assert continuous_params.shape == (MAX_ACTION_CONT_PARAMS,)

        if not (isinstance(obj, BRBody) or isinstance(obj, BRHand) or isinstance(obj, BREye)):
            if action_primitive == ActionPrimitives.NAVIGATE_TO:
                if self.navigate_to_obj_pos(obj, continuous_params[0:2], use_motion_planning=self.use_motion_planning):
                    print("PRIMITIVE: navigate to {} success".format(obj.name))
                    action_exec_status = True
                else:
                    print(f"PRIMITIVE: navigate to {obj.name} with params {continuous_params[0:2]} fail")

            elif action_primitive == ActionPrimitives.GRASP:
                if self.obj_in_hand is None:
                    if isinstance(obj, URDFObject) and hasattr(obj, "states") and object_states.AABB in obj.states:
                        lo, hi = obj.states[object_states.AABB].get_value()
                        volume = get_aabb_volume(lo, hi)
                        if (
                            volume < 0.2 * 0.2 * 0.2 and not obj.main_body_is_fixed
                        ):  # say we can only grasp small objects
                            if (
                                np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position()))
                                < 2
                            ):
                                self.grasp_obj_at_pos(obj, continuous_params[0:3], use_motion_planning=self.use_motion_planning)
                                print(
                                    "PRIMITIVE: grasp {} success, obj in hand {}".format(obj.name, self.obj_in_hand)
                                )
                                action_exec_status = True
                            else:
                                print("PRIMITIVE: grasp {} fail, too far".format(obj.name))
                        else:
                            print("PRIMITIVE: grasp {} fail, too big or fixed".format(obj.name))
                else:
                    print("PRIMITIVE: grasp {} fail, agent already has an object in hand!".format(obj.name))

            elif action_primitive == ActionPrimitives.PLACE_ONTOP:
                if self.obj_in_hand is not None and self.obj_in_hand != obj:
                    print("PRIMITIVE:attempt to place {} ontop {}".format(self.obj_in_hand.name, obj.name))

                    if isinstance(obj, URDFObject):
                        if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
                            state = p.saveState()
                            result = continuous_param_kinematics(
                                "onTop",
                                self.obj_in_hand,
                                obj,
                                True,
                                continuous_params[0:7],
                                use_ray_casting_method=True,
                                max_trials=10,
                                skip_falling=True,
                            )
                            if result:
                                print(
                                    "PRIMITIVE: place {} ontop {} success".format(self.obj_in_hand.name, obj.name)
                                )
                                action_exec_status = True
                                pos = self.obj_in_hand.get_position()
                                orn = self.obj_in_hand.get_orientation()
                                self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
                            else:
                                p.removeState(state)
                                print(
                                    "PRIMITIVE: place {} ontop {} fail, sampling fail".format(
                                        self.obj_in_hand.name, obj.name
                                    )
                                )

                        else:
                            print(
                                "PRIMITIVE: place {} ontop {} fail, too far".format(self.obj_in_hand.name, obj.name)
                            )
                    else:
                        state = p.saveState()
                        result = continuous_param_kinematics(
                            "onFloor",
                            self.obj_in_hand,
                            obj,
                            True,
                            continuous_params[0:7],
                            use_ray_casting_method=True,
                            max_trials=10,
                            skip_falling=True,
                        )
                        if result:
                            print(
                                "PRIMITIVE: place {} ontop {} success".format(self.obj_in_hand.name, obj.name)
                            )
                            action_exec_status = True
                            pos = self.obj_in_hand.get_position()
                            orn = self.obj_in_hand.get_orientation()
                            self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
                        else:
                            print(
                                "PRIMITIVE: place {} ontop {} fail, sampling fail".format(
                                    self.obj_in_hand.name, obj.name
                                )
                            )
                            p.removeState(state)

            # elif action_primitive == ActionPrimitives.PLACE_INSIDE:
            #     if self.obj_in_hand is not None and self.obj_in_hand != obj and isinstance(obj, URDFObject):
            #         print("PRIMITIVE:attempt to place {} inside {}".format(self.obj_in_hand.name, obj.name))
            #         if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
            #             if (
            #                 hasattr(obj, "states")
            #                 and object_states.Open in obj.states
            #                 and obj.states[object_states.Open].get_value()
            #             ) or (hasattr(obj, "states") and not object_states.Open in obj.states):
            #                 state = p.saveState()
            #                 result = sample_kinematics(
            #                     "inside",
            #                     self.obj_in_hand,
            #                     obj,
            #                     True,
            #                     use_ray_casting_method=True,
            #                     max_trials=20,
            #                     skip_falling=True,
            #                 )
            #                 if result:
            #                     print(
            #                         "PRIMITIVE: place {} inside {} success".format(self.obj_in_hand.name, obj.name)
            #                     )
            #                     action_exec_status = True
            #                     pos = self.obj_in_hand.get_position()
            #                     orn = self.obj_in_hand.get_orientation()
            #                     self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
            #                 else:
            #                     print(
            #                         "PRIMITIVE: place {} inside {} fail, sampling fail".format(
            #                             self.obj_in_hand.name, obj.name
            #                         )
            #                     )
            #                     p.removeState(state)
            #             else:
            #                 print(
            #                     "PRIMITIVE: place {} inside {} fail, need open not open".format(
            #                         self.obj_in_hand.name, obj.name
            #                     )
            #                 )
            #         else:
            #             print(
            #                 "PRIMITIVE: place {} inside {} fail, too far".format(self.obj_in_hand.name, obj.name)
            #             )
            # elif action_primitive == ActionPrimitives.OPEN:
            #     if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
            #         if hasattr(obj, "states") and object_states.Open in obj.states:
            #             action_exec_status = True
            #             print("PRIMITIVE open succeeded")
            #             obj.states[object_states.Open].set_value(True)
            #         else:
            #             print("PRIMITIVE open failed, cannot be opened")
            #     else:
            #         print("PRIMITIVE open failed, too far")

            # elif action_primitive == ActionPrimitives.CLOSE:
            #     if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
            #         if hasattr(obj, "states") and object_states.Open in obj.states:
            #             action_exec_status = True
            #             obj.states[object_states.Open].set_value(False)
            #             print("PRIMITIVE close succeeded")
            #         else:
            #             print("PRIMITIVE close failed, cannot be opened")
            #     else:
            #         print("PRIMITIVE close failed, too far")

        else:
            print(f"Attempted to execute a High Level Action whose target was the robot's body! This is incorrect, please pass a sensible high level action argument!")

        state, reward, done, info = super(BehaviorTAMPEnv, self).step(np.zeros(17))
        print("PRIMITIVE satisfied predicates:", info["satisfied_predicates"])
        return state, reward, done, info, action_exec_status

    def grasp_obj_at_pos(self, obj, grasp_offset_and_z_rot, use_motion_planning=False):
        if use_motion_planning:
            x, y, z = obj.get_position()
            x += grasp_offset_and_z_rot[0]
            y += grasp_offset_and_z_rot[1]
            z += grasp_offset_and_z_rot[2]
            hand_x, hand_y, hand_z = self.robots[0].parts["right_hand"].get_position()

            # # add a little randomness to avoid getting stuck
            # x += np.random.uniform(-0.025, 0.025)
            # y += np.random.uniform(-0.025, 0.025)
            # z += np.random.uniform(-0.025, 0.025)

            minx = min(x, hand_x) - 0.5
            miny = min(y, hand_y) - 0.5
            minz = min(z, hand_z) - 0.5
            maxx = max(x, hand_x) + 0.5
            maxy = max(y, hand_y) + 0.5
            maxz = max(z, hand_z) + 0.5

            # compute the angle the hand must be in such that it can grasp the object from its current offset position
            # This involves aligning the z-axis (in the world frame) of the hand with the vector that goes from the hand 
            # to the object. We can find the rotation matrix that accomplishes this rotation by following:
            # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
            hand_to_obj_vector = np.array([x - hand_x, y - hand_y, z - hand_z])
            hand_to_obj_unit_vector = hand_to_obj_vector / np.linalg.norm(hand_to_obj_vector)
            unit_z_vector = np.array(0,0,1)
            c = np.dot(unit_z_vector, hand_to_obj_unit_vector)
            if not c == -1.0:
                v = np.cross(unit_z_vector, hand_to_obj_unit_vector)
                s = np.linalg.norm(v)
                v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + v_x + np.linalg.matrix_power(v_x, 2) * ((1-c)/(s ** 2))
                r = scipy.spatial.transform.Rotation.from_matrix(R)
                euler_angles = r.as_euler('xyz')
                euler_angles[2] += grasp_offset_and_z_rot[3]
            else:
                euler_angles = np.array([0.0, np.pi, 0.0])

            state = p.saveState()
            # plan a motion to above the object
            plan = plan_hand_motion_br(
                robot=self.robots[0],
                obj_in_hand=None,
                end_conf=[x, y, z, euler_angles[0], euler_angles[1], euler_angles[2]],
                hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
                obstacles=self.get_body_ids(include_self=True),
            )
            p.restoreState(state)
            p.removeState(state)

            if plan is not None:
                grasp_success = self.execute_grasp_plan(plan, obj)
                logging.debug("grasp success", grasp_success)
                if grasp_success:
                    self.obj_in_hand = obj
                else:
                    logging.debug("grasp failed")
                    self.reset_and_release_hand()
            else:
                logging.debug("plan is None")
                self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())
                self.reset_and_release_hand()
        else:
            self.obj_in_hand = obj
            obj.set_position(np.array(self.robots[0].parts["right_hand"].get_position()))
            self.robots[0].parts["right_hand"].set_close_fraction(1)
            self.robots[0].parts["right_hand"].trigger_fraction = 1
            p.stepSimulation()
            obj.set_position(np.array(self.robots[0].parts["right_hand"].get_position()))
            self.robots[0].parts["right_hand"].handle_assisted_grasping(
                np.zeros(
                    28,
                ),
                override_ag_data=(obj.body_id[0], -1),
            )

    def execute_grasp_plan(self, plan, obj):
        for x, y, z, roll, pitch, yaw in plan:
            self.robots[0].parts["right_hand"].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()
            # s, a = get_last_action(p)

        x, y, z, roll, pitch, yaw = plan[-1]

        # lower the hand until it touches the object
        for i in range(25):
            self.robots[0].parts["right_hand"].move([x, y, z - i * 0.005], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        # close the hand
        for _ in range(50):
            self.robots[0].parts["right_hand"].set_close_fraction(1)
            self.robots[0].parts["right_hand"].trigger_fraction = 1
            p.stepSimulation()

        grasp_success = (
            self.robots[0]
            .parts["right_hand"]
            .handle_assisted_grasping(
                np.zeros(
                    28,
                ),
                override_ag_data=(obj.body_id[0], -1),
            )
        )

        # reverse the plan and get object close to torso
        for x, y, z, roll, pitch, yaw in plan[::-1]:
            self.robots[0].parts["right_hand"].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        return grasp_success

    def place_obj(self, original_state, target_pos, target_orn, use_motion_planning=False):
        pos = self.obj_in_hand.get_position()
        p.restoreState(original_state)
        p.removeState(original_state)
        if not use_motion_planning:
            self.reset_and_release_hand()
            self.robots[0].parts["right_hand"].force_release_obj()
            self.obj_in_hand.set_position_orientation(target_pos, target_orn)
            self.obj_in_hand = None

        else:
            x, y, z = target_pos
            hand_x, hand_y, hand_z = self.robots[0].parts["right_hand"].get_position()

            minx = min(x, hand_x) - 1
            miny = min(y, hand_y) - 1
            minz = min(z, hand_z) - 0.5
            maxx = max(x, hand_x) + 1
            maxy = max(y, hand_y) + 1
            maxz = max(z, hand_z) + 0.5

            state = p.saveState()
            obstacles = self.get_body_ids()
            obstacles.remove(self.obj_in_hand.body_id[0])
            plan = plan_hand_motion_br(
                robot=self.robots[0],
                obj_in_hand=self.obj_in_hand,
                end_conf=[x, y, z + 0.1, 0, np.pi * 5 / 6.0, 0],
                hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
                obstacles=obstacles,
            )  #
            p.restoreState(state)
            p.removeState(state)

            if plan:
                for x, y, z, roll, pitch, yaw in plan:
                    self.robots[0].parts["right_hand"].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
                    p.stepSimulation()
                released_obj = self.obj_in_hand
                self.obj_in_hand = None

                # release hand
                self.reset_and_release_hand()

                # force release object to avoid dealing with stateful AG release mechanism
                self.robots[0].parts["right_hand"].force_release_obj()
                self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())
                # reset hand

                # reset the released object to zero velocity
                p.resetBaseVelocity(released_obj.get_body_id(), linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

                # let object fall
                for _ in range(100):
                    p.stepSimulation()

    def place_obj_plan(self, original_state, target_pos, target_orn, use_motion_planning=False):
        pos = self.obj_in_hand.get_position()
        p.restoreState(original_state)
        p.removeState(original_state)
        if not use_motion_planning:
            self.reset_and_release_hand()
            self.robots[0].parts["right_hand"].force_release_obj()
            self.obj_in_hand.set_position_orientation(target_pos, target_orn)
            self.obj_in_hand = None

        else:
            x, y, z = target_pos
            hand_x, hand_y, hand_z = self.robots[0].parts["right_hand"].get_position()

            minx = min(x, hand_x) - 1
            miny = min(y, hand_y) - 1
            minz = min(z, hand_z) - 0.5
            maxx = max(x, hand_x) + 1
            maxy = max(y, hand_y) + 1
            maxz = max(z, hand_z) + 0.5

            state = p.saveState()
            obstacles = self.get_body_ids()
            obstacles.remove(self.obj_in_hand.body_id[0])
            plan = plan_hand_motion_br(
                robot=self.robots[0],
                obj_in_hand=self.obj_in_hand,
                end_conf=[x, y, z + 0.1, 0, np.pi * 5 / 6.0, 0],
                hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
                obstacles=obstacles,
            )  #
            p.restoreState(state)
            p.removeState(state)

            if plan:
                for x, y, z, roll, pitch, yaw in plan:
                    self.robots[0].parts["right_hand"].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
                    p.stepSimulation()
                released_obj = self.obj_in_hand
                self.obj_in_hand = None

                # release hand
                self.reset_and_release_hand()

                # force release object to avoid dealing with stateful AG release mechanism
                self.robots[0].parts["right_hand"].force_release_obj()
                self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())
                # reset hand

                # reset the released object to zero velocity
                p.resetBaseVelocity(released_obj.get_body_id(), linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

                # let object fall
                for _ in range(100):
                    p.stepSimulation()

                return plan

    # def convert_nav_action_to_17dim_action(nav_action):
    #     """
    #     :param nav_action: a 3-dim numpy array that represents the x-position, y-position and 
    #     z-rotation of the robot in the world frame 
    #     """
    #     curr_body_pos = self.robots[0].get_position()
    #     curr_body_eul = p.getEulerFromQuaternion(self.robots[0].parts["right_hand"].get_orientation())
    #     curr_right_pos = self.robots[0].parts["right_hand"].get_position()
    #     curr_right_eul = p.getEulerFromQuaternion(self.robots[0].parts["right_hand"].get_orientation())

    #     curr_body_pos_arr = np.zeros(3)
    #     curr_body_pos_arr[0:2] = np.array(curr_body_pos)
    #     new_body_pos_arr = np.zeros(3)
    #     curr_body_pos_arr[0:2] = np.array(nav_action[0:2])
    #     new_body_pos_arr[2] = curr_body_pos_arr[2]
    #     right_pos_offset = curr_body_pos_arr - np.array(curr_right_pos)
    #     new_right_pos = new_body_pos_arr - right_pos_offset



    def sample_fn(self):
        random_point = self.scene.get_random_point()
        x, y = random_point[1][:2]
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    def navigate_to_obj_pos(self, obj, pos_offset):
        """
        navigates the robot to a certain x,y position and selects an orientation
        such that the robot is facing the object. If the navigation is infeasible,
        returns an indication to this effect.

        :param obj: an object to navigate toward
        :param to_pos: a length-2 numpy array (x, y) containing a position to navigate to
        """

        # test agent positions around an obj
        # try to place the agent near the object, and rotate it to the object
        valid_position = None  # ((x,y,z),(roll, pitch, yaw))
        original_position = self.robots[0].get_position()
        original_orientation = self.robots[0].get_orientation()
        base_diff_fn = get_base_difference_fn()

        if isinstance(obj, URDFObject): # must be a URDFObject so we can get its position!
            obj_pos = obj.get_position()
            pos = [pos_offset[0] + obj_pos[0], pos_offset[1] + obj_pos[1], self.robots[0].initial_z_offset]
            yaw_angle = np.arctan2(pos_offset[1], pos_offset[0])
            orn = [0, 0, yaw_angle]
            self.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
            eye_pos = self.robots[0].parts["eye"].get_position()
            ray_test_res = p.rayTest(eye_pos, obj_pos)
            # Test to see if the robot is obstructed by some object, but make sure that object 
            # is not either the robot's body or the object we want to pick up!
            blocked = len(ray_test_res) > 0 and (ray_test_res[0][0] not in (self.robots[0].parts["body"].get_body_id(), obj.get_body_id()))
            if not detect_robot_collision(self.robots[0]) and not blocked:
                valid_position = (pos, orn)

        if valid_position is not None:
            self.robots[0].set_position_orientation(original_position, original_orientation)
            plan = plan_base_motion_br(
                robot=self.robots[0],
                end_conf=[valid_position[0][0], valid_position[0][1], valid_position[1][2]],
                base_limits=(),
                obstacles=self.get_body_ids(),
                override_sample_fn=self.sample_fn,
            )

            plan_num_steps = len(plan)
            ret_actions = np.zeros((17, plan_num_steps - 1))
            plan_arr = np.array(plan)

            for i in range(1, plan_num_steps):
                ret_actions[0:3, i] = base_diff_fn(plan_arr[:, i], plan_arr[:, i-1])
            
            return ret_actions, True
            
        else:
            print("Position commanded is in collision!")
            self.robots[0].set_position_orientation(original_position, original_orientation)
            return np.zeros(17), False


    def reset(self, resample_objects=False):
        obs = super(BehaviorTAMPEnv, self).reset()
        self.obj_in_hand = None

        return obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="igibson/examples/configs/behavior.yaml",
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "gui", "iggui", "pbgui"],
        default="gui",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = BehaviorTAMPEnv(
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 300.0,
        physics_timestep=1.0 / 300.0,
        use_motion_planning=True,
    )
    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(reward, info)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
