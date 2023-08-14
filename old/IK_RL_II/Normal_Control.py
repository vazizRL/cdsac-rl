import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

PATH_TO_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__)))  # Path to robot models

import math
import gym
from gym import spaces
# from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from pkg_resources import parse_version
from scipy.spatial import distance
from operator import add

largeValObservation = 100

RENDER_HEIGHT = 1000   # 720
RENDER_WIDTH = 1500    # 960

# FRANKA_CONTROLLED_JOINTS = 11
FRANKA_JOINT_INDICES_TO_CONTROL = [0, 1, 2, 3, 4, 5, 6, 9, 10]
FRANKA_END_EFFECTOR_IDX = 9
FRANKA_END_EFFECTOR_IDX_L = 10


# TARGET POSITION is [0.64, -0.1, 0.7]


class PandaRobotEnv_(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=5,
                 # actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=70000,  # 7000
                 fixedActionRepetitions=False,
                 distSpecifications=None,
                 maxDist=0.25,  # old val. 0.08
                 maxDeviation=0.25,
                 maxErrorPos=0.1,
                 targetPos=[0.5, 0.0, 0.79],
                 evalFlag=False):


        if distSpecifications is None:
            distSpecifications = [0, 'A']  # 0 = Euclidean distance, A = Use improved distance metric

        # Parameter settings
        self._distance_measure_specifications = distSpecifications
        self._fixed_nr_action_repetitions = fixedActionRepetitions
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._maxDist = maxDist
        self._maxDeviation = maxDeviation
        self._maxErrorPos = maxErrorPos
        self._evalFlag = evalFlag
        self._ray = None
        self._contact_with_force = False
        self._targetPos = targetPos
        self.rewardedOnce = False
        self._cumulative_reward = 0
        self._manual_control = False
        # Debugging
        # self.debugReward = 0
        self._jd = [0.1]*9

        # Rendering
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.7
        self._cam_yaw = 180
        self._cam_pitch = -40

        # Observations & Measurements
        self._envStepCounter = 0
        self.grasps_per_update_interval = 0
        self.grasp_time_steps_needed_per_update_interval = []
        self._observation = []
        self._joint_pos = []
        self._goal_pos = []
        self._gripper_pos = []
        self.contact_info_l = None
        self.contact_info_r = None
        # self._gripper_width = None
        self._gripper_orn_vec = []
        self._gripper_to_goal_vec = []
        self._object_to_target_vec = []
        self._dist_to_obj_primary = 0.0  # Metric used for determination of terminal states
        self._dev_from_goal_vec_primary = 0.0  # Metric used for determination of terminal states
        self._dist_to_target_hold_primary = 9.0  # Metric used for determination of terminal states, init. with abritraty large val.
        self._dist_to_obj_secondary = 0.0  # Metric used for reward computation
        self._dev_from_goal_vec_secondary = 0.0  # Metric used for reward computation
        self._dist_to_target_hold_secondary = 3  # Metric used for reward computation
        self._reward_dist = 0.0  # self._dist_to_obj_secondary translated to a reward
        self._reward_dist_hold = 0.0  # self._dist_to_target_hold_secondary translated to reward
        self._reward_goal_dir = 0.0  # self._dev_from_goal_vec_secondary translated to a reward
        self._rt = 0.008
        self._EE_pos = None
        self._11_to_prepos = None
        self._above_goal_pos = 0.15
        self._not_reached = True
        self._not_reached_2 = True
        self._pause_before_trans = 0
        self._not_reached_3 = False
        self._third_IK_pos = [0.4, 0.6, 0.79]


        # Simulation
        self._urdfRoot = urdfRoot
        self._trayUid = None
        self.blockUid = None
        self._indicator = None

        self._p = p

        # self._robo_path = PATH_TO_MODELS + 'RobotModels/Panda/deps/Panda/panda.urdf'
        self._robo_path = 'C:/Users/XMG/anaconda3/envs/Prototyp_IK/Lib/site-packages/pybullet_data/franka_panda/panda.urdf'

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        # self.seed()
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0, 0], useFixedBase=1)
        # self._num_joints = min(FRANKA_CONTROLLED_JOINTS, self._p.getNumJoints(self._robot))    # Nr of controlled joints
        self._num_joints = self._p.getNumJoints(self._robot)
        self._joint_indices_control = FRANKA_JOINT_INDICES_TO_CONTROL
        self._gripperIndex = min(FRANKA_END_EFFECTOR_IDX,
                                 self._p.getNumJoints(self._robot))  # Index of end-effector-link for Franka Emika Panda
        self._gripperIndex_L = FRANKA_END_EFFECTOR_IDX_L
        self.reset()

        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)

        # Network may have to predict number of repetitions how many times a joint config is to be applied in a row by
        # means of an additional action node:
        additional_action_node = 0 if self._fixed_nr_action_repetitions else 1

        if self._isDiscrete:
            self.action_space = spaces.Discrete(3 + additional_action_node)
        else:
            action_dim = len(self._joint_indices_control) + additional_action_node
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    ##################################
    # Further added helper functions #
    ##################################

    def divergence(self, a, b):
        """
            Distance measure:
            Computed Divergence of two points. Formula (45) from following paper:
                http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.154.8446&rep=rep1&type=pdf

            :param a: Data point a
            :param b: Data point b (has to have same length as a)
        """

        # if not len(a) == len(b):
        #    raise AssertionError

        d = 0
        for i in range(len(a)):
            d += (((a[i] - b[i]) ** 2) / ((a[i] + b[i]) ** 2))
        d *= 2

        return d

    def get_joint_config(self, nr_joints=None):
        """
            Returns a random joint angle in range [-1,1] in radians. One random angle is generated for each controlled joint
            from joint 1 through nr_joints, where nr_joints is the number of joints for which a random config has to be
            returned.

            :param nr_joints: Random joint angles are generated for joints 1 through nr_joints. Optional.
            :return: List of joint angles. One random joint angle in radians per requested joint
        """
        # if nr_joints is None:
        #     nr_joints = len(self._joint_indices_control)
        # return [random.uniform(-1, 1) for _ in range(nr_joints)]

        # For debugging only. Random joint config. is recommended to artificially increase the
        # exploitation
        return [0, 0.65, 0, -1, 0, 1.5, 0.9, 0, 0]

    def apply_joint_config(self, config):
        """
            Instantaneously apply a given joint configuration to a robotic arm.

            :param config: list of joint angles to be applied to joints 1 through n, respectively;
                           n=number of joint angles provided
            :return: -
        """
        for i in range(len(config)):
            self._p.resetJointState(self._robot, i, config[i])
        # time.sleep(1)  # Observe pose

    def get_normalized_vector_from_a_to_b(self, a, b):
        vec = []
        len_of_vec = 0
        for i in range(len(a)):
            entry = b[i] - a[i]
            vec.append(entry)
            # len_of_vec += abs(entry)
            len_of_vec += abs(entry) ** 2
        len_of_vec = np.sqrt(len_of_vec)
        for i in range(len(vec)):
            vec[i] /= len_of_vec
        return vec

    def euler_to_vec_gripper_orientation(self, yaw, pitch=None, roll=None):
        """
           Returns the orientation of the z-axis of the gripper with respect to the world/universe-reference
           coordinate system. The z-axis of the gripper is the one pointing along the direction of the fingers
           of the gripper.
        """
        if isinstance(yaw, tuple):
            yaw = list(yaw)
        if isinstance(yaw, list):
            # list of angles is provided in yaw-variable
            pitch, roll, yaw = yaw[1], yaw[2], yaw[0]

        # Construct rotation matrices
        sin = math.sin
        cos = math.cos
        Rx_yaw = np.array([[1, 0, 0], [0, cos(yaw), -sin(yaw)], [0, sin(yaw), cos(yaw)]])
        Rz_rol = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
        Ry_pit = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])

        # Rotate a coord system initially coincident with ref frame in exact same way as roll/pitch/yaw are applied in
        # simulation. To get the orientation of the axes defining the coord system attached to COM (=Center of mass) of
        # end-effector (=gripper) expressed with respect to ref system.
        R = Rz_rol.dot(Ry_pit.dot(Rx_yaw))

        ee_z_axis = R[:, 2]  # Direction of z-axis of coord system expressing gripper orientation wrt reference frame
        ee_x_axis = R[:, 0]
        ee_y_axis = R[:, 1]

        return ee_z_axis, ee_x_axis, ee_y_axis



    def normalize_vector(self, vec):
        len_of_vec = 0
        for i in range(len(vec)):
            # len_of_vec += abs(vec[i])
            len_of_vec += abs(vec[i]) ** 2
        len_of_vec = np.sqrt(len_of_vec)
        for i in range(len(vec)):
            vec[i] /= len_of_vec
        return vec

    def get_vector_from_a_to_b(self, vec1, vec2):
        vec_ret = []
        for i in range(len(vec1)):
            vec_ret.append(vec1[i] - vec2[i])
        return vec_ret

    def obtain_measurements(self):
        """
            Obtain general purpose measurements like positions of goal object and gripper's center of mass as well as
            measures used for later reward computations.

            :return: -
        """

        ### Obtain general measurements:

        ## Obtain robot's joint space positions for all controlled joints in [joint{1}, joint{nr_controlled_joints}]
        # self._joint_pos = [j[0] for j in self._p.getJointStates(self._robot, range(self._num_joints))]
        self._joint_pos = [j[0] for j in self._p.getJointStates(self._robot, self._joint_indices_control)]

        # GRIPPER WIDTH
        # self._gripper_width = self._p.getJointState(self._robot, -1)

        ## Calculate measure of how close the end-effector (=gripper's Center of Mass (COM)) is to the goal location:
        # Obtain world information
        self._goal_pos, self._goal_orn = p.getBasePositionAndOrientation(self.blockUid)
        self._gripper_pos, gripperOrn_quat = p.getLinkState(self._robot, self._gripperIndex)[0:2]
        self._gripper_pos_l = p.getLinkState(self._robot, self._gripperIndex_L)[0]

        # Euclidean/Cartesian (straight-line) distance calculation from gripper's COM to goal's COM coordinates
        euclideanDistance = distance.euclidean(self._gripper_pos, self._goal_pos)

        # euclideanDistanceTarget = distance.euclidean(self.blockUid, self._targetPos)

        ## Measure of how precisely end-effector (=gripper's fingers) points towards goal location:
        # Calculate vectors
        gripperOrn_eul = self._p.getEulerFromQuaternion(gripperOrn_quat)
        # self._helper_vec = self.get_normalized_vector_from_a_to_b(self._gripper_pos, self._gripper_pos_l)
        self._helper_vec = self.get_vector_from_a_to_b(self._gripper_pos, self._gripper_pos_l)
        self._helper_vec = [self._helper_vec[i] * -0.5 for i in range(len(self._helper_vec))]
        gripper_dx = list(map(add, self._gripper_pos, self._helper_vec))
        self._gripper_to_goal_vec = self.get_normalized_vector_from_a_to_b(gripper_dx, self._goal_pos)
        self._object_to_target_vec = self.get_normalized_vector_from_a_to_b(self._goal_pos, self._targetPos)
        self._gripper_orn_vec = self.euler_to_vec_gripper_orientation(gripperOrn_eul)[0]
        self._gripper_orn_vec = self.normalize_vector(self._gripper_orn_vec)
        # gripper_r_copy = list(map(add, self._gripper_pos, self._gripper_orn_vec*0.025))
        # self._line2 = self._p.addUserDebugLine(gripper_dx, self._goal_pos, [0,0,0],2)

        # Euclidean distance between vector pointing straight from gripper's COM location towards goal's COM
        # location and vector containing direction of z-axis of coordinate system expressing the orientation of COM
        # of end-effector (expressed with respect to reference frame attached to base of robotic arm).
        # (Z-axis of end-effector frame points from end-effector's COM towards its fingers.)
        euclideanDeviation = distance.euclidean(self._gripper_to_goal_vec, self._gripper_orn_vec)

        ### Obtain reward measurements:

        ## Distance based reward signal

        # Compute distance measure underlying continuous distance-based reward signal
        if 0 in self._distance_measure_specifications:
            # Euclidean distance
            newRewardDistance = euclideanDistance
        elif 1 in self._distance_measure_specifications:
            # Divergence metric
            newRewardDistance = self.divergence(self._gripper_pos, self._goal_pos)
        else:
            newRewardDistance = 0.0

        # Translate variable distance measure computed above into a reward:
        if 'A' in self._distance_measure_specifications:
            # Reward the reduction of goal distance compared to that of previous time-step
            self._reward_dist = self._dist_to_obj_secondary - newRewardDistance
        elif 'B' in self._distance_measure_specifications:
            # Reward the absolute negative distance from gripper to goal
            self._reward_dist = -newRewardDistance
        elif 'C' in self._distance_measure_specifications:
            # No continuous distance based reward signal
            self._reward_dist = 0.0
        else:
            pass

        ## Deviation based reward signal

        # Compute deviation measure underlying continuous deviation-from-goal-direction-based reward signal
        if 0 in self._distance_measure_specifications:
            # Euclidean distance
            newRewardDeviation = euclideanDeviation
        elif 1 in self._distance_measure_specifications:
            # Divergence metric
            newRewardDeviation = self.divergence(self._gripper_to_goal_vec, self._gripper_orn_vec)
        else:
            newRewardDeviation = 0.0

        # Translate variable deviation-from-goal-direction measure computed above into a reward:
        if 'A' in self._distance_measure_specifications:
            # Reward reduction of deviance from gripper's orientation to vector pointing to goal compared to that of
            # previous time-step
            self._reward_goal_dir = self._dev_from_goal_vec_secondary - newRewardDeviation
        elif 'B' in self._distance_measure_specifications:
            # Reward the absolute negative deviation of gripper's orientation from vector pointing towards goal
            self._reward_goal_dir = -newRewardDeviation
        elif 'C' in self._distance_measure_specifications:
            # No continuous deviation based reward signal
            self._reward_goal_dir = 0.0
        else:
            pass

        self._dist_to_obj_primary = euclideanDistance
        self._dist_to_obj_secondary = newRewardDistance
        self._dev_from_goal_vec_primary = euclideanDeviation
        self._dev_from_goal_vec_secondary = newRewardDeviation

        ray_start = list(self._p.getLinkState(self._robot, 9)[0])
        ray_start = list(map(add, ray_start, self._gripper_orn_vec * 0.025))
        ray_end = list(self._p.getLinkState(self._robot, 10)[0])
        ray_end = list(map(add, ray_end, self._gripper_orn_vec * 0.025))

        # Add an indicator to see the trajectory of the ray; for debugging only, as the 3D line
        # considerably slows down computation speed
        # self._indicator = self._p.addUserDebugLine(self._gripper_pos, [1,1,1], [0, 0, 0], lineWidth=2)

        # Raytest to check if object is in between the fingers
        self._ray = self._p.rayTest(ray_start, ray_end)

        #Raytest for collision prevention
        self._gripper_orn_world = self._p.getLinkState(self._robot, self._gripperIndex)[5]
        self._gripper_orn_world = self._p.getEulerFromQuaternion(self._gripper_orn_world)
        self._gripper_orn_world_x = self.euler_to_vec_gripper_orientation(self._gripper_orn_world)[1]
        self._gripper_orn_world_y = self.euler_to_vec_gripper_orientation(self._gripper_orn_world)[2]

        magnitude = 0.03
        magnitude_d = 0.0001

        startpos_r = list(self._gripper_pos)
        startpos_r = list(map(add, startpos_r, self._gripper_orn_vec*0.03))

        startpos_l = list(self._gripper_pos_l)
        startpos_l = list(map(add, startpos_l, self._gripper_orn_vec * 0.03))

        endpos_r_f = [startpos_r[0]+self._gripper_orn_world_x[0]*magnitude,
                      startpos_r[1]+self._gripper_orn_world_x[1]*magnitude,
                      startpos_r[2]+self._gripper_orn_world_x[2]*magnitude]

        endpos_r_b = [startpos_r[0] + self._gripper_orn_world_x[0] * -magnitude,
                      startpos_r[1] + self._gripper_orn_world_x[1] * -magnitude,
                      startpos_r[2] + self._gripper_orn_world_x[2] * -magnitude]

        endpos_r_d = [startpos_r[0] + self._gripper_orn_vec[0] * magnitude_d,
                      startpos_r[1] + self._gripper_orn_vec[1] * magnitude_d,
                      startpos_r[2] + self._gripper_orn_vec[2] * magnitude_d]

        endpos_r_ri = [startpos_r[0] + self._gripper_orn_world_y[0] * magnitude,
                      startpos_r[1] + self._gripper_orn_world_y[1] * magnitude,
                      startpos_r[2] + self._gripper_orn_world_y[2] * magnitude]

        endpos_l_f = [startpos_l[0] + self._gripper_orn_world_x[0] * magnitude,
                      startpos_l[1] + self._gripper_orn_world_x[1] * magnitude,
                      startpos_l[2] + self._gripper_orn_world_x[2] * magnitude]

        endpos_l_b = [startpos_l[0] + self._gripper_orn_world_x[0] * -magnitude,
                      startpos_l[1] + self._gripper_orn_world_x[1] * -magnitude,
                      startpos_l[2] + self._gripper_orn_world_x[2] * -magnitude]

        endpos_l_d = [startpos_l[0] + self._gripper_orn_vec[0] * magnitude_d,
                      startpos_l[1] + self._gripper_orn_vec[1] * magnitude_d,
                      startpos_l[2] + self._gripper_orn_vec[2] * magnitude_d]

        endpos_l_le = [startpos_l[0] + self._gripper_orn_world_y[0] * -magnitude,
                      startpos_l[1] + self._gripper_orn_world_y[1] * -magnitude,
                      startpos_l[2] + self._gripper_orn_world_y[2] * -magnitude]

        self._ray_r_f = self._p.rayTest(startpos_r, endpos_r_f)
        self._ray_r_b = self._p.rayTest(startpos_r, endpos_r_b)
        self._ray_r_d = self._p.rayTest(startpos_r, endpos_r_d)
        self._ray_r_ri = self._p.rayTest(startpos_r, endpos_r_ri)
        #
        self._ray_l_f = self._p.rayTest(startpos_l, endpos_l_f)
        self._ray_l_b = self._p.rayTest(startpos_l, endpos_l_b)
        self._ray_l_d = self._p.rayTest(startpos_l, endpos_l_d)
        self._ray_l_le = self._p.rayTest(startpos_l, endpos_l_le)

        # self._p.addUserDebugLine(startpos_r, endpos_r_d, lineColorRGB=[0,0,0], lineWidth=2)
        # self._p.addUserDebugLine(startpos_r, endpos_r_d, lineColorRGB=[0, 0, 0], lineWidth=2)
        # self._p.addUserDebugLine(startpos_l, endpos_l_f, lineColorRGB=[0, 0, 0], lineWidth=2)
        # self._p.addUserDebugLine(startpos_l, endpos_l_d, lineColorRGB=[0, 0, 0], lineWidth=2)

        #######

        self.get_contact_and_target_dist()

        #IK Extension
        self._EE_pos = self._p.getLinkState(self._robot, 11)[0]
        self._gripper_prepos = [self._goal_pos[0], self._goal_pos[1], self._goal_pos[2] + self._above_goal_pos]
        self._11_to_prepos = distance.euclidean(self._EE_pos, self._gripper_prepos)
        self._new_goal_pos = [self._goal_pos[0], self._goal_pos[1], self._goal_pos[2] +0.0015]
        self._11_to_gripper_state = distance.euclidean(self._EE_pos, self._new_goal_pos)
        self._11_to_third_IK = distance.euclidean(self._EE_pos, self._third_IK_pos)

    def reset_logged_train_data(self):
        """
            For logging training progress. Called via PPO agent's callback.
            :return: -
        """
        self.grasps_per_update_interval = 0
        self.grasp_time_steps_needed_per_update_interval = []

    def get_eval_info(self, eval_time_steps=1000):
        """
            Returns information specifically for evaluation phase.
        :return: First position: 1 if goal reached else 0; Second: time steps elapsed since last reset; Third: whether
                    terminated or not
        """
        binary_reward = 0

        if self._dist_to_obj_primary < self._maxDist and self._dev_from_goal_vec_primary < self._maxDeviation:
            binary_reward += 1

        done = (self._envStepCounter == eval_time_steps) or (binary_reward == 1)

        return [binary_reward, self._envStepCounter, done]

    def set_step_counter(self, time_steps):
        self._envStepCounter = time_steps

    # Returns the reward when object is hold with N>=10 and moved towards self._targetPos
    def get_contact_and_target_dist(self):
        self.contact_info_l = self._p.getContactPoints(self.blockUid, self._robot, linkIndexB=10)
        self.contact_info_r = self._p.getContactPoints(self.blockUid, self._robot, linkIndexB=9)
        if self.contact_info_l and self.contact_info_l[0][9] >= 10 and self.contact_info_r and \
                self.contact_info_r[0][9] >= 10:
            self._contact_with_force = 1
            euclidean_distance = distance.euclidean(self._goal_pos, self._targetPos)
            self._dist_to_target_hold_primary = euclidean_distance

            self._reward_dist_hold = self._dist_to_target_hold_secondary - euclidean_distance
            self._dist_to_target_hold_secondary = euclidean_distance
        else:
            self._contact_with_force = 0
            self._reward_dist_hold = 0.0

    def collision_control(self):
        self.elevenorn = self._p.getLinkState(self._robot, 11)[1]
        if not self._ray_r_f[0][0] in [-1, 6]:
            print('F Activated')
            magnitude = 0.1
            revert = [self._ray_r_f[0][3][0] + self._gripper_orn_world_x[0] * -magnitude,
                      self._ray_r_f[0][3][1] + self._gripper_orn_world_x[1] * -magnitude,
                      self._ray_r_f[0][3][2] + self._gripper_orn_world_x[2] * -magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)

        elif not self._ray_r_b[0][0] in [-1, 6]:
            print('B Activated')
            magnitude = 0.10
            revert = [self._ray_r_b[0][3][0] + self._gripper_orn_world_x[0] * magnitude,
                      self._ray_r_b[0][3][1] + self._gripper_orn_world_x[1] * magnitude,
                      self._ray_r_b[0][3][2] + self._gripper_orn_world_x[2] * magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)

        elif not self._ray_r_d[0][0] in [-1, 6]:
            print('D is activated')
            magnitude = 0.1
            revert = [self._ray_r_d[0][3][0] + self._gripper_orn_vec[0] * -magnitude,
                      self._ray_r_d[0][3][1] + self._gripper_orn_vec[1] * -magnitude,
                      self._ray_r_d[0][3][2] + self._gripper_orn_vec[2] * -magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)

        elif not self._ray_r_ri[0][0] in [-1, 6]:
            print('ri is activated')
            magnitude = 0.1
            revert = [self._ray_r_ri[0][3][0] + self._gripper_orn_world_y[0] * -magnitude,
                      self._ray_r_ri[0][3][1] + self._gripper_orn_world_y[1] * -magnitude,
                      self._ray_r_ri[0][3][2] + self._gripper_orn_world_y[2] * -magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)
            ############Collision Control Left###############
        elif not self._ray_l_f[0][0] in [-1, 6]:
            print('LF Activated')
            magnitude = 0.1
            revert = [self._ray_l_f[0][3][0] + self._gripper_orn_world_x[0] * -magnitude,
                      self._ray_l_f[0][3][1] + self._gripper_orn_world_x[1] * -magnitude,
                      self._ray_l_f[0][3][2] + self._gripper_orn_world_x[2] * -magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)

        elif not self._ray_l_b[0][0] in [-1, 6]:
            print('LB Activated')
            magnitude = 0.10
            revert = [self._ray_l_b[0][3][0] + self._gripper_orn_world_x[0] * -magnitude,
                      self._ray_l_b[0][3][1] + self._gripper_orn_world_x[1] * -magnitude,
                      self._ray_l_b[0][3][2] + self._gripper_orn_world_x[2] * -magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)

        elif not self._ray_l_d[0][0] in [-1, 6]:
            print('LD is activated')
            magnitude = 0.1
            revert = [self._ray_l_d[0][3][0] + self._gripper_orn_vec[0] * -magnitude,
                      self._ray_l_d[0][3][1] + self._gripper_orn_vec[1] * -magnitude,
                      self._ray_l_d[0][3][2] + self._gripper_orn_vec[2] * -magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)

        elif not self._ray_l_le[0][0] in [-1, 6]:
            print('le is activated')
            magnitude = 0.1
            revert = [self._ray_l_le[0][3][0] + self._gripper_orn_world_y[0] * magnitude,
                      self._ray_l_le[0][3][1] + self._gripper_orn_world_y[1] * magnitude,
                      self._ray_l_le[0][3][2] + self._gripper_orn_world_y[2] * magnitude]
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, revert, self.elevenorn,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)

    ##############################
    # End added helper functions #
    ##############################

    def reset(self):
        self.terminated = 0
        self.rewardedOnce = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, 0])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, 0,
                   0.000000, 0.000000, 0.0, 1.0)

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.50000, 0,
                   0.000000, 0.000000, 0.0, 1.0)

        self._trayUid = p.loadURDF(os.path.join(self._urdfRoot, "tray/tray.urdf"), 0.450000,
                                   -0.1, 0.63, 0.000000, 0.000000, 1.000000, 0.000000)

        self._trayUid = p.loadURDF(os.path.join(self._urdfRoot, "tray/tray.urdf"), 0.450000,
                                   0.6, 0.63, 0.000000, 0.000000, 1.000000, 0.000000)

        xpos = 0.35 + 0.12 * random.random()
        ypos = 0.5 + 0.2 * random.random()
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        # orn = p.getQuaternionFromEuler([0, 0, ang])
        orn = p.getQuaternionFromEuler([0, 0, math.pi / 2])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, 0.635,
                                   orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0.25, 0.8], useFixedBase=1)  # 0.6
        self._envStepCounter = 0

        # Set robotic arm to random initial pose
        self.apply_joint_config(self.get_joint_config())

        p.stepSimulation()
        self.obtain_measurements()
        self._observation = self.getExtendedObservation()
        # self.debugReward = 0
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def getExtendedObservation(self):
        """
            Construct an observation that serves as input to the reinforcement learning agent.
            :return: Representation of state of robotic arm and its surrounding
        """
        self._observation = []
        self._observation.extend(self._goal_pos)  # Cartesian position of goal
        self._observation.extend(self._gripper_pos)  # Cartesian position of end-effector (right)
        self._observation.extend(self._gripper_pos_l)  # Cartesian position of end-effector (left)
        self._observation.extend(self._gripper_to_goal_vec)  # Normalized vector from gripper to goal
        self._observation.extend(self._gripper_orn_vec)  # Orientation of gripper's z-axis wrt reference frame
        self._observation.extend(self._joint_pos)  # Joint angles
        # self._observation.extend(self._targetPos)              # Cartesian coordinate of the target position
        self._observation.extend([self._contact_with_force])  # Boolean; checks grip state
        self._observation.extend(self._object_to_target_vec)  # Normalized vector from object to target position

        return self._observation

    def step(self, action):

        # Determine how many times in a row commanded actions are supposed to be executed. Min=1, Max=10 times.
        if self._fixed_nr_action_repetitions:
            repetitions = self._actionRepeat
        else:
            repetitions = 2
            # minRep, maxRep = 1, 10
            # repetitions = min(maxRep, max(minRep, int(abs(action[-1] * 10))))

        if self._manual_control:
            # Get joint angles for all controlled joints

            self.collision_control()

            jointPos = self._joint_pos.copy()
            for i in range(len(self._joint_indices_control)-1):
                jointPos[i] = jointPos[i] + np.clip(action[i], -0.5, 0.5)
            jointPos[-1] =jointPos[-1] + action[-1]

            self._joint_pos = jointPos

            # Calculate IK with orientation
        else:
            #targetVel = [2.618, 2.618, 2.618, 2.618, 3.142, 3.142, 3.142, 0.1, 0.1]

            # print('IK I is activated')
            # Set new, desired joint configurations
            if self._11_to_prepos > 0.01 and self._not_reached:
                self._orn_at_prepos = self._p.getQuaternionFromEuler([self._goal_orn[0], -math.pi, self._goal_orn[1]])
                self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, self._gripper_prepos,
                                                                     self._orn_at_prepos,
                                                                     jointDamping=self._jd,
                                                                     solver=0,
                                                                     maxNumIterations=100,
                                                                     residualThreshold=self._rt
                                                                    )
                self._joint_pos = list(self._joint_pos)
                self.collision_control()
                print(self._joint_pos)
                #xy:

            elif self._11_to_gripper_state > 0.001 and self._not_reached_2:
                grip_pos = [self._gripper_prepos[0], self._gripper_prepos[1], self._gripper_prepos[2]-self._above_goal_pos+0.0015]
                # orn = self._p.getQuaternionFromEuler([self._goal_orn[1], -math.pi, self._goal_orn[0]])
                self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, grip_pos, self._orn_at_prepos,
                                                              jointDamping=self._jd,
                                                              solver=0,
                                                              maxNumIterations=100,
                                                              residualThreshold=0.001
                                                              )

                self._joint_pos = list(self._joint_pos)
                self.collision_control()

                self._joint_pos[-1] = self._action_bound
                self._joint_pos[-2] = self._action_bound

                self._not_reached = False

            else:
                self._joint_pos[-1] = -self._action_bound
                self._joint_pos[-2] = -self._action_bound
                self._not_reached_2 = False
                self._pause_before_trans += 1

        if not self._not_reached_2 and self._pause_before_trans > 10:

            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, self._third_IK_pos,
                                                                 self._orn_at_prepos,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)
            self._joint_pos[-1] = -self._action_bound
            self._joint_pos[-2] = -self._action_bound

        if self._contact_with_force and self._11_to_third_IK < 0.01:
            self._not_reached_3 = True

        if self._not_reached_3:
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, self._targetPos,
                                                                 self._orn_at_prepos,
                                                                 jointDamping=self._jd,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._rt
                                                                 )
            self._joint_pos = list(self._joint_pos)
            self._joint_pos[-1] = -self._action_bound
            self._joint_pos[-2] = -self._action_bound

        self._p.setJointMotorControlArray(bodyUniqueId=self._robot,
                                          # jointIndices=range(self._num_joints),
                                          # jointIndices=self._joint_indices_control,
                                          #jointIndices=range(len(self._joint_indices_control)),
                                          jointIndices=self._joint_indices_control,
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=self._joint_pos,
                                          forces=[55] * len(self._joint_indices_control))

        # Step a simulations given number of times to get old joint configurations closer to the new, desired ones
        for i in range(repetitions):

            p.stepSimulation()
            self.obtain_measurements()

            if self._termination:
                break
            self._envStepCounter += 1

        if self._renders:
            time.sleep(1/50)


        self._observation = self.getExtendedObservation()
        # print(self.contact_line)
        # if self._indicator:
        #     self._p.removeUserDebugItem(self._indicator)
        # self._p.removeUserDebugItem(self._line2)
        # del(self._ray)

        done = self._termination
        done = done if not self._evalFlag else False  # In case of evaluation run, eval script will handle termination

        reward = self._reward()
        # print(reward)
        info = self.get_eval_info() if self._evalFlag else {}

        self._p.removeAllUserDebugItems()

        return np.array(self._observation), reward, done, info

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._robot)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    @property
    def _termination(self):

        if self.terminated or self._envStepCounter >= self._maxSteps:
            self._observation = self.getExtendedObservation()
            return True

        if self._dist_to_target_hold_primary < self._maxErrorPos:
            # Goal attained
            self.terminated = 1
            self._observation = self.getExtendedObservation()
            # Log training progress
            self.grasps_per_update_interval += 1
            self.grasp_time_steps_needed_per_update_interval.append(self._envStepCounter)
            return True

        return False

    def _reward(self):

        reward = 0

        # Distance reward
        # Distance needs adjustment only when being too distant from goal
        if self._dist_to_obj_primary > self._maxDist:
            reward += self._reward_dist

        # Orientation reward
        # Orientation needs adjustment only when being too far from desired orientation
        if self._dev_from_goal_vec_primary > self._maxDeviation:
            reward += self._reward_goal_dir

        if self._dist_to_obj_primary < self._maxDist and self._dev_from_goal_vec_primary < self._maxDeviation:
            if not self.rewardedOnce:
                reward += 2
            self.rewardedOnce = True

        # Find constant reward value for blockUid being in between the fingers
        # and a contact force > 10
        if self._ray[0][0] == self.blockUid and self._contact_with_force:
            reward += self._reward_dist_hold

        # Find a constant reward value that is appropriate
        # for blockUid being in between fingers ONLY
        if self._ray[0][0] == self.blockUid:
            reward += 0.00001

        if self._dist_to_target_hold_primary < self._maxErrorPos:
            goalReward = 2
            timeReward = 500 // self._envStepCounter
            reward += (goalReward + timeReward)
        elif self._envStepCounter >= self._maxSteps and not self.rewardedOnce:
            reward -= 2 * (self._maxSteps // 1000)
            # reward -= 2
        # print(reward)
        # print(self._reward_dist)
        # self.debugReward += reward
        # print(self.debugReward)

        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        # _seed = seed
        _step = step


environment = PandaRobotEnv_(renders=True)

param1 = p.addUserDebugParameter("a1", -0.5, 0.5, 0.0)
param2 = p.addUserDebugParameter("a2", -0.5, 0.5, 0)
param3 = p.addUserDebugParameter("a3", 0.0, 1.5, 0)
param4 = p.addUserDebugParameter("a4", -0.5, 0.5, 0)
param5 = p.addUserDebugParameter("a5", -0.5, 0.5, 0)
param6 = p.addUserDebugParameter("a6", -0.5, 0.5, 0)
param7 = p.addUserDebugParameter("a7", -0.5, 0.5, 0)
param8 = p.addUserDebugParameter("a8", -0.5, 0.5, 0)
# param9 = p.addUserDebugParameter("a9", -0.5, 0.5, 0)
# param10 = p.addUserDebugParameter("a10", -0.5, 0.5, 0)
# param11 = p.addUserDebugParameter("a11", -0.5, 0.5, 0)


for i in range(100000000):
    # actions = (p.readUserDebugParameter(param1), p.readUserDebugParameter(param2),
    #            p.readUserDebugParameter(param3), p.readUserDebugParameter(param4),
    #            p.readUserDebugParameter(param5), p.readUserDebugParameter(param6),
    #            p.readUserDebugParameter(param7), p.readUserDebugParameter(param8),
    #            p.readUserDebugParameter(param9))

    actions = (p.readUserDebugParameter(param1), p.readUserDebugParameter(param2),
               p.readUserDebugParameter(param3), p.readUserDebugParameter(param4),
               p.readUserDebugParameter(param5), p.readUserDebugParameter(param6),
               p.readUserDebugParameter(param7), p.readUserDebugParameter(param8)
               )

    # actions = (p.readUserDebugParameter(param1), p.readUserDebugParameter(param2),
    #            p.readUserDebugParameter(param3),
    #            )

    # print(environment._p.getNumJoints(environment._robot))
    # print(environment._p.getJointState(environment._robot, 10))
    # print(environment._dev_from_goal_vec_primary)
    # print(environment._gripper_pos)
    # print(environment._reward_goal_dir)
    # print(environment._robot)

    # if environment.contact_info_r and environment.contact_info_r[0][9]>20:
    #     print(environment.contact_info_r[0][9])
    # else:
    #     pass

    # if not environment.contact_info_r:
    #     pass
    # else:
    #     print(environment.contact_info_r[0][9])
    #
    # if not environment.contact_info_l:
    #     pass
    # else:
    #     print(environment.contact_info_l[0][9])
    # print(environment._ray[0][0]==environment.blockUid)
    # print(environment._gripper_orn_vec)
    # print(environment._envStepCounter)
    # print(environment._ray)
    # print(environment.contact_with_force)
    # print(environment._gripper_to_goal_vec)
    # print(environment._reward_dist_hold)
    # print(environment._dist_to_target_hold_primary)
    # print(environment.terminated)
    # print(environment._reward_dist)
    # print(environment._envStepCounter)
    environment.step(actions)






