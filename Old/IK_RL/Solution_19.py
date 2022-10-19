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

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

FRANKA_CONTROLLED_JOINTS = 11
FRANKA_JOINT_INDICES_TO_CONTROL = [0, 1, 2, 3, 4, 5, 6, 9, 10]
FRANKA_END_EFFECTOR_IDX = 9
FRANKA_END_EFFECTOR_IDX_L = 10


# EE middle: Index 11


# TARGET POSITION is [0.64, -0.1, 0.7]


class PandaRobotEnv_(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepetition=1,  # 5
                 # actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=True,
                 isDiscrete=False,
                 maxSteps=7000,  # 7000
                 fixedActionRepetitions=False,
                 distSpecifications=None,
                 # maxDist=0.25,  # Old val. 0.08
                 # maxDeviation=0.25,
                 maxErrorPos=0.05,
                 targetPos=[0.64, -0.13, 0.79],
                 evalFlag=False):

        if distSpecifications is None:
            distSpecifications = [0, 'A']  # 0 = Euclidean distance, A = Use improved distance metric

        # Parameter settings
        self._distance_measure_specifications = distSpecifications
        self._fixed_nr_action_repetitions = fixedActionRepetitions
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._action_repetition = actionRepetition
        self._isEnableSelfCollision = isEnableSelfCollision
        # self._maxDist = maxDist
        # self._maxDeviation = maxDeviation
        # self._maxErrorPos = maxErrorPos
        self._evalFlag = evalFlag
        self._ray = None
        self._contact_with_force = False
        self._target_pos = targetPos
        self._cumulative_reward = 0
        self._joint_indices_control = FRANKA_JOINT_INDICES_TO_CONTROL
        self._fixed_EE_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        self._joint_damping = [0.1] * 9 # Old 0.1
        self._residual_threshold = 0.002  # Old: 0.003
        self._max_IK_points = 2
        self._max_error_pos = maxErrorPos
        # Debugging
        # self.debugReward = 0
        self._action_low = np.array([-2, -2, -2])   # Old: [-15,-15,-15]
        self._action_high = np.array([2, 2, 2])  # Old: [15, 15, 15]
        self._boundary_selected = False
        self._previously_crashless = 0

        # Rendering
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.7
        self._cam_yaw = 180
        self._cam_pitch = -40

        # Observations & Measurements
        self._current_IK_points = 0
        self.grasps_per_update_interval = 0
        self.grasp_time_steps_needed_per_update_interval = []
        self._observation = []
        self._goal_pos = []
        self._goal_pos_prev = []
        self.contact_info_l = None
        self.contact_info_r = None
        self._EE_to_goal = None
        self._gripper_orn_vec = []
        self._goal_to_target = None
        self._joint_pos = []
        self._dist_to_obj_primary = 0.0  # Metric used for determination of terminal states
        self._dist_to_target_hold_primary = 0.0  # Metric used for determination of terminal states, init. with abritraty large val.
        self._dist_to_obj_secondary = 0.0  # Metric used for reward computation
        self._reward_dist = 0.0  # self._dist_to_obj_secondary translated to a reward
        self._reward_dist_goal_target = 0.0  # self._dist_to_target_hold_secondary translated to reward
        self._reward_goal_dir = 0.0  # self._dev_from_goal_vec_secondary translated to a reward
        self._EE_pos = []
        self._EE_pos_prev = []
        self._right_finger_pos = []
        self._left_finger_pos = []
        self._divergence_EEz_EEpos = []
        self._EE_to_IK = []
        self._envStepCounter = 0
        self._contact_with_env = False
        self._switch_grip_state = False
        self._initialize_grip_switch_std_t = True
        self._initialize_grip_switch_std_f = False
        self._avg_reward = []

        # Simulation
        self._urdfRoot = urdfRoot
        self._trayUid = None
        self._blockUid = None
        self._indicator = None

        self._p = p

        # self._robo_path = PATH_TO_MODELS + 'RobotModels/Panda/deps/Panda/panda.urdf'
        self._robo_path = 'C:/Users/XMG/anaconda3/envs/CognitiveRobotics_Robo_Control-master/Lib/site-packages/pybullet_data/franka_panda/panda.urdf'

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
        # self._joint_indices_control = FRANKA_JOINT_INDICES_TO_CONTROL
        self._right_finger_index = 9
        self._left_finger_index = 10
        self.reset()

        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)

        # Network may have to predict number of repetitions how many times a joint config is to be applied in a row by
        # means of an additional action node:
        # additional_action_node = 0 if self._fixed_nr_action_repetitions else 1

        action_dim = 3
        #Test Run

        self.action_space = spaces.Box(self._action_low, self._action_high)
        # self._action_bound = 1
        # action_high = np.array([self._action_bound] * action_dim)
        # self.action_space = spaces.Box(-action_high, action_high)

        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    ##################################
    # Further added helper functions #
    ##################################

    def reset_logged_train_data(self):
        """
            For logging training progress. Called via PPO agent's callback.
            :return: -
        """
        self.grasps_per_update_interval = 0
        self.grasp_time_steps_needed_per_update_interval = []

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
        return [0, 0.65, 0, -1, 0, 1.5, 0.9, 0, 0, 0.05, 0.05]

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

        return ee_z_axis

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
        # Obtain general measurements:
        self._EE_pos = self._p.getLinkState(self._robot, 11)[0]
        self._goal_pos, _ = p.getBasePositionAndOrientation(self._blockUid)
        right_finger_orn = p.getLinkState(self._robot, self._right_finger_index)[1]
        self._left_finger_pos = p.getLinkState(self._robot, self._left_finger_index)[0]

        # Calculate DIRECTION of vectors EE->Object and Object->Target
        # self._EE_to_goal_vec =self.get_normalized_vector_from_a_to_b(self._EE_pos, self._goal_pos)
        # self._goal_to_target_vec = self.get_normalized_vector_from_a_to_b(self._goal_pos, self._target_pos)

        # Calculate orientation of gripper
        right_finger_orn = self._p.getEulerFromQuaternion(right_finger_orn)
        self._gripper_orn_vec = self.euler_to_vec_gripper_orientation(right_finger_orn)
        self._gripper_orn_vec = self.normalize_vector(self._gripper_orn_vec)

        # Chi-square distribution of grippers orientation in reference to self._gripper_to_goal_vec
        # self._divergence_EEz_EEpos = self.divergence(self._EE_to_goal_vec,self._gripper_orn_vec)

        # Translate variable distance measure computed above into a reward:

        # Reward the reduction of goal distance compared to that of previous time-step
        # self._reward_dist = self._dist_to_obj_secondary - newRewardDistance

        # Reward the absolute negative distance from gripper to goal

        # No continuous distance based reward signal
        # self._reward_dist = 0.0

        ray_start = list(self._p.getLinkState(self._robot, 9)[0])
        ray_start = list(map(add, ray_start, self._gripper_orn_vec * 0.025))
        ray_end = list(self._p.getLinkState(self._robot, 10)[0])
        ray_end = list(map(add, ray_end, self._gripper_orn_vec * 0.025))

        # Add an indicator to see the trajectory of the ray; for debugging only, as the 3D line
        # considerably slows down computation speed
        # self._indicator = self._p.addUserDebugLine(ray_start, ray_end, [0, 0, 0], lineWidth=2)

        # Raytest to check if object is in between the fingers
        self._ray = self._p.rayTest(ray_start, ray_end)

        self.get_contact_info()

    def set_step_counter(self, time_steps):
        self._envStepCounter = time_steps

    # Returns the reward when object is hold with N>=10 and moved towards self._targetPos
    def get_contact_info(self):
        self.contact_info_l = self._p.getContactPoints(self._blockUid, self._robot, linkIndexB=10)
        self.contact_info_r = self._p.getContactPoints(self._blockUid, self._robot, linkIndexB=9)

        if self._p.getContactPoints(6) and self._p.getContactPoints(6)[0][2] != self._blockUid:
            self._contact_with_env = True

        if self.contact_info_l and self.contact_info_l[0][9] >= 10 and self.contact_info_r and \
                self.contact_info_r[0][9] >= 10:
            self._contact_with_force = 1
            # self._reward_dist_hold = distance.euclidean(self._goal_pos, self._target_pos)
        else:
            self._contact_with_force = 0
            # self._reward_dist_hold = 0.0

    def get_reward_measures(self):

        distance_EE_goal_prev = distance.euclidean(self._EE_pos_prev, self._goal_pos)
        distance_EE_goal = distance.euclidean(self._EE_pos, self._goal_pos)
        distance_goal_target_prev = distance.euclidean(self._goal_pos_prev, self._target_pos)
        distance_goal_target = distance.euclidean(self._goal_pos, self._target_pos)

        # Distance Calculations
        self._EE_to_goal = distance_EE_goal
        self._goal_to_target = distance_goal_target

        # Absolute reward distance
        # self._reward_dist = (-1) * distance_EE_goal
        # self._reward_dist_goal_target = (-1) * distance_goal_target
        self._reward_dist = distance_EE_goal_prev - distance_EE_goal
        self._reward_dist_goal_target = distance_goal_target_prev - distance_goal_target

    def get_prepos(self):
        if bool(self._EE_pos_prev):
            self._EE_pos_prev = self._EE_pos
            self._goal_pos_prev = self._goal_pos
        else:
            # self._EE_pos_prev = [0.7015704326197145, 0.2500000000001144, 1.1637235212338237]
            self._EE_pos_prev = self._p.getLinkState(self._robot, 11)[0]
            self._goal_pos_prev = self._p.getBasePositionAndOrientation(self._blockUid)[0]

    ##############################
    # End added helper functions #
    ##############################

    def reset(self):
        self.terminated = 0
        self._current_IK_points = 0
        self._envStepCounter = 0
        self._contact_with_env = False
        self._switch_grip_state = False
        self._initialize_grip_switch_std_t = True
        self._initialize_grip_switch_std_f = False
        self._boundary_selected = False
        self._previously_crashless = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, 0])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, 0,
                   0.000000, 0.000000, 0.0, 1.0)

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.50000, 0,
                   0.000000, 0.000000, 0.0, 1.0)

        self._trayUid = p.loadURDF(os.path.join(self._urdfRoot, "tray/tray.urdf"), 0.540000,
                                   -0.1, 0.63, 0.000000, 0.000000, 1.000000, 0.000000)

        self._trayUid = p.loadURDF(os.path.join(self._urdfRoot, "tray/tray.urdf"), 0.540000,
                                   0.6, 0.63, 0.000000, 0.000000, 1.000000, 0.000000)

        xpos = 0.40 + 0.10 * random.random()
        ypos = 0.5 + 0.2 * random.random()
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        # orn = p.getQuaternionFromEuler([0, 0, ang])
        orn = p.getQuaternionFromEuler([0, 0, (math.pi / 2)-0.2])
        self._blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, 0.64,
                                    orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0.25, 0.8], useFixedBase=1)  # 0.6

        # Set robotic arm to random initial pose
        self.apply_joint_config(self.get_joint_config())

        p.stepSimulation()
        self.obtain_measurements()
        self._observation = self.getExtendedObservation()
        # self.debugReward = 0
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def getExtendedObservation(self):
        """
            Construct an observation that serves as input to the reinforcement learning agent.
            :return: Representation of state of robotic arm and its surrounding
        """
        # Old: self._observation = []
        # self._observation.extend(self._goal_pos)
        observation = []
        observation.extend(self._goal_pos)  # Cartesian position of goal
        observation.extend(self._EE_pos)  # Cartesian position of end-effector
        # self._observation.extend(self._targetPos)             # Cartesian coordinate of the target position (optional
        observation.append(self._contact_with_force)  # Boolean; checks grip state

        return observation

    def step(self, action):

        # action = [action, self._goal_pos[1], 0.9]
        print(action)

        self.get_prepos()

        ik_step_count_limit = 0
        self._EE_to_IK = distance.euclidean(self._EE_pos, action)
        self._current_IK_points += 1
        # Step a simulations given number of times to get old joint configurations closer to the new, desired ones
        # self._residual_threshold = 0.003
        while self._EE_to_IK > self._residual_threshold:

            if self._action_low[0] in action or self._action_low[1] in action or \
                    self._action_low[2] in action or self._action_high[0] in action or \
                    self._action_high[1] in action or self._action_high[2] in action:
                self._boundary_selected = True
                print('Activated')
                # break

            # self._p.removeAllUserDebugItems()
            self._EE_to_IK = distance.euclidean(self._EE_pos, action)
            self._joint_pos = self._p.calculateInverseKinematics(self._robot, 11, action,
                                                                 self._fixed_EE_orn,
                                                                 jointDamping=self._joint_damping,
                                                                 solver=0,
                                                                 maxNumIterations=100,
                                                                 residualThreshold=self._residual_threshold
                                                                 )
            self._joint_pos = list(self._joint_pos)

            if self._ray[0][0] == self._blockUid or self._initialize_grip_switch_std_f:
                self._joint_pos[-1] = -1
                self._joint_pos[-2] = -1
                self._initialize_grip_switch_std_f = True
                # self._initialize_grip_switch = False

            # Set new, desired joint configurations
            self._p.setJointMotorControlArray(bodyUniqueId=self._robot,
                                              # jointIndices=range(self._num_joints),
                                              jointIndices=self._joint_indices_control,
                                              controlMode=self._p.POSITION_CONTROL,
                                              targetPositions=self._joint_pos,
                                              # forces=[50]*self._num_joints)
                                              forces=[200] * len(self._joint_indices_control))  # Old 150

            for i in range(self._action_repetition):
                self._envStepCounter += 1
                ik_step_count_limit += 1
                p.stepSimulation()
                self.obtain_measurements()
                # self._p.addUserDebugLine(self._EE_pos, [self._goal_pos[0],
                #                                         self._goal_pos[1], self._goal_pos[2]+0.008])
                termination = self._termination

                if self._contact_with_env:
                    break
                    # pass

            if self._contact_with_env:
                break
                # pass

            if ik_step_count_limit > 160:  # Old: 160
                break
                # pass

            if self._renders:
                time.sleep(0.05)  # Old value 0.35

        self._observation = self.getExtendedObservation()
        print(self._EE_pos)
        # print(self.contact_line)
        # if self._indicator:
        #     self._p.removeUserDebugItem(self._indicator)
        # self._p.removeUserDebugItem(self._line2)
        # del(self._ray)

        self.get_reward_measures()

        done = self._termination
        # done = done if not self._evalFlag else False  # In case of evaluation run, eval script will handle termination

        reward = self._reward()

        print('The reward is_:{}\n'.format(reward))
        # print(self._current_IK_points)
        # print(self._goal_pos)
        # print(self._contact_with_force)
        # print(self.contact_info_l)
        # print(self.contact_info_r)
        # print(self._ray[0][0])
        # print(self._termination)
        info = {}

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

        if self.terminated or self._current_IK_points >= self._max_IK_points:
            self._observation = self.getExtendedObservation()
            return True

        if self._contact_with_env or self._boundary_selected:
            self._observation = self.getExtendedObservation()
            return True

        # if self._goal_to_target < self._max_error_pos:
        #     # Goal attained
        #     self.terminated = 1
        #     self._observation = self.getExtendedObservation()
        #     # Log training progress
        #     self.grasps_per_update_interval += 1
        #     self.grasp_time_steps_needed_per_update_interval.append(self._envStepCounter)
        #     return True

        return False

    def _reward(self):

        reward = 0

        # reward += self._reward_dist
        reward += self._reward_dist

        reward += self._reward_dist_goal_target

        # Normalize distance reward. 2.05 is max. distance punishment in case of absoulte reward
        # reward /= (self._max_IK_points + 0.5)         # Old: reward /= 2.05

        if self._boundary_selected:
            reward = -1     # Old: (self._max_IK_points + 1)

        if self._contact_with_force and not self._switch_grip_state:
            reward = 0.5        # Old: reward += 1
            self._switch_grip_state = True

        # Find a constant reward value that is appropriate
        # for blockUid being in between fingers ONLY
        # if self._ray[0][0] == self._blockUid:
        #     reward += 0.00001

        if self._goal_to_target < self._max_error_pos:
            goalReward = 1
            timeReward = 740 // self._envStepCounter
            reward += (goalReward + timeReward)

        # elif self._current_IK_points >= self._max_IK_points:
        #     reward -= 1

        if self._contact_with_env and not self._boundary_selected:
            # reward -= (self._max_IK_points + 1)
            # print('Previously Crashless: {}'.format(self._previously_crashless))
            reward = (-0.95 + (self._previously_crashless*0.1))

        self._previously_crashless += 1
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        # _seed = seed
        _step = step

environment = PandaRobotEnv_(renders=True)
# param1 = p.addUserDebugParameter("a1", -1, 1, 1.0)
# param2 = p.addUserDebugParameter("a2", -1, 1, 0.0)
# param3 = p.addUserDebugParameter("a3", -1, 1, 0.8)
# param4 = p.addUserDebugParameter("a4", -0.5, 0.5, 0)
# param5 = p.addUserDebugParameter("a5", -0.5, 0.5, 0)
# param6 = p.addUserDebugParameter("a6", -0.5, 0.5, 0)
# param7 = p.addUserDebugParameter("a7", -0.5, 0.5, 0)
# param8 = p.addUserDebugParameter("a8", -0.5, 0.5, 0)
# param9 = p.addUserDebugParameter("a9", -0.5, 0.5, 0)
# param10 = p.addUserDebugParameter("a10", -0.5, 0.5, 0)
# param11 = p.addUserDebugParameter("a11", -0.5, 0.5, 0)

#
action_nr = 0
debug_step_counter = 0

for i in range(5000000):

    action_nr += 1

    # actions = (p.readUserDebugParameter(param1), p.readUserDebugParameter(param2),
    #            p.readUserDebugParameter(param3), p.readUserDebugParameter(param4),
    #            p.readUserDebugParameter(param5), p.readUserDebugParameter(param6),
    #            p.readUserDebugParameter(param7), p.readUserDebugParameter(param8),
    #            p.readUserDebugParameter(param9))

    # actions = [p.readUserDebugParameter(param1), p.readUserDebugParameter(param2),
    #            p.readUserDebugParameter(param3)
    #            ]

    # rand_x = random.uniform(0.25, 1)
    # rand_y = random.uniform(-0.4, 0.9)
    # rand_z = random.uniform(0.55, 1)

    if action_nr == 1:
        actions = [environment._goal_pos[0],environment._goal_pos[1], environment._goal_pos[2] + 0.15]
        # actions = [0.7015704326197145, 0.2500000000001144, 1.1637235212338237]
    elif action_nr == 2:
        actions = [environment._goal_pos[0], environment._goal_pos[1], environment._goal_pos[2] + 0.001]
    elif action_nr == 3:
        actions = [environment._goal_pos[0], environment._goal_pos[1], environment._goal_pos[2] + 0.35]
    else:
        rand_x = random.uniform(0.25, 0.7)
        rand_y = random.uniform(-0.2, 0.5)
        rand_z = random.uniform(0.7, 0.9)

        actions = [rand_x, rand_y, 0.9]
        # actions = [0.7015704326197145, 0.2500000000001144, 1.1637235212338237]

    #print(environment._p.getNumJoints(environment._robot))
    #print(environment._p.getJointState(environment._robot, 10))
    #print(environment._dev_from_goal_vec_primary)
    #print(environment._gripper_pos)
    #print(environment._reward_goal_dir)
    #print(environment._robot)


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
    #print(environment._ray[0][0]==environment.blockUid)
    #print(environment._gripper_orn_vec)
    #print(environment._envStepCounter)
    #print(environment._ray)
    #print(environment.contact_with_force)
    #print(environment._gripper_to_goal_vec)
    #print(environment._reward_dist_hold)
    #print(environment._dist_to_target_hold_primary)
    #print(environment.terminated)
    #print(environment._reward_dist)
    #print(environment._envStepCounter)
    # environment.step(actions)
    # print(actions)
    # print(environment._goal_pos)
    observation, reward, done, info = environment.step(actions)
    if done:
        environment.reset()
        action_nr = 0








