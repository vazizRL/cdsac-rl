import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

PATH_TO_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__)))  # Path to robot models

import math
import gym
from gym import spaces
from datetime import datetime
# from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from pkg_resources import parse_version
import matplotlib.pyplot as plt

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

# Decalre Time Stamp
now = datetime.now()
TIME_STAMP = now.strftime("_%Y_%d_%m_%H_%M_%S___%f")
TIME_STAMP = now.strftime('_%Y_%d_%m__%H_%M_%S__%f')
PATH = 'Arrays/'
MODEL_ID = 'Arrays/' + TIME_STAMP
LABEL_ID = 'Arrays/' + 'Label' + TIME_STAMP
if not os.path.exists(PATH):
    os.makedirs(PATH)

# TARGET POSITION is [0.64, -0.1, 0.7]


class PandaRobotEnv_(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepetition=1,  # 5
                 # actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=7000,  # 7000
                 fixedActionRepetitions=False,
                 distSpecifications=None,
                 # maxDist=0.25,  # Old val. 0.08
                 # maxDeviation=0.25,
                 maxErrorPos=0.1,
                 evalFlag=False,
                 seed=0):

        # Parameter settings
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._isEnableSelfCollision = isEnableSelfCollision
        # self._maxDist = maxDist
        # self._maxDeviation = maxDeviation
        # self._maxErrorPos = maxErrorPos
        self._evalFlag = evalFlag
        self._ray = None
        self._cumulative_reward = 0
        self._action_low =  None
        self._action_high = None
        self._base_mass = 0.5
        self._robo_path = 'C:/Users/XMG/anaconda3/envs/CognitiveRobotics_Robo_Control-master/Lib/site-packages/pybullet_data/franka_panda/panda.urdf'
        self._positional_array = None
        self._positional_sigma = 0.025
        self._positional_clip_factor = 0.025
        # Debugging

        # Rendering
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.7
        self._cam_yaw = 180
        self._cam_pitch = -40

        # Observations & Measurements
        self._observation = []
        self._envStepCounter = 0
        self._avg_reward = []

        # Simulation
        self._seed = seed
        self._urdfRoot = urdfRoot
        self._trayUid = None
        self._blockUid = None
        self._indicator = None
        self._p = p

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self.reset()

        observationDim = len(self.getExtendedObservation())

        self._img_data = None

        observation_high = np.array([largeValObservation] * observationDim)

        # self.action_space = spaces.Box(self._action_low, self._action_high)
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
        pass


    def set_step_counter(self, time_steps):
        self._envStepCounter = time_steps


    def get_reward_measures(self):
        pass

    def create_color(self):
        red = random.uniform(0, 1)
        green = random.uniform(0, 1)
        blue = random.uniform(0, 1)

        return red, green, blue

    def create_sphere(self, basePosition):
        # radius = abs(random.gauss(mu=0.024, sigma=0.001))
        radius = 0.024
        red, green, blue = self.create_color()
        index = 0

        x_y = [basePosition[0], basePosition[1]]

        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)), - (axis -
                                            self._positional_clip_factor), axis + self._positional_clip_factor)
            basePosition[index] = axis
            index += 1

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                            radius=radius,
                                            rgbaColor=[red,green,blue,1],
                                            specularColor=[0.4, .4, 0.4])

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                  radius=radius)

        return self._p.createMultiBody(basePosition=basePosition,
                                       baseOrientation=[1, 1, 1, 1],
                                       baseMass=self._base_mass,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)

    def create_box(self, basePosition):
        # width = abs(random.gauss(mu=0.02, sigma=0.001))
        # length = abs(random.gauss(mu=0.02, sigma=0.001))
        # height = abs(random.gauss(mu=.02, sigma=0.001))
        width = 0.02
        length = 0.02
        height = 0.02

        x_y = [basePosition[0], basePosition[1]]
        index = 0
        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)), - (axis -
                                            self._positional_clip_factor), axis + self._positional_clip_factor)
            basePosition[index] = axis
            index += 1

        red, green, blue = self.create_color()

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[width, length, height],
                                            rgbaColor=[red, green, blue, 1],
                                            specularColor=[0.4, .4, 0.4])

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=[width, length, height])

        self._blockUid = self._p.createMultiBody(basePosition=basePosition,
                                       baseOrientation=[1, 1, 1, 1],
                                       baseMass=self._base_mass,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)
        return self._blockUid

    def create_capsule(self, basePosition):
        # radius = abs(random.gauss(mu=0.02,sigma=0.0001))
        # height = abs(random.gauss(mu=0.04, sigma=0.001))
        radius = 0.02
        height = 0.04

        x_y = [basePosition[0], basePosition[1]]
        index = 0
        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)), - (axis -
                            self._positional_clip_factor), axis + self._positional_clip_factor)
            basePosition[index] = axis
            index += 1

        red, green, blue = self.create_color()

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_CAPSULE,
                                            radius=radius,
                                            length=height,
                                            rgbaColor=[red, green, blue, 1],
                                            specularColor=[0.4, .4, 0.4]
                                            )

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CAPSULE,
                                                  radius=radius,
                                                  height=height)

        orn = [math.pi / 2, 0, random.uniform(0, math.pi/2)]
        orn = self._p.getQuaternionFromEuler(orn)
        return self._p.createMultiBody(basePosition=basePosition,
                                       baseOrientation=orn,
                                       baseMass=self._base_mass,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)

    def create_cylinder(self, basePosition):
        # radius = abs(random.gauss(mu=0.015, sigma=0.0005))
        # height = abs(random.gauss(mu=0.04, sigma=0.001))
        radius = 0.015
        height = 0.045

        index = 0
        x_y = [basePosition[0], basePosition[1]]
        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)), - (axis -
                            self._positional_clip_factor), axis + self._positional_clip_factor)
            basePosition[index] = axis
            index += 1

        red, green, blue = self.create_color()

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                            radius=radius,
                                            length=height,
                                            rgbaColor=[red, green, blue, 1],
                                            specularColor=[0.4, .4, 0.4]
                                            )

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                  radius=radius,
                                                  height=height)

        orn = [math.pi / 2, 0, random.uniform(0, math.pi/2)]
        orn = self._p.getQuaternionFromEuler(orn)
        return self._p.createMultiBody(basePosition=basePosition,
                                       baseOrientation=orn,
                                       baseMass=self._base_mass,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    ##############################
    # End added helper functions #
    ##############################

    def reset(self):
        self.terminated = 0
        self._envStepCounter = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, 0])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, -0.3, 0,
                   0.000000, 0.000000, 0.0, 1.0)
        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.50000, 0,
                   0.000000, 0.000000, 0.0, 1.0)

        # randInd = random.randint(1, 4)
        base_x = 0.7
        base_y = 0.4                        # base_y = 0.2
        base_z = 0.65
        selected = []

        options = {1: self.create_sphere,
                   2: self.create_cylinder,
                   3: self.create_capsule,
                   4: self.create_box
                   }

        self._positional_array = np.zeros(16)
        cube_pos = random.randint(0, 15)
        self._positional_array[cube_pos] = 1
        counter = 0

        for object_type in self._positional_array:
            counter += 1
            if object_type == 0:
                randInd = random.randint(1,3)
            else:
                randInd = 4

            selected.append(randInd)
            options[selected[-1]]([base_x, base_y, base_z])
            base_y += 0.15
            if counter % 4 == 0:
                base_x -= 0.15
                base_y = 0.4                # base_y = 0.2

        p.setGravity(0, 0, -10)

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
        observation = []

        return observation

    def generate_data(self):
        self._p.stepSimulation()
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(  # cameraTargetPosition=base_pos,
            cameraTargetPosition=[0.45, 0.6, 0.6],      # cameraTargetPosition=[0.45, 0.4, 0.6],
            distance=0.6,  # self._rgb_dist
            yaw=-90,  # self._rgb_yaw
            pitch=-90,
            roll=0,
            upAxisIndex=2
        )

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         # aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         aspect=1,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        camera_output_matrix = self._p.getCameraImage(width=100,
                                                    height=100,
                                                     viewMatrix=view_matrix,
                                                    projectionMatrix=proj_matrix,
                                                    renderer=self._p.ER_BULLET_HARDWARE_OPENGL)


        # print(output)
        if self._renders:
            time.sleep(0.05)  # Old value 0.35

        self._observation = self.getExtendedObservation()

        # done = done if not self._evalFlag else False  # In case of evaluation run, eval script will handle termination

        self._img_data = np.array(self.rgb2gray(camera_output_matrix[2]))
        self._img_data /= 255



        return camera_output_matrix[2], self._positional_array

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

    def _reward(self):

        reward = 0

        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset



environment = PandaRobotEnv_(renders=True)

time.sleep(0.2)
environment.generate_data()
action_nr = 0
debug_step_counter = 0


plt.figure()
plt.grid(False)
plt.imshow(environment._img_data, cmap='gray')
plt.show()



