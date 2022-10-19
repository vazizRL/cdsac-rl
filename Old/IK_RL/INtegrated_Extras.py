PATH_TO_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__)))  # Path to robot models
PATH_TO_ADR_CNN = 'C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/Model__2__SII5_2021_06_03_9998'
.
.
.
# Bullet Object Constructors API
    def create_color(self):
        red = random.uniform(0, 1)
        green = random.uniform(0, 1)
        blue = random.uniform(0, 1)

        return red, green, blue

    def create_sphere(self, basePosition, baseMass):
        # radius = abs(random.gauss(mu=0.024, sigma=0.001))
        radius = 0.024
        red, green, blue = self.create_color()
        index = 0

        x_y = [basePosition[0], basePosition[1]]

        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)),
                           - (axis - self._positional_clip_factor),
                           axis + self._positional_clip_factor)
            basePosition[index] = axis
            index += 1

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                            radius=radius,
                                            rgbaColor=[red, green, blue, 1],
                                            specularColor=[0.4, .4, 0.4])

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                  radius=radius)

        return self._p.createMultiBody(basePosition=basePosition,
                                       baseOrientation=[1, 1, 1, 1],
                                       baseMass=baseMass,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)

    def create_wall(self, basePosition, dimensions, baseMass):
        red = 255 / 255
        green = 51 / 255
        blue = 51 /255
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=dimensions,
                                            rgbaColor=[red, green, blue, 0.85],
                                            specularColor=[0, 0, 0])

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=dimensions)
        # orn = self._p.getQuaternionFromEuler([math.pi/2, 0, 0])
        # print('Orientation is: {}'.format(orn))
        self._wall_Uid =  self._p.createMultiBody(basePosition=basePosition,
                                baseOrientation=[0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
                                baseMass=baseMass,
                                baseVisualShapeIndex=visualShapeId,
                                baseCollisionShapeIndex=collisionShapeId
                                )

    def create_box(self, basePosition, baseMass):
        # width = abs(random.gauss(mu=0.02, sigma=0.001))
        # length = abs(random.gauss(mu=0.02, sigma=0.001))
        # height = abs(random.gauss(mu=.02, sigma=0.001))
        width = 0.02
        length = 0.02
        height = 0.02

        x_y = [basePosition[0], basePosition[1]]
        index = 0
        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)),
                                           - (axis - self._positional_clip_factor),
                                           axis + self._positional_clip_factor)


            basePosition[index] = axis
            index += 1

        red, green, blue = self.create_color()
        # red, green, blue = 220/255, 20/255, 60/255

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[width, length, height],
                                            rgbaColor=[red, green, blue, 1],
                                            specularColor=[0.4, .4, 0.4])

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=[width, length, height])

        # self._blockUid = self._p.createMultiBody(basePosition=basePosition,
        #                         baseOrientation=[1, 1, 1, 1],
        #                         baseMass=baseMass,
        #                         baseCollisionShapeIndex=collisionShapeId,
        #                         baseVisualShapeIndex=visualShapeId)
        return self._p.createMultiBody(basePosition=basePosition,
                                baseOrientation=[1, 1, 1, 1],
                                baseMass=baseMass,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId)
        # return self._blockUid

    def create_capsule(self, basePosition, baseMass):
        # radius = abs(random.gauss(mu=0.02,sigma=0.0001))
        # height = abs(random.gauss(mu=0.04, sigma=0.001))
        radius = 0.02
        height = 0.04

        x_y = [basePosition[0], basePosition[1]]
        index = 0
        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)),
                                           - (axis - self._positional_clip_factor),
                                           axis + self._positional_clip_factor)
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

        orn = [math.pi / 2, 0, random.uniform(0, math.pi / 2)]
        orn = self._p.getQuaternionFromEuler(orn)
        return self._p.createMultiBody(basePosition=basePosition,
                                       baseOrientation=orn,
                                       baseMass=baseMass,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)

    def create_cylinder(self, basePosition, baseMass):
        # radius = abs(random.gauss(mu=0.015, sigma=0.0005))
        # height = abs(random.gauss(mu=0.04, sigma=0.001))
        radius = 0.015
        height = 0.045

        index = 0
        x_y = [basePosition[0], basePosition[1]]
        for axis in x_y:
            axis = np.clip(abs(random.gauss(mu=axis, sigma=self._positional_sigma)),
                                           - (axis - self._positional_clip_factor),
                                           axis + self._positional_clip_factor)

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

        orn = [math.pi / 2, 0, random.uniform(0, math.pi / 2)]
        orn = self._p.getQuaternionFromEuler(orn)
        return self._p.createMultiBody(basePosition=basePosition,
                                       baseOrientation=orn,
                                       baseMass=baseMass,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def take_image(self, camera_target_pos=(0.45, 0.6, 0.6), distance_cam=0.6,
                   yaw_pitch_roll=(-90, -90, 0), up_axis=2, res=(100,100)):

        yaw, pitch, roll = yaw_pitch_roll
        width, height = res

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(  # cameraTargetPosition=base_pos,
            cameraTargetPosition=camera_target_pos,      # cameraTargetPosition=[0.45, 0.4, 0.6],
            distance=distance_cam,  # self._rgb_dist
            yaw=yaw,  # self._rgb_yaw
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis
        )

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         # aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         aspect=1,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        camera_output_matrix = self._p.getCameraImage(width=width,
                                                    height=height,
                                                    viewMatrix=view_matrix,
                                                    projectionMatrix=proj_matrix,
                                                    # renderer=self._p.ER_BULLET_HARDWARE_OPENGL
                                                    renderer=self._p.ER_TINY_RENDERER)

        camera_output_matrix_gray = np.array(self.rgb2gray(camera_output_matrix[2]))
        camera_output_matrix_gray = np.expand_dims(camera_output_matrix_gray, axis=0)
        camera_output_matrix_gray = np.expand_dims(camera_output_matrix_gray, axis=3)
        camera_output_matrix_gray = camera_output_matrix_gray / 255
        return camera_output_matrix_gray
.
.
.
    # Partially Overwritten
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

        self.create_wall([0.4, 0.16, 0.775], [0.4, 0.15, 0.05], 1000)

        self._trayUid = p.loadURDF(os.path.join(self._urdfRoot, "tray/tray.urdf"), 0.380000,
                                   -0.25, 0.63, 0.000000, 0.000000, 1.000000, 0.000000)

        base_x = 0.7
        base_y = 0.4  # Old: 0.2
        base_z = 0.65
        selected = []

        options = {1: self.create_sphere,
                   2: self.create_cylinder,
                   3: self.create_capsule,
                   4: self.create_box
                   }

        self._positional_array = np.zeros(16)
        cube_pos = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14])
        # cube_pos = 7
        self._positional_array[cube_pos] = 1
        counter = 0

        for object_type in self._positional_array:
            counter += 1
            if object_type == 0:
                randInd = random.randint(1, 3)
            else:
                randInd = 4

            selected.append(options[randInd]([base_x, base_y, base_z], self._base_mass))
            base_y += 0.15
            if counter % 4 == 0:
                base_x -= 0.15
                base_y = 0.4  # Old: 0.2

        # Capture Object distribution
        gray_scale = self.take_image()
        np.save('Arrays/D3_Test', gray_scale)
        prediction_value = np.argmax(cnn_model.predict(gray_scale))
        print('The prediction index is: {}'.format(prediction_value))
        self._blockUid = selected[prediction_value]

        p.setGravity(0, 0, -10)
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0.5, 0.7], useFixedBase=1)

        # Set robotic arm to random initial pose
        self.apply_joint_config(self.get_joint_config())

        p.stepSimulation()

        self.obtain_measurements()
        self._observation = self.getExtendedObservation()
        # self.debugReward = 0
        return np.array(self._observation)