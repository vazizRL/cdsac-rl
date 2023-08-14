import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from Object_identification import PandaRobotEnv_
import time

now = datetime.now()
TIME_STAMP = now.strftime('_%Y_%d_%m__%H_%M_%S__%f')
PATH = 'Arrays/'
MODEL_ID = 'Arrays/' + '_y_deviation_' + TIME_STAMP
LABEL_ID = 'Arrays/' + '_y_dev_Label' + TIME_STAMP

if not os.path.exists(PATH):
    os.makedirs(PATH)

data_set = []
label_set = []

env = PandaRobotEnv_(renders=False)

start = time.time()
for i in range(1000):
    env.render=False
    data, label = env.generate_data()
    data_set.append(data)
    label_set.append(label)
    env.reset()

np.save(MODEL_ID, data_set)
np.save(LABEL_ID, label_set)

end = time.time()
speed = end - start
print(speed)
print('DONE')
# print(data_set[i])
# plt.figure()
# plt.grid(False)
# plt.imshow(data_set[i])
# plt.show()
# print(label_set[i])



