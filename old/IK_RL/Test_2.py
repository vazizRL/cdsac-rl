import numpy as np
import os
import pybullet as p
import time
import pybullet_data
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

now = datetime.now()
TIME_STAMP = now.strftime('_%Y_%d_%m__%H_%M_%S__%f')
PATH = 'Arrays/'
MODEL_ID = 'Arrays/' + TIME_STAMP

# if not os.path.exists(PATH):
#     os.makedirs(PATH)
#
# array = np.zeros(40000)
#
# np.save(MODEL_ID, array)

IMG_SIZE = 75

rgb = np.load("Arrays/TrainData/img.npy")
label_set = np.load("Arrays/TrainData/label.npy")

# label = np.load('C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/Label_2021_03_03__18_28_24__600325.npy')
# print(label)

test_images = rgb[:5]
test_labels = label_set[:5]
counter = 0
for image in test_images:
    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    plt.figure()
    plt.grid(False)
    plt.imshow(image, cmap='gray')
    plt.title("Label: " + str(label_set[counter]))
    plt.show()
    counter += 1


# for i in range(10):
#     plt.figure()
#     plt.grid(False)
#     plt.imshow(rgb[i], cmap='gray')
#     plt.title("Label: " + str(label_set[i]))
#     plt.show()