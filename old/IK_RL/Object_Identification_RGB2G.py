import numpy as np
from datetime import datetime

now = datetime.now()
TIME_STAMP = now.strftime('_%Y_%d_%m__%H_%M_%S__%f')
PATH = 'Arrays/'
DATA_ID = 'Arrays/' + '_y_dev_' + 'GRAYSCALE' + TIME_STAMP


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


rgb_data = np.load('Arrays/_y_deviation__2021_12_03__23_51_51__445927.npy')

gray_data = rgb2gray(rgb_data)

np.save(DATA_ID, gray_data)

