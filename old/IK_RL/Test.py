import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PATH = 'Arrays/Model__2__SII5_2021_06_03__20_40_59'
# model = tf.keras.models.load_model(PATH)

test_data = np.load('Arrays/_y_dev_GRAYSCALE_2021_12_03__23_54_20__230988.npy')
test_label = np.load('Arrays/_y_dev_Label_2021_12_03__23_51_51__445927.npy')

# test_data = np.load('Arrays/GRAYSCALE_2021_04_03__17_31_37__429068.npy')
# test_label = np.load('Arrays/Label_2021_04_03__12_59_22__278994.npy')
test_data = np.expand_dims(test_data, axis=3)


# test_data /= 255

test_batch = test_data[10:15]
label_batch = test_label[10:15]
noisy = []
def add_noise(image_i):
    batch, row, col, ch = image_i.shape
    mean = 0
    var = 0 / (255*255)
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (batch, row, col, ch))
    # gauss = gauss.reshape(batch, row, col, ch)
    noisy = image_i + gauss
    return noisy

noisy_image = add_noise(test_batch)
noisy_image_2 = add_noise(test_batch)
count_ini = 0
for image in noisy_image:
    plt.figure()
    plt.grid(False)
    plt.imshow(image, cmap='gray')
    plt.xlabel('Label: {}'.format(label_batch[count_ini]))
    plt.show()
    count_ini += 1

# count_ini = 0
# for image in noisy_image_2:
#     plt.figure('Noisy_Image_2')
#     plt.grid(False)
#     plt.imshow(image, cmap='gray')
#     plt.xlabel('Prediction: {}'.format(label_batch[count_ini]))
#     plt.show()
#     count_ini += 1

# plt.figure()
# plt.grid(False)
# plt.imshow(noisy_image, cmap='gray')
# plt.xlabel('Prediction: {}'.format(test_label[500]))
# plt.title('Noisy Image')
# plt.show()


# for i in [100, 200, 300, 400, 500, 600, 1000, 1300, 1500, 1700, 1800, 1999]:
#     plt.figure(i)
#     plt.grid(False)
#     plt.imshow(test_data[i], cmap='gray')
#     plt.xlabel('Prediction: {}'.format(test_label[i]))
#     plt.show()

