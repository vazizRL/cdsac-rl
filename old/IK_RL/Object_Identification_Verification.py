import tensorflow as tf
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt

# ADR Parameters
EPOCHS = 2
BATCH_SIZE = 5
TRAIN_TEST_RATIO = 0.9
NOISE_VAR = 0               # Max 60
CHECKING_INSTANCES = 2

PATH_TO_MODEL = 'Arrays/Model__2__SII5_2021_06_03_9998'
img_data = np.load("C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/GRAYSCALE_2021_04_03__17_31_37__429068.npy")
label_set = np.load('C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/Label_2021_04_03__12_59_22__278994.npy')

# train_data, train_labels = img_data[:18000], label_set[:18000]
test_data, test_labels = img_data[18000:], label_set[18000:]
del img_data

# train_data = np.expand_dims(train_data, axis=3)
test_data = np.expand_dims(test_data, axis=3)

# train_data /= 255
test_data /= 255

def add_noise(image_i):
    batch, row, col, ch = image_i.shape
    mean = 0
    var = NOISE_VAR / (255*255)
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (batch, row, col, ch))
    # gauss = gauss.reshape(batch, row, col, ch)
    noisy = image_i + gauss
    return noisy

test_data_1 = add_noise(test_data)
test_data_2 = add_noise(test_data)

model = models.load_model(PATH_TO_MODEL)
prediction_1 = 0
prediction_2 = 1
hit_buffer = []
# prediction_1 = model.predict(test_data_1, batch_size=len(test_data_1))
# prediction_2 = model.predict(test_data_2, batch_size=len(test_data_2))

for i in range(len(test_data)):
    test_data_1_i = np.expand_dims(test_data_1[i], axis=0)
    test_data_2_i = np.expand_dims(test_data_2[i], axis=0)

    p1 = model.predict(test_data_1_i)
    p2 = model.predict(test_data_2_i)

    prediction_1_conf = np.max(p1)
    prediction_2_conf = np.max(p2)

    prediction_1 = np.argmax(p1)
    prediction_2 = np.argmax(p2)

    if prediction_1_conf > prediction_2_conf:
        prediction = prediction_1
    else:
        prediction = prediction_2
    # while prediction_1 != prediction_2:
    #     prediction_1 = np.argmax(model.predict(test_data_1_i))
    #     prediction_2 = np.argmax(model.predict(test_data_2_i))

    if np.argmax(test_labels[i]) == prediction:
        hit_buffer.append(1)
    else:
        hit_buffer.append(0)

accuracy_adr = sum(hit_buffer)/len(hit_buffer)
print('The ADR accuracy is: {}'.format(accuracy_adr))
    # prediction_1 = np.argmax(prediction_1)
    # print('Prediction is: {} \n Actual is: {} \n'.format(prediction_1, np.argmax(test_labels[i])))
    # prediction_1_activated = np.append(prediction_activated, (np.argmax(prediction_1[i])))
    # prediction_1_activated = np.argmax(prediction_1[i])

# print(prediction_1[100])
# print(int(prediction_1_activated[100]))
# print(np.argmax(test_labels[100]))


# for i in range(len(sub_data)):
#     # for i in range(len(prediction)):
#     #     prediction[i] = int(prediction[i])
#     plt.figure()
#     plt.grid(False)
#     plt.imshow(sub_data[i], cmap='gray')
#     plt.xlabel('Prediction: {}'.format(np.argmax(prediction[i])))
#     plt.title('Actual: {}.'.format(sub_label[i]))
#     plt.show()








