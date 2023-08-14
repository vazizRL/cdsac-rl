import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from datetime import datetime
import os
import time


# Bookkeeping Functions
now = datetime.now()
TIME_STAMP = now.strftime('_%Y_%d_%m__%H_%M_%S')
PATH = 'Arrays\ADR_MODEL' + TIME_STAMP
MODEL_ID = PATH + '/' + 'ARD_MODEL'

LOADING = False
LOADING_PATH = ''

# Enable TensorBoard Logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=PATH, profile_batch=5,)
# writer = tf.summary.FileWriter(PATH)

# ADR Parameters
TRAINING_STEPS = 200000  # 100000
HYPER_BATCH_SIZE = 1000
TRAINING_PACKAGES = int(TRAINING_STEPS / HYPER_BATCH_SIZE)
TRAIN_TEST_RATIO = 0.9
NOISE_VAR = 0               # Max 60
NOISE_VAR_STEP_SIZE = 3
LEARNING_RATE = 0.0001
PERFORMANCE_MAX = 0
PERFORMANCE_MIN = 0
PERFORMANCE_DIFFERENCE = 0
COUNTER_MIN_PERFORMANCE = 0

if not os.path.exists(PATH):
    os.makedirs(PATH)

data_set = np.load("C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/GRAYSCALE_2021_04_03__17_31_37__429068.npy")
label_set = np.load('C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/Label_2021_04_03__12_59_22__278994.npy')

train_data, train_labels = data_set[:18000], label_set[:18000]
test_data, test_labels = data_set[18000:], label_set[18000:]

noisy = []

# Data Preprocessing
train_data = np.expand_dims(train_data, axis=3)
test_data = np.expand_dims(test_data, axis=3)
print(train_data.shape)

train_data /= 255
test_data /= 255

### Add Helper Functions ###
# Create Noise Function
def add_noise(image_i):
    batch, row, col, ch = image_i.shape
    mean = 0
    var = NOISE_VAR/(255*255)   # 130
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (batch, row, col, ch))
    # gauss = gauss.reshape(batch, row, col, ch)
    noisy = image_i + gauss
    return noisy

def get_step_increase_factor(difference):
    increase_param = difference // 0.05
    return increase_param

### End Helper Functions ###

# Create Model Version 2 or Load the Model
if not LOADING:
    # Define the CNN Model
    model = models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(100, 100, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation='softmax'))

    # opt =tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
else:
    model = tf.keras.models.load_model(LOADING_PATH)

    # Train the Model
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

counter_epochs = 0
# ### Test ###
# rand_idx = np.random.choice(18000, HYPER_BATCH_SIZE, replace=False)
# train_batch = train_data[rand_idx]
# train_batch = add_noise(train_batch)
# label_batch = train_labels[rand_idx]
# plt.figure()
# plt.grid(False)
# plt.imshow(train_batch[20], cmap='gray')
# plt.xlabel('Prediction: {}'.format(label_batch[20]))
# plt.show()
#
# plt.figure()
# plt.grid(False)
# plt.imshow(train_batch[2], cmap='gray')
# plt.xlabel('Prediction: {}'.format(label_batch[2]))
# plt.show()

for i in range(TRAINING_PACKAGES):
    print('PACKAGE NUMBER: {}'.format(i))
    rand_idx = np.random.choice(18000, HYPER_BATCH_SIZE, replace=False)
    train_batch = train_data[rand_idx]
    train_batch = add_noise(train_batch)
    label_batch = train_labels[rand_idx]



    rand_idx_test = np.random.choice(2000, int(HYPER_BATCH_SIZE*0.1), replace=False)
    test_batch = test_data[rand_idx_test]
    test_label_batch = train_labels[rand_idx_test]
    # Train the model on the data
    tensorboard_callback.set_model(model)
    history = model.fit(train_batch, label_batch, epochs=counter_epochs+1,
                        validation_data=(test_batch, test_label_batch),
                        initial_epoch=counter_epochs,
                        callbacks=[tensorboard_callback])
    counter_epochs +=1

    # if history.history['acc'] < PERFORMANCE_MIN:
    #     PERFORMANCE_MIN = history.history['acc']

    if history.history['acc'][0] > PERFORMANCE_MAX:
        PERFORMANCE_MAX = history.history['acc'][0]

    PERFORMANCE_DIFFERENCE = PERFORMANCE_MAX - PERFORMANCE_MIN
    if PERFORMANCE_DIFFERENCE > 0.05:
        COUNTER_MIN_PERFORMANCE += 1
        NOISE_VAR_INCREASE_FACTOR = get_step_increase_factor(PERFORMANCE_DIFFERENCE)
        NOISE_VAR += NOISE_VAR_STEP_SIZE * NOISE_VAR_INCREASE_FACTOR

        # PERFORMANCE_MIN = COUNTER_MIN_PERFORMANCE * 0.05
        PERFORMANCE_MIN = PERFORMANCE_MAX
# Save Model after Training
model.save(MODEL_ID)

print('''The ADR Parameters @ end of training: \n
            1) PERFORMANCE_MIN: {} \n
            2) COUTNER_MIN_PERFORMANCE: {} \n
            3) PERFORMANCE_MAX: {} \n
            4) NOISE_VAR: {}'''.format(PERFORMANCE_MIN, COUNTER_MIN_PERFORMANCE,
                                       PERFORMANCE_MAX, NOISE_VAR))
