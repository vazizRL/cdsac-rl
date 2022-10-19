import tensorflow as tf
from tensorflow.keras import models
import os
from datetime import datetime
from Object_Identification_ADR import PandaRobotEnv_
import numpy as np


# Bookkeeping Functions
now = datetime.now()
TIME_STAMP = now.strftime('_%Y_%d_%m__%H_%M_%S')
PATH = 'Arrays/'
MODEL_ID = 'Arrays/' + 'Model_ADR' + TIME_STAMP

if not os.path.exists(PATH):
    os.makedirs(PATH)

# Main Parameters
TRAINING_STEPS = 36000
HYPER_BATCH_SIZE = 1000
TRAINING_PACKAGES = int(TRAINING_STEPS / HYPER_BATCH_SIZE)
TRAIN_TEST_RATIO = 0.9

LOADING = False
LEARNING_RATE = 0.001
LOADING_PATH = ''

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

env = PandaRobotEnv_(renders=False)

for i in range(TRAINING_PACKAGES):
    rgb_data = []
    label_data = []
    for j in range(HYPER_BATCH_SIZE):
        rgb_value, labels = env.generate_data()
        rgb_data.append(rgb_value)
        label_data.append(labels)
        env.reset()

    rgb_data = np.array(rgb_data)
    label_data = np.array(label_data)
    # Image Preprocessing
    gray_scale_images = env.rgb2gray(rgb_data)
    gray_scale_images = env.add_noise(gray_scale_images)
    gray_scale_images /= 255
    gray_scale_images = np.expand_dims(gray_scale_images, axis=3)

    # Divide in Train and Test Batches
    train_data = gray_scale_images[:int(len(gray_scale_images)*TRAIN_TEST_RATIO)]
    test_data = gray_scale_images[int(len(gray_scale_images)*TRAIN_TEST_RATIO):]

    train_labels = label_data[:int(len(gray_scale_images)*TRAIN_TEST_RATIO)]
    test_labels = label_data[int(len(gray_scale_images)*TRAIN_TEST_RATIO):]

    # Train the Model on the generated Batch
    history = model.fit(train_data, train_labels, epochs=1,
                        validation_data=(test_data, test_labels))


# Save Model after Training
model.save(MODEL_ID)



