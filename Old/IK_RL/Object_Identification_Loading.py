import tensorflow as tf
from tensorflow import keras #Keras = High-level API
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from datetime import datetime
import os


# Learning Parameters
number_epochs = 5

# Bookkeeping Functions
now = datetime.now()
TIME_STAMP = now.strftime('_%Y_%d_%m__%H_%M_%S')
PATH = 'Arrays/'
MODEL_ID = 'Arrays/' + 'Model__2__SII' + str(number_epochs) + TIME_STAMP
LOADING_ID = 'Arrays/Model__2__25_2021_06_03__19_44_20'


if not os.path.exists(PATH):
    os.makedirs(PATH)

data_set = np.load("C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/GRAYSCALE_2021_04_03__17_31_37__429068.npy")
label_set = np.load('C:/Users/XMG/Desktop/Master/Masterarbeit/IK_RL/Arrays/Label_2021_04_03__12_59_22__278994.npy')

train_data, train_labels = data_set[:18000], label_set[:18000]
test_data, test_labels = data_set[18000:], label_set[18000:]

# Data Preprocessing
train_data = np.expand_dims(train_data, axis=3)
test_data = np.expand_dims(test_data, axis=3)
print(train_data.shape)

train_data /= 255
test_data /= 255

model = keras.models.load_model(LOADING_ID)

# Train the Model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


# Train the model on the data
history = model.fit(train_data, train_labels, epochs=number_epochs,
                    validation_data=(test_data, test_labels))

# Save Model after Training
model.save(MODEL_ID)

