# randomizerTrain.py
# Author: Valentino Berta
# Date: 12/11/2021

# Explanation:
#   Linear regression is an algorithm which returns values 0 or 1
#   i'm going to use the current time as an input
#   being the current timestamp right now the start each time the autoclicker is turned on
#   and the algorithm is going to return if it should click or not depending on the time

# ----------

import numpy as np
import pandas as pd
import random
import tensorflow as tf

from tensorflow.keras import layers

np.set_printoptions(precision=3, suppress=True)

click_train = pd.read_csv('./data.csv', names=['clicks', 'type'])

# clicks counting towards array
numClicks = 50

# array
clicks = []

# Split |
for i in range(len(click_train.values)):
    click_train.values[i][0] = (click_train.values[i][0].split('|'))
    # remove last one as it has an extra bar
    del click_train.values[i][0][-1]
    # convert to float
    click_train.values[i][0] = [float(x) for x in click_train.values[i][0]]

    # Make all arrays same length
    if len(click_train.values[i][0]) < numClicks:
        for j in range(numClicks - len(click_train.values[i][0])):
            click_train.values[i][0].append(0)
    elif len(click_train.values[i][0]) > numClicks:
        del click_train.values[i][0][numClicks:]

    score=0
    if click_train.values[i][1] == 'human':
        score=100
    else:
        score=0

    clicks.append([
        click_train.values[i][0],
        score
    ])

random.shuffle(clicks)

features = np.array([x[0] for x in clicks])
labels = np.array([x[1] for x in clicks])

model = tf.keras.Sequential([
    layers.Dense(numClicks),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(features, labels, epochs=3)

human_pred = [0.0, 0.122, 0.194, 0.298, 0.403, 0.498, 0.601, 0.701, 0.811, 0.918, 1.026, 1.121, 1.237, 1.352, 1.463, 1.578, 1.691, 1.909, 1.99, 2.059, 2.312, 2.418, 2.52, 2.622, 2.743, 2.859, 2.956, 3.042, 3.186, 3.454, 3.543, 3.627, 3.721, 3.831, 3.925, 4.03, 4.14, 4.248, 4.353, 4.465, 4.575, 4.689, 4.789, 4.914, 5.031, 5.133, 5.245, 5.377, 5.487, 5.597]
simple_pred = [0.0, 0.041, 0.088, 0.135, 0.181, 0.227, 0.274, 0.321, 0.369, 0.414, 0.461, 0.508, 0.554, 0.6, 0.647, 0.694, 0.742, 0.79, 0.837, 0.884, 0.93, 0.978, 1.024, 1.073, 1.119, 1.166, 1.212, 1.259, 1.305, 1.352, 1.399, 1.447, 1.494, 1.541, 1.588, 1.635, 1.681, 1.728, 1.775, 1.82, 1.867, 1.914, 1.962, 2.009, 2.057, 2.104, 2.152, 2.199, 2.247, 2.294]
#simple_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

pred = np.asarray(human_pred[0:numClicks]).reshape(1, -1)

# predict human
print(
    'human',
    model.predict(
        pred
    )
)

pred = np.asarray(simple_pred[0:numClicks]).reshape(1, -1)

# predict simple
print(
    'simple',
    model.predict(
        pred
    )
)

# -------------

# human_data = np.asarray(human).astype(np.float32)
# simple_data = np.asarray(simple).astype(np.float32)

# # Get min number of samples
# min_samples = min(len(human_data), len(simple_data))
# training_size = int(min_samples* 0.8)

# x_train = human_data[:training_size]
# x_test = human_data[training_size:]

# y_train = simple_data[:training_size]
# y_test = simple_data[training_size:]

# model = tf.keras.Sequential([
#     layers.Dense(numClicks)
# ])
# model.compile(optimizer='adam',loss='mse')
# model.fit(x_train, y_train, epochs = 5)
