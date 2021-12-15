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
numClicks = 90

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
model.fit(features, labels, epochs=10)

human_pred = [2,3,5,6,7,8,9,11,12,13,15,16,17,18,19,21,22,23,24,24,24,24,24,24,26,27,28,29,31,32,33,34,35,36,38,39,40,41,42,43,44,46,46,48,48,50,51,52,53,54,55,56,58,58,60,61,62,63,64,65,66,67,68,70,71,72,73,74,75,76,77,78,80,81,82,84,84,86,87,88,90,91,92,93,94,95,96,97,98,99]
simple_pred = [1,2,2,3,3,4,5,5,6,6,7,7,8,8,9,10,10,11,11,12,12,13,13,14,15,15,16,16,17,17,18,18,19,19,20,21,21,22,22,23,23,24,24,25,26,26,27,27,28,28,29,29,30,30,31,31,32,33,33,34,34,35,35,36,36,37,38,38,39,39,40,40,41,41,42,43,43,44,44,45,45,46,46,47,48,48,49,49,50,50]
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
