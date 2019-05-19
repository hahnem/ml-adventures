# XOR gate train test
from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

"""
XOR gate data
A B Q
0 0 0
0 1 1
1 0 1
1 1 0
"""

# xor data and labels ([[A0,B0], .., [A3,B3]])
xordata = np.array([[0,0],[0,1],[1,0],[1,1]])
# xor results Q0,Q1,Q2,Q3
xorlabels = np.array([0,1,1,0])
# make a lot of training data
features = np.tile(xordata,(100000,1))
labels = np.tile(xorlabels,(100000,))

# setup model dense layers, 2 -> 4 -> 4 -> 2, input and output matching input and output data, and two dense 4 layers seems to do the trick..
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
model = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.nn.relu, input_dim=(2)),
  tf.keras.layers.Dense(4, activation=tf.nn.relu),
  tf.keras.layers.Dense(2, activation='softmax'),
])

# some standard values..
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(features, labels, epochs=3, batch_size=128)

# evaluate model
x_test = np.tile(xordata,(100000,1))
y_test = np.tile(xorlabels,(100000,))
score = model.evaluate(x_test, y_test, batch_size=128)
print("score:", score)

# save model so predict.py can use it
print("Saving model to xor_model.5")
tf.keras.models.save_model(model, 'xor_model.h5')

