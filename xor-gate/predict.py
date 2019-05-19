# test generated model
from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras

"""
XOR gate data
A B Q
0 0 0
0 1 1
1 0 1
1 1 0
"""

model = new_model = keras.models.load_model('xor_model.h5')
q0 = np.argmax(model.predict(np.array([[0,0]])))
q1 = np.argmax(model.predict(np.array([[0,1]])))
q2 = np.argmax(model.predict(np.array([[1,0]])))
q3 = np.argmax(model.predict(np.array([[1,1]])))

print("XOR Predictions table:");
print("-------------");
print("| A | B | Q |");
print("-------------");
print("| 0 | 0 |", q0, "|")
print("| 0 | 1 |", q1, "|")
print("| 1 | 0 |", q2, "|")
print("| 1 | 1 |", q3, "|")
print("-------------");
