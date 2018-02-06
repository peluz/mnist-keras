#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.datasets import mnist
from keras import utils
from random import randrange
import numpy as np
import matplotlib.pyplot as plt

# read the test dataset
_, (x_test, y_test) = mnist.load_data()

# reshape the data
x_test = np.reshape(x_test, (-1, 28, 28, 1))
x_test = x_test.astype('float32')

# Normalize the values
x_test /= 255

# Convert labels to one-hot vectors
y_test = utils.to_categorical(y_test, num_classes=10)

# load trained model
classifier = load_model("mnistClassifier.h5")

# plot ten random images from test set with its classification
fig = plt.figure()
columns = 5
rows = 2
test_size = x_test.shape[0]
for i in range(10):
    image = x_test[randrange(test_size)]
    prediction = classifier.predict(image)
    ax = plt.subplot(rows, columns, i)
    ax.set_title(prediction)
    ax.imshow(image)
plt.show()