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

# plot random images from test set with its classification as
# long as user wants
columns = 5
rows = 5
test_size = x_test.shape[0]
a = 'a'

while(a != 'n'):
    fig = plt.figure(figsize=(8, 8))
    for i in range(25):
        index = randrange(test_size)
        image = np.expand_dims(x_test[index], axis=0)
        prediction = classifier.predict(image)
        ax = plt.subplot(rows, columns, i + 1)
        ax.set_title("Prediction:{}{} Truth:{}".format(
            np.argmax(prediction), '\n', np.argmax(y_test[index])))
        image = np.reshape(image, (28, 28))
        ax.imshow(image)
    plt.tight_layout()
    plt.show()

    a = 'a'
    while(a != 'y' and a != 'n'):
        a = raw_input(
            "Do you want to see more image samples?(y/n).")

# Overall model accuracy
score = classifier.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
