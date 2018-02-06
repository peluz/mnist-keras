from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.datasets import mnist
from keras.initializers import Orthogonal
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import utils


import numpy as np


# reading the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the values
x_train /= 255
x_test /= 255

# Convert labels to one-hot vectors
y_train = utils.to_categorical(y_train, num_classes=10)
y_test = utils.to_categorical(y_test, num_classes=10)

# Initialize weights with random orthogonal matrix
initializer = Orthogonal()

# Build the net
model = Sequential()

# First conv layer with relu activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1),
                 kernel_initializer=initializer,
                 padding='same'))

# Batch Normalization layer
model.add(BatchNormalization())

# Second conv layer with relu activation
model.add(Conv2D(64, (3, 3), activation='relu',
                 kernel_initializer=initializer,
                 padding='same'))

# Batch Normalization layer
model.add(BatchNormalization())

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Droupout layer
model.add(Dropout(0.5))

# First fully connected layer plus relu activation
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer=initializer))

# Batch Normalization layer
model.add(BatchNormalization())

# Dropout Layer
model.add(Dropout(0.5))

# Softmax layer for classification
model.add(Dense(10, activation='softmax', kernel_initializer=initializer))

# compile the model with adam optmizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model and save it every epoch
saver = ModelCheckpoint('mnistClassifier.h5')
# learning rate decays if learning plateaus
decay = ReduceLROnPlateau(factor=0.2, patience=5)
# stop training if validation loss stops improving
stop = EarlyStopping(min_delta=0.001, patience=5)
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1,
          callbacks=[saver, decay, stop])
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
