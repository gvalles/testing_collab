import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import numpy as np


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in2 = open("y.pickle", "rb")
y = pickle.load(pickle_in2)

# before feeding the data into the model we need to normalize it
# in this case scale it
# divide by 255 the max number of pixels
# print(np.max(X))

X = X/255

model = Sequential()
# first is the no. filters, window size and the input shape
# the first element is not good input
model.add(Conv2D(64, (3,3), input_shape= X.shape[1:]))
# taken from text version
# model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))

# dont need the input shape again so far we have 2 hidden layers
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))

# flatten data becase dense wants 1d data while right now we have 2d
model.add(Flatten())
model.add(Dense(64, activation="relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])


model.fit(X, y, batch_size=32, validation_split=0.1, epochs=3)

model.save("cats vs dogs.model")

# it is taking a while to load because i used a bigger img size

# we get 0.7794 acc and val_acc: 0.7743 in just 3 epochs that is pretty good













