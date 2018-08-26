# in this video we will cover how to analyze and optimze our models using tensorboard
# helpful to look at accuracy, loss etc
# we will do a keras callback with tensorboard
# there are other ways to do
# with model checkpoint you can save the model best on parametors
# like accuracy and loss

from tensorflow.keras.callbacks import TensorBoard
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import numpy as np
import time

# save model with time
NAME = "cats-vs-dogs-cnn64x2".format(int(time.time()))

# now we can just pass it into the fitment
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

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


model.fit(X, y, batch_size=32, validation_split=0.1, epochs=3, callbacks=[tensorboard])

# tensorboard --logdir /Users/guillermovalleschavez/PycharmProjects/intermediate_python/tensorflow, keras deep learning/logs
# have to do pwd because the path is messed up and then run the command

# you want to look at the graph and you can increase the epochs until the accuracy starts to go down
# and the loss up
# that is when you start to overfit
# look at the validation loss and validation accuracy


# one really cool thing that i can do is to process the data on my computer and save the prepared
# data into a pickle or numpy save train the model in collab with their free gpus
























