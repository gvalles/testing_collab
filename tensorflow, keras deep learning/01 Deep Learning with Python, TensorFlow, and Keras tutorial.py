import tensorflow as tf
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist # 28 x 28 images of hand written digits
# unpack dataset this is easy because is pre pared is hard to do on your own data

(x_train, y_train), (x_test,y_test) = mnist.load_data()

# show one of the images
# x_train is just a multi dimensional array which is all a tensor is
# the image is non color
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()

# we will scale or normalize the data
# the values are now scale between 1 and 0 this is not necessary but helps to improve accuracy

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# there are differnt models built in this the most basic and common is a feedfoward backpropagation
model = tf.keras.models.Sequential()
# we want to flatten the array to less dimensions if you are building a convnet then you probably
# should not do it
# this is our input layer
model.add(tf.keras.layers.Flatten())
# we will make 2 hidden layers
# first parametor is how many units in the layer or neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# output layer 10 because there are 10 classes
# softmax for a probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# the model is done we just need parametor for the model
# loss is the degree of error a NN is not trying to maximize for accuracy but for minimizing loss
# the way you calculate it makes a huge difference
# sparse is for many classes
# for binary you should use binary
# metrics are what you want to measure
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# train the model
model.fit(x_train,y_train, epochs=3)

# we get 0.9764 accuracy in only 3 epocs that is pretty good

# you want your model to generalize well and learn patterns about your data not just memorize your data
# and overfit

# you want the accuracy from the validation and the fitment to be in the same range you dont want a big
# delta otherwise it means you have overfit
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
# we get this for the loss and 0.10263666713070124 0.9684 for the accuracy

# we can save the model
# model.save("hand written recognition.model")

# load the model later on

# new_model = tf.keras.models.load_model("hand written recognition.model")

# lets say we want to predict so all we need to do is
# remember predict always takes a list
# predictions = model.predict([x_test])

# predictions will output an array
# so we can use numpy to translate that into what we can understand
import numpy as np
# to print the prediction of the first element
print(np.argmax(predictions[0]))

# if you want to actually see if is right visually you can
# plt.imshow(x_test[0])
# plt.show()

# in real applications you have to find the model that best works with your data









