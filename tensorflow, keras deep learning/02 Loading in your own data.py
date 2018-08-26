# we will build a dog or cat classifier with NNs

import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


datadir = "/Users/guillermovalleschavez/Downloads/kagglecatsanddogs_3367a/PetImages"
categories = ["Dog", "Cat"]

# for category in categories:
#     # gets to us to the path to cats or dogs dir
#     path = os.path.join(datadir, category)
#     # iterate through images
#     for img in os.listdir(path):
#         # convert images to an array convert to grascale to reduce image size and thus training time
#         # color is not essential in this case if it was then it would be good to leave it
#         img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
#         # just to show we are on the right track
#     #     plt.imshow(img_array, cmap="gray")
#     #     plt.show()
#         break
#     break

# we will rezise all the images because they come in different shapes the bigger the size
# the less pixeled it looks
img_size = 50

# new_array = cv2.resize(img_array, (img_size, img_size))
# plt.imshow(new_array, cmap="gray")
# plt.show()

training_data = []

# we will convert the output where 0 means dog and 1 cat doesnt really matter the order
# we are using the index of the list that we are creating to do so
def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        clas_num = categories.index(category)
        for img in os.listdir(path):
            # handle broken images
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, clas_num])
            except Exception as e:
                pass


# for a binary classification task like this one you want to have your data balanced 50 50
create_training_data()
# print(len(training_data))

# the next thing is to shuffle the data so that a random dog and image is chosen

random.shuffle(training_data)

print(len(training_data))

# we will pack the shuffled data before putting it in the model

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
# we cant pass a list to the NN
# we need to reshape to -1
# one is because of gray scale if it is color then is 3
X = np.array(X).reshape(-1, img_size, img_size, 1)

# last thing is to save your reshaped arrays with pick or np save
pickle_out = open("X.pickle", "wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# to open
# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)














