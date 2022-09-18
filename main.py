import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras import layers
import PIL
from sklearn.model_selection import train_test_split
import model
X = []
y = []
image_size = 300
labels = ["APPLE ROT LEAVES", "HEALTHY LEAVES", "LEAF BLOTCH", "SCAB LEAVES"]
for i in labels:
    folderPath = os.path.join("./APPLE_DISEASE_DATASET", i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
        y.append(i)

X = np.array(X)
Y = np.array(y)
print(np.shape(X))
print(np.shape(Y))

y_new = []
for i in y:
    y_new.append(labels.index(i))
y = y_new
y = tf.keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


X_train = X_train / 255
X_test = X_test / 255



Model=model.Model_Art()
Model.fit(X_train, y_train, epochs=50)
Model.evaluate(X_test, y_test)
Model.summary()
Model.save("finalmodel")

