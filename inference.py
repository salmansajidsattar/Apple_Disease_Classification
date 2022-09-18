from tensorflow import keras
import cv2
import os
class_names = ['APPLE ROT LEAVES','HEALTHY LEAVES','LEAF BLOTCH','SCAB LEAVES']
img_path="D:\Apple_disease_Classification\APPLE_DISEASE_DATASET\HEALTHY LEAVES/1001.JPG.jpeg"
model = keras.models.load_model('D:\Apple_disease_Classification\\finalmodel')
img = cv2.imread(img_path)
img = cv2.resize(img, (300, 300))
import numpy as np
img=np.array([img/255])
print(class_names[np.argmax(model.predict(img))])