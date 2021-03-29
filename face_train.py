import os
import cv2
import numpy as np
import pickle
from PIL import Image

DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(DIR, "Images")

faceCascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

currId = 0
label_Dic = {}
x_train = []
y_label = []

# find images in a file and then convert images into number for training recognizer
for root_dir, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root_dir, file)
            label = os.path.basename(os.path.dirname(path)).lower()

            if label in label_Dic:
                pass
            else:
                label_Dic[label] = currId
                currId += 1

            id_ = label_Dic[label]

            pillow_image = Image.open(path).convert("L")
            size = (550, 550)
            final = pillow_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pillow_image, "uint8")
            faces = faceCascade.detectMultiScale(
                image_array,
                scaleFactor=1.5,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id_)

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_Dic, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")
