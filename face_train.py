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

# Find images in a file and then encode images for training recognizer
for root_dir, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        # Find paths to images 
        if file.endswith("png") or file.endswith("jpg"):
            imagePath = os.path.join(root_dir, file)
            imageLabel = os.path.basename(os.path.dirname(imagePath))
           #If image has not been found yet then add to label dictionary
            if imageLabel in label_Dic:
                pass
            else:
                label_Dic[imageLabel] = currId
                currId += 1

            # Encode images for recognizer to train on
            id_ = label_Dic[imageLabel]
            pillow_image = Image.open(imagePath).convert("L")
<<<<<<< HEAD
            # size = (550, 500)
            # final = pillow_image.resize(size, Image.ANTIALIAS)
=======
            #size = (550, 500)
            #final = pillow_image.resize(size, Image.ANTIALIAS)
>>>>>>> de1b8988d468c5aafccc05e276b3ebf7f1739794
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

# Export labels dictionary to file to be used in Attendance.py
with open('labels.pickle', 'wb') as f:
    pickle.dump(label_Dic, f)

#Train recognizer and save learned information in yml file
recognizer.train(x_train, np.array(y_label))
recognizer.save("trainer.yml")    
