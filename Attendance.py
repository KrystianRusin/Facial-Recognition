import cv2
import pickle
import xlwt
import csv
from datetime import datetime

# wb = Workbook()
row = 0
col = 0
nameList = []
f = open('Attendance.csv', "w+")
f.writelines('Name , Date')


faceCascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open('labels.pickle', 'rb') as f:
    original_labels = pickle.load(f)
    labels = {v:k for k,v in original_labels.items()}

video_capture = cv2.VideoCapture(0)

def markAttendance(name):
        with open('Attendance.csv', 'r+') as f:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

while True:
    returnCode, camera = video_capture.read()

    grayScale = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        grayScale,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        roi_gray = grayScale[y:y+h, x:x+w]

        id_, confidendence = recognizer.predict(roi_gray)

        if confidendence<=85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color =  (255,255,255)
            stroke = 2
            if name not in nameList:
                nameList.append(name)
                markAttendance(name)
            cv2.putText(camera, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


        cv2.rectangle(camera, (x, y), (x+w, y+h), (153,50,204), 2)

    cv2.imshow('Video', camera)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        f = open('Attendance.csv', "w+")
        f.close()
        break



video_capture.release()
cv2.destroyAllWindows()
