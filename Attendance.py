import cv2
<<<<<<< HEAD
import xlsxwriter
from datetime import datetime


#Create a workbook, add sheet 
workbook = xlsxwriter.Workbook('Attendance01.xlsx')
worksheet = worksheet.add_worksheet()
row = 0
col = 0

=======

>>>>>>> parent of 3313453 (Update Attendance.py)
faceCascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

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
        cv2.rectangle(camera, (x, y), (x+w, y+h), (153,50,204), 2)

    cv2.imshow('Video', camera)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
