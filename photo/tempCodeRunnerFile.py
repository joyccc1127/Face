import os
import cv2 as cv
import numpy as np

name = ["IU", "Jennie", "Jisoo", "Lisa", "Rose"]

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_train.yml")

img = cv.imread('IU.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Detect the face
basepath = os.path.dirname(os.path.abspath(cv.__file__))
haar_cascade = cv.CascadeClassifier(basepath + "\data\haarcascade_frontalface_default.xml")

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
for (x,y,w,h) in face_rect:
    face = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(face)
    print(f"Label : {name(label)} with a confidence of {confidence}")

    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness = 2)
    cv.putText(img, str(name[label]), img)

cv.waitKey(0)
