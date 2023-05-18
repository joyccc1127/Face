import cv2 as cv
import os

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

img = cv.imread('photo/blackpink.jpg')
# cv.imshow('IU', img)

resized_img = rescaleFrame(img)
# cv.imshow('IU Resized', resized_img)

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

basepath = os.path.dirname(os.path.abspath(cv.__file__))
haar_cascade = cv.CascadeClassifier(basepath + "\data\haarcascade_frontalface_default.xml")

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
print(f'Number of faces found : {len(face_rect)}')

for (x,y,w,h) in face_rect:
    cv.rectangle(resized_img, (x,y), (x+w, y+h), (0,0,255), thickness=2)
cv.imshow('Detected face', resized_img)

cv.waitKey(0)