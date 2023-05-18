import os
import cv2 as cv
import numpy as np

name = ["IU", "Jennie", "Jisoo", "Lisa", "Rose"]
Dir = r"C:/Project/face/dataset"

basepath = os.path.dirname(os.path.abspath(cv.__file__))
haar_cascade = cv.CascadeClassifier(basepath + "\data\haarcascade_frontalface_default.xml")

features = []
labels = []

def create_train():
    for person in name:
        path = os.path.join(Dir, person)
        label = name.index(person)

        for img_filename in os.listdir(path):
            img_path = os.path.join(path, img_filename)

            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            face_rect =  haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
            for (x,y,w,h) in face_rect:
                face = gray[y:y+h, x:x+w]
                features.append(face)
                labels.append(label)

create_train()
print(f"Features : {len(features)}")
print(f"Labels : {len(labels)}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()

features = np.array(features, dtype = "object")
labels = np.array(labels)
face_recognizer.train(features, labels)

face_recognizer.save("face_train.yml")