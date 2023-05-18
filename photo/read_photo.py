import cv2 as cv

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

# Read photo
img = cv.imread('photo/IU1.jpg')
# cv.imshow('IU', img)

resized_img = rescaleFrame(img)
cv.imshow('IU Resized', resized_img)

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

cv.waitKey(0)