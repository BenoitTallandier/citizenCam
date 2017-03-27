import numpy as np
import cv2
import math
import copy
import time

cap = cv2.VideoCapture('b1.webm',0)

ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec1 = np.ones((n,m))*255
grayPrec2 = np.ones((n,m))*255
grayPrec3 = np.ones((n,m))*255
grayPrec4 = np.ones((n,m))*255


x = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    frame2 = abs(frame - framePrec)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    grayMin = np.minimum(gray,grayPrec1)
    grayMin = np.minimum(grayMin,grayPrec2)
    grayMin = np.minimum(grayMin,grayPrec3 )
    grayMin = np.minimum(grayMin,grayPrec4)
    grayMin = grayMin.astype(np.uint8)
    kernel1 = np.ones((5,5), np.uint8)
    kernel2 = np.ones((7,7), np.uint8)
    grayOuvert = cv2.morphologyEx(grayMin, cv2.MORPH_OPEN, kernel1)
    grayOuvert = cv2.morphologyEx(grayOuvert, cv2.MORPH_CLOSE, kernel2)

    frameAffiche = copy.deepcopy(grayOuvert)
    im2, contours, hierarchy = cv2.findContours(grayOuvert.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (150,150,150), 2)

    cv2.imshow('frame',frameAffiche)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    grayPrec4 = copy.deepcopy(grayPrec3)
    grayPrec3 = copy.deepcopy(grayPrec2)
    grayPrec2 = copy.deepcopy(grayPrec1)
    grayPrec1 = copy.deepcopy(gray)
    framePrec = copy.deepcopy(frame)
    #time.sleep(1)
cap.release()
cv2.destroyAllWindows()
