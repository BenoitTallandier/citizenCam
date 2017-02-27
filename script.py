import numpy as np
import cv2
import math
import copy
import time
cap1 = cv2.VideoCapture('terrain_vide.avi',0)
ret, frameVide = cap1.read()
grayVide = cv2.cvtColor(frameVide, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture('b1.webm',0)

ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
x = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    ## code perso
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayPrec = cv2.cvtColor(framePrec, cv2.COLOR_BGR2GRAY)
    #gray = abs(gray - grayPrec)
    #ret1,gray = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    #gray = cv2.medianBlur(gray,9)
    ##

    frame2 = cv2.absdiff(frame, frameVide)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,40,255,0)
    frameAffiche = frame.copy()
    im2, contours, hierarchy = cv2.findContours(gray.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('frame',gray)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow('frame',frameAffiche)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        time.sleep(2)
    framePrec = copy.deepcopy(frame)
    #time.sleep(1)
cap.release()
cv2.destroyAllWindows()
