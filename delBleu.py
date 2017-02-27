import numpy as np
import cv2
import math
import copy
import time

cap = cv2.VideoCapture('b1.webm',0)
ret1, framePrec = cap.read()
framePrec[:,:,0] = 0
(n,m,z) = framePrec.shape

while(cap.isOpened()):
    ret, frame = cap.read()

    ## code perso
    frame[:,:,0] = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayPrec = cv2.cvtColor(framePrec, cv2.COLOR_BGR2GRAY)
    gray = abs(gray - grayPrec)
    ret1,gray = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    gray = cv2.medianBlur(gray,1)

    ret,gray = cv2.threshold(gray,50,255,0)

    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    framePrec = copy.deepcopy(frame)
    #time.sleep(1)
cap.release()
cv2.destroyAllWindows()
