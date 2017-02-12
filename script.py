import numpy as np
import cv2
import math
import copy

cap = cv2.VideoCapture('b1.webm')


ret1, framePrec = cap.read()
x = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frame2 = cv2.absdiff(frame, framePrec)
    gray = frame2 #cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    framePrec = copy.deepcopy(frame)

cap.release()
cv2.destroyAllWindows()