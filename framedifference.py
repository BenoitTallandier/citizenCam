import numpy as np
import cv2
import math
import copy
import time
cap = cv2.VideoCapture('b1.webm',0)

ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
x = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    frame2 = abs(frame - framePrec)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret1,gray = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    framePrec = copy.deepcopy(frame)
    #time.sleep(1)
cap.release()
cv2.destroyAllWindows()
