import numpy as np
import cv2
import math
import copy
import time
cap = cv2.VideoCapture('b1.webm',0)

ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec = np.ones((n,m))*255
x = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    frame2 = abs(frame - framePrec)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grayMin = np.minimum(gray,grayPrec)
    cv2.imshow('frame',grayMin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    grayPrec = copy.deepcopy(gray)
    framePrec = copy.deepcopy(frame)
    #time.sleep(1)
cap.release()
cv2.destroyAllWindows()
