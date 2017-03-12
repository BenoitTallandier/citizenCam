import numpy as np
import cv2
import math
import copy
import time
cap1 = cv2.VideoCapture('terrain_vide.avi',0)
ret, frameVide = cap1.read()
cap = cv2.VideoCapture('b1.webm',0)

while(cap.isOpened()):
    ret, frame = cap.read()


    #frame2 = cv2.absdiff(frame, frameVide)
    frame2 = abs(frame - frameVide)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('p'):
        time.sleep(2)
    #time.sleep(1)
cap.release()
cv2.destroyAllWindows()
