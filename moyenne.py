import numpy as np
import cv2
import math
import copy
import time

cap = cv2.VideoCapture('b1.webm',0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret,frame3 = cap.read()

(n,m,z) = frame1.shape
x = 0
seuil = 1.5
while(cap.isOpened()):
    ret, frame4 = cap.read()

    frameFinale = abs(frame4-frame3)
    gray = cv2.cvtColor(frameFinale, cv2.COLOR_BGR2GRAY)
    for i in range(n):
        for j in range(m):
            maxStd = 0;
            for k in range(3):
                x = np.std(np.array([frame1[i][j][k],frame2[i][j][k],frame3[i][j][k],frame4[i][j][k]]))
                maxStd = max(x,maxStd)
            if maxStd < seuil:
                gray[i][j] = 0
        print(i*100/n)
    cv2.imshow('frame',gray)
    frame1 = frame2.copy()
    frame2 = frame3.copy()
    frame3 = frame4.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
