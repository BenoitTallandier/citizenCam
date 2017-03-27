import numpy as np
import cv2
import math
import copy
import time
from xml.dom.minidom import *

cap = cv2.VideoCapture('b1.webm',0)
ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec1 = np.ones((n,m))*255
grayPrec2 = np.ones((n,m))*255
doc = parse('result.xml')
framesXml = doc.getElementsByTagName("frame")
totalFrameXml = len(framesXml)

erreur = 50

compteur = 0
rougeOk = 0
rougePasOk = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    maskR = cv2.inRange(frame, np.array([60,60,110]), np.array([100,100,150]))
    maskB = cv2.inRange(frame, np.array([190,180,160]), np.array([230,220,200]))
    outputR = cv2.bitwise_and(frame, frame, mask = maskR)
    #outputB = cv2.bitwise_and(frame,frame,mask=maskB)
    gray = cv2.cvtColor(outputR.copy(), cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    kernel1 = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
    kernel2 = np.ones((31,31), np.uint8)
    grayOuvert = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel1)
    grayOuvert = cv2.morphologyEx(grayOuvert.copy(), cv2.MORPH_CLOSE, kernel2)
    grayOuvert = cv2.morphologyEx(grayOuvert.copy(), cv2.MORPH_CLOSE, kernel2)

    m2, contours, hierarchy = cv2.findContours(grayOuvert.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if compteur<totalFrameXml:
            totalRectangle = framesXml[compteur].getElementsByTagName("rectangle")
            find=False
            for rect in totalRectangle:
                xr = int(rect.getElementsByTagName("x")[0].firstChild.nodeValue)
                yr = int(rect.getElementsByTagName("y")[0].firstChild.nodeValue)
                wr = int(rect.getElementsByTagName("h")[0].firstChild.nodeValue)
                hr = int(rect.getElementsByTagName("w")[0].firstChild.nodeValue)
                color = rect.getElementsByTagName("color")[0]

                if x>xr-erreur and y>yr-erreur and x+w<xr+wr+erreur and y+h<yr+hr+erreur:
                    rougeOk = rougeOk+1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
                    find = True
                    break
            if find==False :
                rougePasOk = rougePasOk +1
            message = "bon : %d, pas bon : %d" %(rougeOk, rougePasOk)
            cv2.putText(frame,message,(20,20),0,0.3,(0,0,255))


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.05)
    compteur = compteur + 1



cap.release()
cv2.destroyAllWindows()
