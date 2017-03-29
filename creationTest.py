import numpy as np
import cv2
import math
import copy
import time
from xml.dom.minidom import Document

cap = cv2.VideoCapture('b1.webm',0)
ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec1 = np.ones((n,m))*255
grayPrec2 = np.ones((n,m))*255
doc = Document()
result = doc.createElement("result")

compteur = 0
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
    frameXml = doc.createElement("frame")
    frameXml.setAttribute("num", str(compteur))
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

        rectXml = doc.createElement("rectangle")

        posX = doc.createElement("x")
        posX.appendChild(doc.createTextNode(str(x)))
        rectXml.appendChild(posX)

        posY = doc.createElement("y")
        posY.appendChild(doc.createTextNode(str(y)))
        rectXml.appendChild(posY)

        posW = doc.createElement("w")
        posW.appendChild(doc.createTextNode(str(w)))
        rectXml.appendChild(posW)

        posH = doc.createElement("h")
        posH.appendChild(doc.createTextNode(str(h)))
        rectXml.appendChild(posH)

        color = doc.createElement("color")
        color.appendChild(doc.createTextNode("rouge"))
        rectXml.appendChild(color)

        frameXml.appendChild(rectXml)

    result.appendChild(frameXml)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        docu = open("result.xml","wb")
        result.writexml(docu)
        docu.close()
        break
    #time.sleep(0.05)
    compteur = compteur + 1


docu = open("result.xml","wb")
result.writexml(docu)
docu.close()
cap.release()
cv2.destroyAllWindows()
