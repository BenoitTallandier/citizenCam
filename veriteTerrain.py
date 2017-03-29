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

x = 0

def draw(event,x,y,flags,param):
    if event == 1:
        cv2.circle(frame,(x,y),5,(255,0,0),-1)
        cv2.imshow('frame',frame)
    if event == 2:
        cv2.circle(frame,(x,y),5,(0,0,255),-1)
        cv2.imshow('frame',frame)
    if event == 3:
        cv2.circle(frame,(x,y),5,(0,255,0),-1)
        cv2.imshow('frame',frame)


cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw)
retour = False
while(cap.isOpened() ):
    if(retour == False):
        ret, frame = cap.read()
        save = copy.deepcopy(frame)
    else:
        frame = copy.deepcopy(save)
        retour = False

    cv2.imshow('frame',frame)

    k=cv2.waitKey(1)
    if k == 27:
        retour = True
    if k == ord('q'):
        break

    time.sleep(0.1)
cap.release()
cv2.destroyAllWindows()
