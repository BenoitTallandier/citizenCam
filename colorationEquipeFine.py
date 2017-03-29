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
while(cap.isOpened()):
    ret, frame = cap.read()

    frame2 = abs(frame - framePrec)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    grayMin = np.minimum(gray,grayPrec1)
    grayMin = np.minimum(grayMin,grayPrec2)
    grayMin = grayMin.astype(np.uint8)
    kernel1 = np.ones((5,5), np.uint8)
    kernel2 = np.ones((7,7), np.uint8)
    grayOuvert = cv2.morphologyEx(grayMin, cv2.MORPH_OPEN, kernel1)
    grayOuvert = cv2.morphologyEx(grayOuvert, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(grayOuvert, cv2.MORPH_CLOSE, kernel2)

    frameAffiche = copy.deepcopy(frame)
    im2, contours, hierarchy = cv2.findContours(grayOuvert.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    frameMask = cv2.bitwise_and(frameAffiche,frameAffiche,mask = mask)
    frameAffiche = copy.deepcopy(frame)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if h>30 and w>30 :
            somme = 0
            bleu = int(np.mean(frameMask[y:y+h,x:x+h,0:1]))
            vert = int(np.mean(frameMask[y:y+h,x:x+h,1:2]))
            rouge = int(np.mean(frameMask[y:y+h,x:x+h,2:3]))

            moyenne = "(%d,%d,%d)" %(rouge,vert,bleu)
            if rouge > bleu+5 and rouge > vert+5:
                message = "rouge"
                cv2.putText(frameAffiche,message,(x+w+10,y+h),0,0.3,(0,0,255))
                #   cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,0,255), 2)
                if compteur<totalFrameXml:
                    totalRectangle = framesXml[compteur].getElementsByTagName("rectangle")
                    find=False
                    for rect in totalRectangle:
                        xr = int(rect.getElementsByTagName("x")[0].firstChild.nodeValue)
                        yr = int(rect.getElementsByTagName("y")[0].firstChild.nodeValue)
                        wr = int(rect.getElementsByTagName("h")[0].firstChild.nodeValue)
                        hr = int(rect.getElementsByTagName("w")[0].firstChild.nodeValue)
                        color = rect.getElementsByTagName("color")[0].firstChild.nodeValue

                        if x>xr-erreur and y>yr-erreur and  x+w<xr+wr+erreur and y+h<yr+hr+erreur:
                            rougeOk = rougeOk+1
                            cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,0,0), 2)
                            find = True
                            break
                    if find==False :
                        rougePasOk = rougePasOk +1
                    message = "bon : %d, pas bon : %d" %(rougeOk, rougePasOk)
                    cv2.putText(frameAffiche,message,(20,20),0,0.3,(0,0,0))


            elif bleu>rouge+5 and bleu>vert+5:
                message = "bleu"
                cv2.putText(frameAffiche,message,(x+w+10,y+h),0,0.3,(0,255,0))
                cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,255,0), 2)

            else:
                message = "blanc"
                cv2.putText(frameAffiche,message,(x+w+10,y+h),0,0.3,(255,0,0))
                cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,255,0), 2)


            cv2.putText(frameAffiche,moyenne,(x+w+10,y),0,0.3,(255,0,0))
    compteur = compteur + 1
    cv2.imshow('frame',frameAffiche)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    grayPrec2 = copy.deepcopy(grayPrec1)
    grayPrec1 = copy.deepcopy(gray)
    framePrec = copy.deepcopy(frame)
    #time.sleep(1)
cap.release()
cv2.destroyAllWindows()
