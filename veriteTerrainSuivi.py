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
xMouse=0
yMouse=0
isOnScreen = True
def draw(event,x,y,flags,param):
    if event == 0:
        global xMouse
        global yMouse
        xMouse,yMouse = x,y

        #print xMouse,yMouse

liste = []

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw)
ret, frame = cap.read()
cv2.imshow('frame',frame)
k = cv2.waitKey(0)
print k
timer = 0.125;
compteur = 0
doc = Document()
doc = parse('result.xml')
framesXml = doc.getElementsByTagName("frame")
result = doc.getElementsByTagName("result")[0]
totalFrameXml = len(framesXml)
compteur = 0;
stop = False
while(cap.isOpened() ):
    ret, frame = cap.read()
    frequence = 1/timer
    message = "frame : %d; img/s : %d, rec : %r" %(compteur,frequence,isOnScreen)
    if compteur<totalFrameXml:
        for rect in framesXml[compteur].getElementsByTagName("point"):
            xr = int(rect.getElementsByTagName("x")[0].firstChild.nodeValue)
            yr = int(rect.getElementsByTagName("y")[0].firstChild.nodeValue)
            color = rect.getElementsByTagName("color")[0].firstChild.nodeValue
            if color=="red":
                cv2.circle(frame,(xr,yr),15,(0,0,100),-1)
            else:
                cv2.circle(frame,(xr,yr),15,(0,100,0),-1)
    if isOnScreen :
        cv2.putText(frame,message,(25,25),0,0.7,(0,0,255),)
    else:
        cv2.putText(frame,message,(25,25),0,0.7,(0,125,0))
    #cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    if k == 32:
        isOnScreen = not(isOnScreen)
    elif k == 65365:
        timer = timer/2
    elif k ==65366:
        timer = 2*timer
    elif k==10:
        cv2.putText(frame,"Pause",(250,250),0,8,(125,0,0),4)
        cv2.imshow('frame',frame)
        space = cv2.waitKey(0)
        while( space != 10):
            space = cv2.waitKey(0)
    elif k == ord('z'):
        stop = True
        break
    elif k == ord('q'):
        break

    #cv2.imshow('frame',frame)

    if isOnScreen:
        liste.append((xMouse,yMouse))
        cv2.circle(frame,(xMouse,yMouse),5,(255,0,0),-1)

    else :
        liste.append((-1,-1))
    compteur = compteur +1
    cv2.imshow('frame',frame)
    time.sleep(timer)


cap.release()
cv2.destroyAllWindows()
compteur = 0
if stop==False:
    print("Nombre de frame : %d"%totalFrameXml)
    for (x,y) in liste:
        f=[node for node in framesXml if node.attributes["num"].value == str(compteur)]
        if len(f)>0:
            f = f[0]
        else:
            f = doc.createElement("frame")
            f.setAttribute("num", str(compteur))

        if (x,y) != (-1,-1):
            ptXml = doc.createElement("point")

            posX = doc.createElement("x")
            posX.appendChild(doc.createTextNode(str(x)))
            ptXml.appendChild(posX)

            posY = doc.createElement("y")
            posY.appendChild(doc.createTextNode(str(y)))
            ptXml.appendChild(posY)

            color = doc.createElement("color")
            color.appendChild(doc.createTextNode("white"))
            ptXml.appendChild(color)
        f.appendChild(ptXml)
        if compteur>= totalFrameXml:
            result.appendChild(f)
        compteur = compteur + 1

    docu = open("result.xml","wb")
    result.writexml(docu)
    docu.close()
