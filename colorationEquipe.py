import numpy as np
import cv2
import math
import copy
import time
from xml.dom.minidom import *

rougeOk = 0
rougePasOk = 0
blancOk = 0
blancPasOk = 0
pasClasse = 0

doc = parse('result.xml')
framesXml = doc.getElementsByTagName("frame")
totalFrameXml = len(framesXml)

def compareCouleur(x,y,w,h):
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe1-[luminance,erreur,erreur], equipe1+[luminance,erreur,erreur])
    nb1 = cv2.countNonZero(dst)
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe2-[luminance,erreur,erreur], equipe2+[luminance,erreur,erreur])
    nb2 = cv2.countNonZero(dst)
    #cv2.putText(frameAffiche,str(nb1),(x+w+10,y+h-10),0,0.3,(0,0,255))
    #cv2.putText(frameAffiche,str(nb2),(x+w+10,y+h+10),0,0.3,(0,0,255))

    comp,distance = compareCentre(x+w/2, y+h/2, compteur)
    if nb1>(w*h)/facteur or (nb1>0 and nb2==0):
        coul = "red"
    elif nb2>(w*h)/facteur:
        coul = "white"
    else:
        coul = "On sait pas"
    message = compte(coul,comp)
    cv2.putText(frameAffiche,message,(x+w,y+h),0,0.3,(0,0,255))
    cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,0,0), 2)




def compte(s,result):
    global rougeOk,rougePasOk,blancOk,blancPasOk,pasClasse
    if s==result and s =="red":
        rougeOk = rougeOk +1
        return "rouge"
    elif s!= result and s=="red":
        rougePasOk = rougePasOk +1
        return "faux rouge"
    elif s==result and s=="white":
        blancOk = blancOk+1
        return "blanc"
    elif s!=result and s=="white":
        blancPasOk = blancPasOk+1
        return "faux blanc"
    else:
        pasClasse = pasClasse +1
        return "on sait pas"

def compareCentre(x,y,compteur):
    retour = "erreur"
    if compteur< totalFrameXml:
        f = framesXml[compteur]
        #print "    frame %s" % f.attributes["num"].value
        distance = 3500
        xd = 0
        yd = 0
        for pt in f.getElementsByTagName("point"):
            xr = int(pt.getElementsByTagName("x")[0].firstChild.nodeValue)
            yr = int(pt.getElementsByTagName("y")[0].firstChild.nodeValue)
            dist = math.sqrt((x-xr)*(x-xr)+(y-yr)*(y-yr))
            if dist<distance:
                #print "     nouvelle distance : %d, ancienne %d"%(dist,distance)
                xd,yd = xr,yr
                distance = dist
                retour = pt.getElementsByTagName("color")[0].firstChild.nodeValue
    #print x,y,xd,yd,compteur,retour
    return retour,distance


def draw(event,x,y,flags,param):
    if event == 1:
        global equipe1
        equipe1 = frame[y][x]
        print equipe1
        cv2.circle(frame,(x,y),1,(255,0,0),-1)
        cv2.imshow('frame',frame)
    if event == 2:
        global equipe2
        equipe2 = frame[y][x]
        print equipe2
        cv2.circle(frame,(x,y),1,(0,0,255),-1)
        cv2.imshow('frame',frame)

cap = cv2.VideoCapture('b1.webm',0)

ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec1 = np.ones((n,m))*255
grayPrec2 = np.ones((n,m))*255
grayPrec3 = np.ones((n,m))*255
grayPrec4 = np.ones((n,m))*255

equipe1 =  [0,0,0] #rouge
equipe2 = [0,0,0] #blanc
erreur = 10
luminance = 30
facteur = 1000

x = 0
ret,frame = cap.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#cv2.namedWindow('frame')
#cv2.setMouseCallback('frame',draw)
#cv2.imshow('frame',frame)
#k=cv2.waitKey()
#equipe1 = np.array([73,70,126])
#equipe2 = np.array([210,193,194])
equipe1 = np.array([3,65,149])  #hsb
equipe2 = np.array([111,35,190]) #hsb


k = cv2.waitKey(0)
compteur = 0;
while(cap.isOpened()):
    ret, frame = cap.read()

    frame2 = abs(frame - framePrec)
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    grayMin = np.minimum(gray,grayPrec1)
    grayMin = np.minimum(grayMin,grayPrec2)
    grayMin = np.minimum(grayMin,grayPrec3 )
    grayMin = np.minimum(grayMin,grayPrec4)
    grayMin = grayMin.astype(np.uint8)
    kernel1 = np.ones((5,5), np.uint8)
    kernel2 = np.ones((7,7), np.uint8)
    grayOuvert = cv2.morphologyEx(grayMin, cv2.MORPH_OPEN, kernel1)
    grayOuvert = cv2.morphologyEx(grayOuvert, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(grayOuvert, cv2.MORPH_CLOSE, kernel2)

    frameAffiche = copy.deepcopy(frame)
    im2, contours, hierarchy = cv2.findContours(grayOuvert.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #frameAffiche = cv2.bitwise_and(frameAffiche,frameAffiche,mask = mask)
    frameAffiche = copy.deepcopy(frame)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if h>20 and w>20 :
            compareCouleur(x,y,w,h)
    compteur = compteur + 1
    c = "rouge:%d, faux rouge:%d, blanc:%d, faux blanc:%d, non identifie:%d"%(rougeOk,rougePasOk,blancOk,blancPasOk,pasClasse)
    cv2.putText(frameAffiche,c,(50,50),0,0.6,(0,0,255))
    cv2.imshow('frame',frameAffiche)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    grayPrec4 = copy.deepcopy(grayPrec3)
    grayPrec3 = copy.deepcopy(grayPrec2)
    grayPrec2 = copy.deepcopy(grayPrec1)
    grayPrec1 = copy.deepcopy(gray)
    framePrec = copy.deepcopy(frame)
    time.sleep(0.2)
cap.release()
cv2.destroyAllWindows()
