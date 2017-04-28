import numpy as np
import cv2
import math
import copy
import time
from xml.dom.minidom import *

rougeRouge = 0
rougeBlanc = 0
rougeArbitre = 0
blancRouge = 0
blancBlanc = 0
blancArbitre = 0
arbitreRouge = 0
arbitreBlanc = 0
arbitreArbitre = 0
rejetRouge = 0
rejetBlanc = 0
rejetArbitre = 0

DADR=0 #detecte dans l'algo et detecte en realite
DADRtemp = 0
DANR=0
NADR = 0
cumulTaille = 0

seuilComptabilisationDetection = 50

joueurPris = []
joueur1 = []
joueur2 = []
doc = parse('result.xml')
framesXml = doc.getElementsByTagName("frame")
totalFrameXml = len(framesXml)

def compareCouleur(x,y,w,h,compteur):
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe1-[luminance,erreur,erreur], equipe1+[luminance,erreur,erreur])
    nb1 = cv2.countNonZero(dst)
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe2-[luminance,erreur,erreur], equipe2+[luminance,erreur,erreur])
    nb2 = cv2.countNonZero(dst)
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe3-[luminance,erreur,erreur], equipe3+[luminance,erreur,erreur])
    nb3 = cv2.countNonZero(dst)

    #cv2.putText(frameAffiche,str(nb1),(x+w+10,y+h-10),0,0.3,(0,0,255))
    #cv2.putText(frameAffiche,str(nb2),(x+w+10,y+h+10),0,0.3,(0,0,255))

    comp,distance,xretour,yretour = compareCentre(x+w/2, y+h/2, compteur)
    cv2.circle(frameAffiche,(x+w/2,y+h/2),3,(255,0,0),-1)

    if nb3>(w*h)/facteur:
        coul = "arbitre"
    elif nb1>(w*h)/facteur or (nb1>0 and nb2==0):
        coul = "red"
    elif nb2>(w*h)/facteur:
        coul = "white"
    else:
        coul = "none"
    message = compte(coul,comp)
    cv2.putText(frameAffiche,message,(x+w,y+h),0,0.3,(0,0,255))
    cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,0,0), 2)



def calculVitesse(x1,y1,w,h):
    global joueur1
    x = x1 + w/2
    y = y1 + h/2
    distance = 2000
    for j in joueur1:
        dist = math.sqrt((x-j[0])*(x-j[0])+(y-j[1])*(y-j[1]))
        if dist<distance:
            distance = dist
    cv2.putText(frameAffiche,"v=%d"%distance,(x1+w+2,y1+h-10),0,0.3,(255,0,0))
    return distance

def compte(s,result):
    global rougeRouge,rougeBlanc,rougeArbitre,blancRouge,blancBlanc,blancArbitre,arbitreRouge,arbitreBlanc,arbitreArbitre,rejetRouge,rejetBlanc,rejetArbitre
    if s=="red":
        if result=="red":
            rougeRouge = rougeRouge+1
            return "rouge"
        elif result =="white":
            rougeBlanc = rougeBlanc +1
            return "rougeBlanc"
        elif result=="arbitre":
            rougeArbitre = rougeArbitre +1
            return "rougeArbitre"
    elif s=="white":
        if result=="red":
            blancRouge = blancRouge +1
            return "blancRouge"
        elif result=="white":
            blancBlanc = blancBlanc +1
            return "blancBlanc"
        elif result == "arbitre":
            blancArbitre = blancArbitre +1
            return "blancArbitre"
    elif s=="arbitre":
        if result=="red":
            arbitreRouge = arbitreRouge +1
            return "arbitreRouge"
        elif result=="white":
            arbitreBlanc = arbitreBlanc +1
            return "arbitreBlanc"
        elif result == "arbitre":
            arbitreArbitre = arbitreArbitre +1
            return "arbitreArbitre"
    elif s=="none":
        if result=="red":
            rejetRouge = rejetRouge +1
            return "rejetRouge"
        elif result=="white":
            rejetBlanc = rejetBlanc +1
            return "rejetBlanc"
        elif result == "arbitre":
            rejetArbitre = rejetArbitre +1
            return "rejetArbitre"
    return "ERREUR"

def compareCentre(x,y,compteur):
    retour = "erreur"
    if compteur<totalFrameXml:
        f = framesXml[compteur]
        #print "    frame %s, demande %d" %(f.attributes["num"].value,compteur)
        distance = seuilComptabilisationDetection
        xd = 0
        yd = 0
        index = -1
        c=0
        for pt in f.getElementsByTagName("point"):
            xr = int(pt.getElementsByTagName("x")[0].firstChild.nodeValue)
            yr = int(pt.getElementsByTagName("y")[0].firstChild.nodeValue)
            dist = math.sqrt((x-xr)*(x-xr)+(y-yr)*(y-yr))

            if dist<distance and not (c in joueurPris):
                #print "     nouvelle distance : %d, ancienne %d"%(dist,distance)
                index = c
                xd,yd = xr,yr
                distance = dist
                retour = pt.getElementsByTagName("color")[0].firstChild.nodeValue
            c = c+1
        compteDetection(index,len(f.getElementsByTagName("point")))
        if index< len(f.getElementsByTagName("point")):
            joueurPris.append(index)
        #print" frame %d, x=%d, y=%d"%(compteur,xd,yd)
        #cv2.circle(frameAffiche,(xd,yd),5,(0,155,100),-1)
    #print x,y,xd,yd,compteur,retour
    return retour,distance,xd,yd

def compteDetection(index,tailleMax):
    ### index<0 -> on a termine la frame
    global DADR,DANR,NADR,DADRtemp
    if index <-1: #on fait le total
        NADR = NADR + tailleMax - DADRtemp
    elif index >= tailleMax or index == -1:
        DANR = DANR+1
    else :
        DADR = DADR + 1
        DADRtemp = DADRtemp +1



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

equipe1 =  [0,0,0] #rouge
equipe2 = [0,0,0] #blanc
erreur = 10
luminance = 30
facteur = 1000

x = 0
#ret,frame = cap.read()
#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#cv2.namedWindow('frame')
#cv2.setMouseCallback('frame',draw)
#cv2.imshow('frame',frame)
#k=cv2.waitKey()
#equipe1 = np.array([73,70,126])
#equipe2 = np.array([210,193,194])
equipe1 = np.array([3,65,149])  #hsb
equipe2 = np.array([111,35,190]) #hsb
equipe3 = np.array([108,160,163]) #hsb arbitre

k = 0
compteur = 1;
while(cap.isOpened() and compteur<=2000):
    ret, frame = cap.read()
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(framePrec, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.absdiff(gray, gray1)
    ret, grayOuvert = cv2.threshold(frame2,10,255,cv2.THRESH_TOZERO)

    frameAffiche = copy.deepcopy(frame)
    im2, contours, hierarchy = cv2.findContours(grayOuvert.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #frameAffiche = cv2.bitwise_and(frameAffiche,frameAffiche,mask = mask)
    frameAffiche = copy.deepcopy(frame)
    joueur2 = []
    for pt in framesXml[compteur].getElementsByTagName("point"):
        xr = int(pt.getElementsByTagName("x")[0].firstChild.nodeValue)
        yr = int(pt.getElementsByTagName("y")[0].firstChild.nodeValue)
        cv2.circle(frameAffiche,(xr,yr),3,(0,0,100),-1)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if h>40 and w>40:
            compareCouleur(x,y,w,h,compteur)
            cv2.putText(frameAffiche,"%d %d"%(h,w),(x+w,y+h-30),0,0.3,(0,100,0))
            #calculVitesse(x,y,w,h)
            #joueur2.append([x+w/2,y+h/2])
    compteDetection(-2,len(framesXml[compteur].getElementsByTagName("point")))
    joueurPris = []
    joueur1 = copy.copy(joueur2)
    cumulTaille = cumulTaille + len(framesXml[compteur].getElementsByTagName("point"))
    c = "frame:%d, DADR:%d,  NADR:%d, DANR:%d, total:%d / %d"%(compteur,DADR,NADR,DANR,DADR+NADR,cumulTaille)
    DADRtemp = 0
    #c = "frame:%d, rouge:%d, faux rouge:%d, blanc:%d, faux blanc:%d, arbitre:%d, non identifie:%d"%(compteur,rougeOk,rougePasOk,blancOk,blancPasOk,arbitre,pasClasse)
    cv2.putText(frameAffiche,c,(50,50),0,0.6,(0,0,255))
    compteur = compteur + 1
    cv2.imshow('frame',frameAffiche)
    #k=cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    framePrec = copy.deepcopy(frame)
    #print(compteur)
    #time.sleep(5)
cap.release()
cv2.destroyAllWindows()

print ("DADR:%d,  NADR:%d, DANR:%d, total:%d / %d\n\n"%(DADR,NADR,DANR,DADR+NADR,cumulTaille))
print("        rouge|blanc|arbitre|rejet\n rouge | %d | %d | %d | %d \nblanc | %d | %d | %d | %d \narbitre | %d | %d | %d | %d \n"%(rougeRouge,blancRouge,arbitreRouge,rejetRouge,rougeBlanc,blancBlanc,arbitreBlanc,rejetBlanc,rougeArbitre,blancArbitre,arbitreArbitre,rejetArbitre))
