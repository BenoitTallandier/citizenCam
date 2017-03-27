import numpy as np
import cv2
import math
import copy
import time

cap = cv2.VideoCapture('b1.webm',0)

ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec1 = np.ones((n,m))*255
grayPrec2 = np.ones((n,m))*255


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
    frameMask = cv2.cvtColor(frameMask,cv2.COLOR_BGR2HSV)
    frameAffiche = copy.deepcopy(frame)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if h>30 and w>30 :
            somme = 0
            compteur = 0;
            teinte = int(np.mean(frameMask[y:y+h,x:x+h,0:1]))
            saturation = int(np.mean(frameMask[y:y+h,x:x+h,1:2]))
            lumiere = int(np.mean(frameMask[y:y+h,x:x+h,2:3]))
            #totalTeinte.append(teinte)
            #totalSaturation.append(saturation)
            #totalLumiere.append(lumiere)
            moyenne = "(%d,%d,%d)" %(teinte,saturation,lumiere)


            if teinte >160 or teinte < 30:
                message = "rouge"
                cv2.putText(frameAffiche,message,(x+w+10,y+h),0,0.3,(0,0,255))
                cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,0,255), 2)

            elif lumiere >90:
                message = "blanc"
                cv2.putText(frameAffiche,message,(x+w+10,y+h),0,0.3,(255,0,0))
                cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,255,0), 2)

            elif lumiere<60 and saturation<20:
                message = "Arbitre"
                cv2.putText(frameAffiche,message,(x+w+10,y+h),0,0.3,(0,0,0))
                cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,0,0), 2)

            else:
                message ="on sait pas"
                cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frameAffiche,moyenne,(x+w+10,y),0,0.3,(255,0,0))

    cv2.imshow('frame',frameAffiche)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    grayPrec2 = copy.deepcopy(grayPrec1)
    grayPrec1 = copy.deepcopy(gray)
    framePrec = copy.deepcopy(frame)
    time.sleep(0)
cap.release()
cv2.destroyAllWindows()
