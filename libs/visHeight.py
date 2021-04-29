import cv2
import numpy as np

def visHeight(frame, handToTableDist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    tempS = np.copy(s)
    tempS = tempS.astype('int16')
    tempS[tempS != 0] -= int(((handToTableDist)**1.3) * 20)
    tempS[tempS < 1] = 1
    tempS = tempS.astype('uint8')
    s[s != 0] = tempS[s != 0]

    tempV = np.copy(v)
    tempV = tempV.astype('int16')
    tempV[tempV != 0] -= int(((handToTableDist)**1.6) * 20)
    tempV[tempV < 1] = 1
    tempV = tempV.astype('uint8')
    v[v!=0] = tempV[v!=0]

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def handAsDepthframe(mask, caliDepthframe,tabledist):
    caliDepthframe = caliDepthframe.astype('float64')
    caliDepthframe = caliDepthframe - (tabledist+100)
    caliDepthframe *= (255.0 / 1200)
    caliDepthframe = caliDepthframe.astype('uint8')
    colored = cv2.cvtColor(caliDepthframe, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(colored, colored, mask=mask)
