import cv2
from cv2 import aruco
import numpy as np
import math
from classes.realsense import RealSense
def calculateCenter(x1,y1,x2,y2):
    center = [-1,-1]
    center[0] = int((x2 - x1)/2 + x1)
    center[1] = int((y2 - y1)/2 + y1)
    return center

def getColorBetweenMarkers(image, colorMarkers):
    x1 = int(colorMarkers[0][0][0])
    y1 = int(colorMarkers[0][0][1])
    x2 = int(colorMarkers[1][0][0])
    y2 = int(colorMarkers[1][0][1])
    center = calculateCenter(x1, y1, x2, y2)
    return np.array(image[center[1], center[0]])

def detectPink(colorframe, lower_pink, upper_pink):
    gray = cv2.cvtColor(colorframe, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    colorMarkersUpper = []
    colorMarkersLower = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 11:
                c = corners[i][0]
                colorMarkersUpper.append(c)
            if ids[i] == 10:
                c = corners[i][0]
                colorMarkersLower.append(c)
    colorframe = cv2.cvtColor(colorframe, cv2.COLOR_BGR2HSV)
    if len(colorMarkersUpper) == 2:
        upper_pink = getColorBetweenMarkers(colorframe, colorMarkersUpper)
    if len(colorMarkersLower) == 2:
        lower_pink = getColorBetweenMarkers(colorframe, colorMarkersLower)

    return lower_pink, upper_pink