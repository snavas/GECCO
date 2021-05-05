import cv2
from cv2 import aruco
import numpy as np
import math
from classes.realsense import RealSense

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters_create()
parameters.errorCorrectionRate = 0.7

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

def detectcolorFlat(colorframe, lower_color, upper_color):
    grey = cv2.cvtColor(colorframe, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(grey, aruco_dict, parameters=parameters)

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
        upper_color = getColorBetweenMarkers(colorframe, colorMarkersUpper)
    if len(colorMarkersLower) == 2:
        lower_color = getColorBetweenMarkers(colorframe, colorMarkersLower)

    return lower_color, upper_color

def getUpperLowerSquare(colorMarkers, colorframe, colorspace):
    x1 = int(colorMarkers[0][0][0])
    y1 = int(colorMarkers[0][0][1])
    x2 = int(colorMarkers[1][0][0])
    y2 = int(colorMarkers[1][0][1])
    center = calculateCenter(x1, y1, x2, y2)

    extent = 15
    innerRectangleXIni = center[0] - extent
    innerRectangleYIni = center[1] - extent
    innerRectangleXFin = center[0] + extent
    innerRectangleYFin = center[1] + extent
    roi = colorframe[innerRectangleYIni +
                     1:innerRectangleYFin, innerRectangleXIni +
                                           1:innerRectangleXFin]
    hsvRoi = cv2.cvtColor(roi, colorspace)

    lower = np.array(
        [hsvRoi[:, :, 0].min() - 3, hsvRoi[:, :, 1].min() - 3, hsvRoi[:, :, 2].min() - 3])
    upper = np.array(
        [hsvRoi[:, :, 0].max() + 3, hsvRoi[:, :, 1].max() + 3, hsvRoi[:, :, 2].max() + 3])
    
    return lower, upper

def getUpperLowerCircle(colorMarkers, colorframe, colorspace):
    x1 = int(colorMarkers[0][0][0])
    y1 = int(colorMarkers[0][0][1])
    x2 = int(colorMarkers[1][0][0])
    y2 = int(colorMarkers[1][0][1])
    center = calculateCenter(x1, y1, x2, y2)

    colorframe = cv2.cvtColor(colorframe, colorspace)

    r = 15
    mask = np.zeros(colorframe.shape[:2], dtype="uint8")
    cv2.circle(mask, (center[0], center[1]), r, 255, -1)
    roi = cv2.bitwise_and(colorframe, colorframe, mask=mask)

    upper = np.array(
        [roi[:, :, 0].max(), roi[:, :, 1].max(), roi[:, :, 2].max()])
    roi[np.where((roi == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    lower = np.array(
        [roi[:, :, 0].min(), roi[:, :, 1].min(), roi[:, :, 2].min()])

    return lower, upper

def detectcolor3D(colorframe, lower_color, upper_color, colorspace):
    grey = cv2.cvtColor(colorframe, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(grey, aruco_dict, parameters=parameters)

    colorMarkersA = []
    colorMarkersB = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 1:
                c = corners[i][0]
                colorMarkersA.append(c)
            if ids[i] == 2:
                c = corners[i][0]
                colorMarkersB.append(c)

    if len(colorMarkersA) == 2 & len(colorMarkersB) == 2:
        lower_colorA, upper_colorA = getUpperLowerCircle(colorMarkersA, colorframe, colorspace)
        lower_colorB, upper_colorB = getUpperLowerCircle(colorMarkersB, colorframe, colorspace)
        lower_color = np.array([min(lower_colorA[0], lower_colorB[0]),
                               min(lower_colorA[1], lower_colorB[1]),
                               min(lower_colorA[2], lower_colorB[2])])
        upper_color = np.array([max(upper_colorA[0], upper_colorB[0]),
                               max(upper_colorA[1], upper_colorB[1]),
                               max(upper_colorA[2], upper_colorB[2])])
    elif len(colorMarkersA) == 2:
        lower_color, upper_color = getUpperLowerCircle(colorMarkersA, colorframe, colorspace)
    elif len(colorMarkersB) == 2:
        lower_color, upper_color = getUpperLowerCircle(colorMarkersB, colorframe, colorspace)

    return lower_color, upper_color