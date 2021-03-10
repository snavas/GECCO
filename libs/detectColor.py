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

def detectPinkFlat(colorframe, lower_pink, upper_pink):
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
        upper_pink = getColorBetweenMarkers(colorframe, colorMarkersUpper)
    if len(colorMarkersLower) == 2:
        lower_pink = getColorBetweenMarkers(colorframe, colorMarkersLower)

    return lower_pink, upper_pink

def detectPink3D(colorframe, lower_pink, upper_pink):
    grey = cv2.cvtColor(colorframe, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(grey, aruco_dict, parameters=parameters)

    colorMarkers = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 1:
                c = corners[i][0]
                colorMarkers.append(c)

    if len(colorMarkers) == 2:
        x1 = int(colorMarkers[0][0][0])
        y1 = int(colorMarkers[0][0][1])
        x2 = int(colorMarkers[1][0][0])
        y2 = int(colorMarkers[1][0][1])
        center = calculateCenter(x1, y1, x2, y2)

        innerRectangleXIni = center[0] - 25
        innerRectangleYIni = center[1] - 25
        innerRectangleXFin = center[0] + 25
        innerRectangleYFin = center[1] + 25
        roi = colorframe[innerRectangleYIni +
                    1:innerRectangleYFin, innerRectangleXIni +
                                          1:innerRectangleXFin]
        hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = np.array(
            [hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
        upper = np.array(
            [hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])
        h = hsvRoi[:, :, 0]
        s = hsvRoi[:, :, 1]
        v = hsvRoi[:, :, 2]
        hAverage = np.average(h)
        sAverage = np.average(s)
        vAverage = np.average(v)

        hMaxSensibility = max(abs(lower[0] - hAverage), abs(upper[0] - hAverage))
        sMaxSensibility = max(abs(lower[1] - sAverage), abs(upper[1] - sAverage))
        vMaxSensibility = max(abs(lower[2] - vAverage), abs(upper[2] - vAverage))

        lower_pink = np.array([hAverage - hMaxSensibility, sAverage - sMaxSensibility, vAverage - vMaxSensibility])
        upper_pink = np.array([hAverage + hMaxSensibility, sAverage + sMaxSensibility, vAverage + vMaxSensibility])
        cv2.rectangle(colorframe, (innerRectangleXIni, innerRectangleYIni),
                      (innerRectangleXFin, innerRectangleYFin), (255, 0, 0), 0)

    return lower_pink, upper_pink