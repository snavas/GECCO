import cv2
from cv2 import aruco
import numpy as np

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
    gray = cv2.cvtColor(colorframe, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    image = aruco.drawDetectedMarkers(colorframe.copy(), corners, ids)

    colorMarkersUpper = []
    colorMarkersLower = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 10:
                c = corners[i][0]
                colorMarkersUpper.append(c)
            if ids[i] == 11:
                c = corners[i][0]
                colorMarkersLower.append(c)

    if len(colorMarkersUpper) == 2 & len(colorMarkersLower) == 2:
        lower_pink = getColorBetweenMarkers(image, colorMarkersLower)
        upper_pink = getColorBetweenMarkers(image, colorMarkersUpper)
    return lower_pink, upper_pink


def calibrateViaARUco(originalframe, segmentedframe, matrix):
    gray = cv2.cvtColor(originalframe, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(originalframe.copy(), corners, ids)

    calibrationMatrix = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 0:
                c = corners[i][0]
                calibrationMatrix.append([c[0, 0], c[0, 1]])

    if len(calibrationMatrix) == 4:
        matrix = calibrationMatrix
    return frame_markers, matrix

    # OLD CODE, IT DOESNT NOT MAKE SENSE ANYMORE
    if len(calibrationMatrix) == 4:
        cv2.putText(segmentedframe, "CALIBRATED (4)", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1,
                    cv2.LINE_AA)
        matrix = calibrationMatrix
    elif len(matrix) == 4:
        cv2.putText(segmentedframe, "CALIBRATED (OLD)", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(segmentedframe, "NOT CALIBRATED", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
    return segmentedframe, matrix


