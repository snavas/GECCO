import cv2
from cv2 import aruco
import numpy as np

def calibrateViaRedSquares(originalframe, segmentedframe, matrix):
    hsv = cv2.cvtColor(originalframe, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([0, 50, 50])
    upper_yellow = np.array([5, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    ## old approach
    # edges = cv2.Canny(mask, 100, 255, 3)
    # dilated = cv2.dilate(edges, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    # blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
    # ret, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    ## new try
    res = cv2.bitwise_and(originalframe, originalframe, mask=mask)
    imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, (20, 20))
    blurred = cv2.blur(opening, (4, 4))

    # https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    bw_nlbls, bw_lbls, bw_stats, bw_cent = cv2.connectedComponentsWithStats(blurred, 4, cv2.CV_32S)

    squares = []
    minPixSize = 900
    for i in range(0, bw_nlbls):
        x1 = bw_stats[i, cv2.CC_STAT_LEFT]
        y1 = bw_stats[i, cv2.CC_STAT_TOP]
        x2 = x1 + bw_stats[i, cv2.CC_STAT_WIDTH]
        y2 = y1 + bw_stats[i, cv2.CC_STAT_HEIGHT]
        size = bw_stats[i, cv2.CC_STAT_AREA]
        if (size > minPixSize and size < 2000):
            center, radius = cv2.minEnclosingCircle(np.array([(x1, y1), (x1, y2), (x2, y1), (x2, y2)]))
            color = (0, 0, 255)
            cv2.circle(segmentedframe, (x1, y1), 2, color, 2)
            cv2.circle(segmentedframe, (x1, y2), 2, color, 2)
            cv2.circle(segmentedframe, (x2, y1), 2, color, 2)
            cv2.circle(segmentedframe, (x2, y2), 2, color, 2)
            color = (0, 255, 0)
            cv2.circle(segmentedframe, (int(center[0]), int(center[1])), int(radius), color, 2)
            squares.append(center)
    if len(squares) == 4:
        cv2.putText(segmentedframe, "CALIBRATED (4)", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1,
                    cv2.LINE_AA)
        matrix = squares
    elif len(matrix) == 4:
        cv2.putText(segmentedframe, "CALIBRATED (OLD)", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(segmentedframe, "NOT CALIBRATED", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
    return segmentedframe, matrix



def calibrateViaARUco(originalframe, screen_corners, target_corners):
    gray = cv2.cvtColor(originalframe, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(np.zeros(originalframe.shape), corners, ids)

    calibrationMatrix = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 0:
                c = corners[i][0]
                calibrationMatrix.append([c[0, 0], c[0, 1]])

    if len(calibrationMatrix) == 4:
        num_rows, num_cols, _ = originalframe.shape
        screen_corners = np.array([
            [num_cols, num_rows],
            [num_cols, 0],
            [0, num_rows],
            [0, 0]
        ], np.float32)

        # TODO: choose different aruco codes for every corner instead of ordering like this
        b = np.sum(calibrationMatrix, axis=1)
        idx = (-b).argsort()
        target_corners = np.array(np.take(calibrationMatrix, idx, axis=0), np.float32)
    return frame_markers, screen_corners, target_corners