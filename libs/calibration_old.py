import cv2
import numpy as np

def getDraw(originalframe, segmentedframe, matrix):
    hsv = cv2.cvtColor(originalframe, cv2.COLOR_BGR2HSV)
    #lower_orange = np.array([5, 50, 50], np.uint8)
    #upper_orange = np.array([15, 255, 255], np.uint8)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([5, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)



    res = cv2.bitwise_and(originalframe, originalframe, mask=mask)
    imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray, (5, 5), 0)  # TODO: VERY BASIC, TRY OTHER FILTERS
    ret, thresholded = cv2.threshold(blurred, 50, 255, 0)  # TODO: VERY BASIC, TRY OTHER THRESHHOLDS
    contours, h = cv2.findContours(thresholded, 1, 2)
    squares = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if (len(approx) == 4) & (cv2.contourArea(cnt) > 100):
            contour_poly = cv2.approxPolyDP(cnt, 3, True)
            center, radius = cv2.minEnclosingCircle(contour_poly)
            color = (0, 0, 255)
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



