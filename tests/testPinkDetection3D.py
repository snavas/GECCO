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

lower_pink = np.array([110, 80, 80])
upper_pink = np.array([170, 255, 255])

def getUpperLowerSquare(colorMarkers, colorframe):
    x1 = int(colorMarkers[0][0][0])
    y1 = int(colorMarkers[0][0][1])
    x2 = int(colorMarkers[1][0][0])
    y2 = int(colorMarkers[1][0][1])
    center = calculateCenter(x1, y1, x2, y2)
    cv2.putText(colorframe, "x", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1,
                cv2.LINE_AA)

    innerRectangleXIni = center[0] - 15
    innerRectangleYIni = center[1] - 15
    innerRectangleXFin = center[0] + 15
    innerRectangleYFin = center[1] + 15
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

    average = np.array([np.average(h), np.average(s), np.average(v)])
    maxSense = np.array([max(abs(lower[0] - average[0]), abs(upper[0] - average[0])),
                         max(abs(lower[1] - average[1]), abs(upper[1] - average[1])),
                         max(abs(lower[2] - average[2]), abs(upper[2] - average[2]))])
    cv2.rectangle(colorframe, (innerRectangleXIni, innerRectangleYIni),
                  (innerRectangleXFin, innerRectangleYFin), (255, 0, 0), 0)
    return average, maxSense, colorframe
def getUpperLowerCircle(colorMarkers, colorframe):
    x1 = int(colorMarkers[0][0][0])
    y1 = int(colorMarkers[0][0][1])
    x2 = int(colorMarkers[1][0][0])
    y2 = int(colorMarkers[1][0][1])
    center = calculateCenter(x1, y1, x2, y2)

    r = 15
    mask = np.zeros(colorframe.shape[:2], dtype="uint8")
    cv2.circle(mask, (center[0], center[1]), r, 255, -1)
    roi = cv2.bitwise_and(colorframe, colorframe, mask=mask)
    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    upper = np.array(
        [hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])
    hsvRoi[np.where((hsvRoi == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    lower = np.array(
        [hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
    h = hsvRoi[:, :, 0]
    s = hsvRoi[:, :, 1]
    v = hsvRoi[:, :, 2]

    average = np.array([np.average(h), np.average(s), np.average(v)])
    maxSense = np.array([max(abs(lower[0] - average[0]), abs(upper[0] - average[0])),
                         max(abs(lower[1] - average[1]), abs(upper[1] - average[1])),
                         max(abs(lower[2] - average[2]), abs(upper[2] - average[2]))])
    cv2.circle(colorframe, (center[0], center[1]),
                  15, (255, 0, 0), 0)
    return average, maxSense, colorframe

def main():
    device = RealSense('752112070204', False)
    file = False
    #print("Color intrinsics: ", device.getcolorintrinsics())
    #print("Depth intrinsics: ", device.getdepthintrinsics())
    # Initiate ORB detector
    orb = cv2.ORB_create()
    flag = 500
    try:
        while True:
            image = device.getcolorstream()
            if file:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
            parameters = aruco.DetectorParameters_create()
            parameters.errorCorrectionRate = 0.7
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            colorframe = aruco.drawDetectedMarkers(image.copy(), corners, ids)

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
                averageA, maxSenseA, colorframe = getUpperLowerCircle(colorMarkersA, colorframe)
                lower_pinkA = np.array(
                    [averageA[0] - maxSenseA[0], averageA[1] - maxSenseA[1], averageA[2] - maxSenseA[2]])
                upper_pinkA = np.array(
                    [averageA[0] + maxSenseA[0], averageA[1] + maxSenseA[1], averageA[2] + maxSenseA[2]])

                averageA, maxSenseB, colorframe = getUpperLowerCircle(colorMarkersB, colorframe)
                lower_pinkB = np.array(
                    [averageA[0] - maxSenseB[0], averageA[1] - maxSenseB[1], averageA[2] - maxSenseB[2]])
                upper_pinkB = np.array(
                    [averageA[0] + maxSenseB[0], averageA[1] + maxSenseB[1], averageA[2] + maxSenseB[2]])
                lower_pink = np.array([min(lower_pinkA[0], lower_pinkB[0]),
                                       min(lower_pinkA[1], lower_pinkB[1]),
                                       min(lower_pinkA[2], lower_pinkB[2])])
                upper_pink = np.array([max(upper_pinkA[0], upper_pinkB[0]),
                                       max(upper_pinkA[1], upper_pinkB[1]),
                                       max(upper_pinkA[2], upper_pinkB[2])])
                cv2.putText(colorframe, "color calibration markers detected", (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,
                            (255, 0, 0), 1,
                            cv2.LINE_AA)

            elif len(colorMarkersA) == 2:
                average, maxSense, colorframe = getUpperLowerCircle(colorMarkersA, colorframe)
                lower_pink = np.array(
                    [average[0] - maxSense[0], average[1] - maxSense[1], average[2] - maxSense[2]])
                upper_pink = np.array(
                    [average[0] + maxSense[0], average[1] + maxSense[1], average[2] + maxSense[2]])
                cv2.putText(colorframe, "color calibration markers detected", (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,
                            (255, 0, 0), 1,
                            cv2.LINE_AA)
            elif len(colorMarkersB) == 2:
                average, maxSense, colorframe = getUpperLowerCircle(colorMarkersB, colorframe)
                lower_pink = np.array(
                    [average[0] - maxSense[0], average[1] - maxSense[1], average[2] - maxSense[2]])
                upper_pink = np.array(
                    [average[0] + maxSense[0], average[1] + maxSense[1], average[2] + maxSense[2]])
                cv2.putText(colorframe, "color calibration markers detected", (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,
                            (255, 0, 0), 1,
                            cv2.LINE_AA)
            else:
                cv2.putText(colorframe, "NO COLOR DETECTION", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255),
                            1,
                            cv2.LINE_AA)




            #cv2.imshow('Detected Color', image)
            # Show images
            cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.imshow('Output Frame', colorframe)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()
        pass

if __name__ == '__main__':
    main()