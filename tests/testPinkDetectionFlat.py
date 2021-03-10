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

def main():
    device = RealSense('752112070204')
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
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            image = aruco.drawDetectedMarkers(image.copy(), corners, ids)

            colorMarkers = []
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == 10:
                        c = corners[i][0]
                        colorMarkers.append(c)

            if len(colorMarkers) == 2:
                x1 = int(colorMarkers[0][0][0])
                y1 = int(colorMarkers[0][0][1])
                x2 = int(colorMarkers[1][0][0])
                y2 = int(colorMarkers[1][0][1])
                center = calculateCenter(x1, y1, x2, y2)
                colorR = int(image[center[1], center[0]][2])
                colorG = int(image[center[1], center[0]][1])
                colorB = int(image[center[1], center[0]][0])
                print("color:", image[center[1], center[0]])
                cv2.putText(image, "x", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(image, "color calibration markers detected", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                            (colorB, colorG, colorR), 1,
                            cv2.LINE_AA)
            else:
                cv2.putText(image, "NO COLOR DETECTION", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1,
                            cv2.LINE_AA)

            #cv2.imshow('Detected Color', image)
            # Show images
            cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.imshow('Output Frame', image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()
        pass

if __name__ == '__main__':
    main()