## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
from classes.realsense import RealSense
from classes.objloader import *
import copy
import numpy as np
import cv2
import os
#import screeninfo

CV_PI = 3.1415926535897932384626433832795

def main():
    device = RealSense(21312312312)
    print("Color intrinsics: ", device.getcolorintrinsics())
    print("Depth intrinsics: ", device.getdepthintrinsics())

    try:
        while True:
            #image2 = device.getdepthstream()
            #image2 = cv2.applyColorMap(cv2.convertScaleAbs(image2, alpha=0.03), cv2.COLORMAP_BONE)
            image1 = device.getcolorstream()
            cv2.imwrite("../raw_output.png", image1)
            image2 = copy.deepcopy(image1)
            image3 = copy.deepcopy(image1)
            image4 = copy.deepcopy(image1)
            # Color Extraction + Shape identification
            # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
            # ORANGE
            hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
            lower_orange = np.array([0, 0, 0],np.uint8)
            upper_orange = np.array([255, 150, 150],np.uint8)
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            res = cv2.bitwise_and(image2, image2, mask=mask)
            imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(imgray, (5, 5), 0)  # TODO: VERY BASIC, TRY OTHER FILTERS
            #sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            #sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
            ret, thresholded = cv2.threshold(blurred, 50, 255, 0)  # TODO: VERY BASIC, TRY OTHER THRESHHOLDS
            #thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
            contours, h = cv2.findContours(thresholded, 1, 2)
            #src_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            #src_gray = cv2.blur(src_gray, (3, 3))
            #canny_output = cv2.Canny(src_gray, 100, 100 * 2)
            #contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            squares = []
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                if (len(approx) == 4) & (cv2.contourArea(cnt)>25):
                    #x,y,w,h = cv2.boundingRect(cnt)
                    #cv2.rectangle(image1, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    #cv2.drawContours(image1, [cnt], 0, (0, 0, 255), -1)
                    contour_poly = cv2.approxPolyDP(cnt, 3, True)
                    #boundRect = cv2.boundingRect(contour_poly)
                    center, radius = cv2.minEnclosingCircle(contour_poly)
                    color=(0,255,255)
                    #cv2.drawContours(image1, contour_poly, 1, color)
                    #cv2.rectangle(image1, (int(boundRect[0]), int(boundRect[1])), \ (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), color, 2)
                    cv2.circle(image1, (int(center[0]), int(center[1])), int(radius), color, 2)
                    squares.append(center)
            #if len(squares) ==  3:
            #    cv2.putText(image1, "CALIBRATED: Detected 3 squares", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #else:
            #    cv2.putText(image1, "ERROR: NOT-CALIBRATED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            image2 = res
            # RED
            hsv = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([0, 50, 50])
            upper_yellow = np.array([5, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            res = cv2.bitwise_and(image3, image3, mask=mask)

            imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            #blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
            #ret, thresholded = cv2.threshold(blurred, 50, 255, 0)

            ret, thresholded = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)
            opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, (12,12))
            blurred = cv2.blur(opening,(4,4))

            contours, h = cv2.findContours(blurred, 1, 2)

            edges = cv2.Canny(thresholded, 66, 133, 3)
            lines = cv2.HoughLines(edges, 1, CV_PI/180, 50, 0, 0)

            # no big difference
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #close = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
            #contours, h = cv2.findContours(close, 1, 2)

            contours, h = cv2.findContours(thresholded, 1, 2)

            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                if (len(approx) == 4) & (cv2.contourArea(cnt)>25):
                    contour_poly = cv2.approxPolyDP(cnt, 3, True)
                    center, radius = cv2.minEnclosingCircle(contour_poly)
                    color=(0,0,255)
                    cv2.circle(image1, (int(center[0]), int(center[1])), int(radius), color, 2)

            image3 = res

            # green
            hsv = cv2.cvtColor(image4, cv2.COLOR_BGR2HSV)
            lower_green = np.array([100, 50, 50], np.uint8)
            upper_green = np.array([140, 255, 255], np.uint8)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            res = cv2.bitwise_and(image4, image4, mask=mask)

            imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
            ret, thresholded = cv2.threshold(blurred, 50, 255, 0)
            contours, h = cv2.findContours(thresholded, 1, 2)
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                if (len(approx) == 4) & (cv2.contourArea(cnt)>25):
                    contour_poly = cv2.approxPolyDP(cnt, 3, True)
                    center, radius = cv2.minEnclosingCircle(contour_poly)
                    color=(255,0,0)
                    cv2.circle(image1, (int(center[0]), int(center[1])), int(radius), color, 2)

            image4 = res
            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RealSense', cv2.WND_PROP_FULLSCREEN)
            #screen_id = 2
            #screen = screeninfo.get_monitors()[1]
            #cv2.moveWindow('RealSense', screen.x - 1, screen.y - 1)
            cv2.setWindowProperty("RealSense", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            images_H1 = np.hstack((image1, image2))
            images_H2 = np.hstack((image3, image4))
            images = np.vstack((images_H1, images_H2))
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()

if __name__ == '__main__':
    main()