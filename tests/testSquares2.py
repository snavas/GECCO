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
            image = device.getcolorstream()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([0, 50, 50])
            upper_yellow = np.array([5, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            #res = cv2.bitwise_and(image, image, mask=mask)
            #imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(mask, 100, 255, 3)
            dilated = cv2.dilate(edges, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
            ret, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

            # https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
            bw_nlbls, bw_lbls, bw_stats, bw_cent = cv2.connectedComponentsWithStats(thresholded, 4, cv2.CV_32S)

            minPixSize = 900
            for i in range(0, bw_nlbls):
                x1 = bw_stats[i, cv2.CC_STAT_LEFT]
                y1 = bw_stats[i, cv2.CC_STAT_TOP]
                x2 = x1 + bw_stats[i, cv2.CC_STAT_WIDTH]
                y2 = y1 + bw_stats[i, cv2.CC_STAT_HEIGHT]
                size = bw_stats[i, cv2.CC_STAT_AREA ]
                if (size > minPixSize and size < 2000):
                    center, radius = cv2.minEnclosingCircle(np.array([(x1, y1), (x1, y2), (x2, y1), (x2, y2)]))
                    color = (0, 0, 255)
                    cv2.circle(image, (x1, y1), 2, color, 2)
                    cv2.circle(image, (x1, y2), 2, color, 2)
                    cv2.circle(image, (x2, y1), 2, color, 2)
                    cv2.circle(image, (x2, y2), 2, color, 2)
                    color = (0, 255, 0)
                    cv2.circle(image, (int(center[0]), int(center[1])), int(radius), color, 2)

            cv2.imshow('RealSense', image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()

if __name__ == '__main__':
    main()