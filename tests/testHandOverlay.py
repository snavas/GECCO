## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
import hashlib

import os

import cv2
from cv2 import aruco
from classes.realsense import RealSense
from classes.objloader import *
import copy
import numpy as np
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import libs.utils as utils
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#import screeninfo
import math
import time

import win32api
import win32con
import win32gui
from win32api import GetSystemMetrics

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
marker23 = cv2.imread("6x6_1000-0.png")

def main():
    device = RealSense('C:/Users/s_nava02/sciebo/GECCO/pinktest.bag')
    # device = RealSense("752112070204")
    file = False
    #print("Color intrinsics: ", device.getcolorintrinsics())
    #print("Depth intrinsics: ", device.getdepthintrinsics())
    # Initiate ORB detector
    orb = cv2.ORB_create()
    flag = -1
    aruco_markers = True
    try:
        while True:
            image = device.getcolorstream()
            if file:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            depth = device.getdepthstream()
            #image = cv2.imread("../material/raw_output.png")
            #screenshot = image.copy()
            #if flag == 0:
            #    cv2.imwrite("../material/raw_output.png", screenshot)
            #flag -= 1
            ###################################################
            # def gethandmask(colorframe image):
            ###################################################
            # Convert BGR to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # define range of blue color in HSV // (145, 100, 20), (155, 255, 255)
            # Todo: From RCARAP (IDK why it works so differently 'bad')
            #lower_pink = np.array([140, 0.1 * 255, 0.05 * 255])
            #upper_pink = np.array([170, 0.8 * 255, 0.6 * 255])
            # Todo: New approach, still not working as good as javascript RCARAP, it needs to be refined later
            lower_pink = np.array([130, 100, 100])
            upper_pink = np.array([170, 255, 255])
            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_pink, upper_pink)
            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(image, image, mask=mask)
            # The result above has black background, but we want it green to convert it to transparent overlay
            res[np.all(res == (0, 0, 0), axis=-1)] = (0, 255, 0)

            # Drawing red squares (or QR codes) for calibration
            msize = 50  # marker size
            height, width, channels = res.shape
            print(height," ", width)
            if not aruco_markers:
                cv2.rectangle(res, (0, 0), (msize, msize), (0, 0, 255), -1) # top left
                cv2.rectangle(res, (width, 0), (width-msize, msize), (0, 0, 255), -1) # top right
                cv2.rectangle(res, (0, height), (msize, height-msize), (0, 0, 255), -1) # bot left
                cv2.rectangle(res, (width, height), (width-msize, height-msize), (0, 0, 255), -1) # bot right
            else:
                # TODO: It would be better to generate the ARUCO  using the dictionary instead of loading them from file (and resizing)
                # tl = aruco.drawMarker(aruco_dict,1, msize) #this doesn't work, because tl dimension
                # print(res.shape)
                # print(tl.shape)
                y_offset = height - msize
                x_offset = width - msize
                tl = cv2.resize(marker23, (msize,msize), interpolation = cv2.INTER_NEAREST)
                res[0:0 + tl.shape[0], 0:0 + tl.shape[1]] = tl
                res[0:0 + tl.shape[0], x_offset:x_offset + tl.shape[1]] = tl
                res[y_offset:y_offset + tl.shape[0], 0:0 + tl.shape[1]] = tl
                res[y_offset:y_offset+tl.shape[0], x_offset:x_offset+tl.shape[1]] = tl

            # Show images
            cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
            #os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')  # To make window active
            cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Output Frame', res)

            # Overlay - It only works on windows
            if True:
                hwnd = win32gui.FindWindow(None, "Output Frame")
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED) # no idea, but it goes together with transparency
                win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,255,0), 0, win32con.LWA_COLORKEY) # black as transparent
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, -5, -5, GetSystemMetrics(0), GetSystemMetrics(1), 0) # always on top
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE) # maximiced

            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()
        pass

if __name__ == '__main__':
    main()