## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
import hashlib

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

# import pyrealsense2 as rs
# import numpy as np
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
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import screeninfo
import math
import time

import win32api
import win32con
import win32gui
from win32api import GetSystemMetrics

res = cv2.imread("green.png")
marker0 = cv2.imread("6x6_1000-0.png")

# Drawing red squares (or QR codes) for calibration
msize = 300  # marker size
height, width, channels = res.shape

y_offset = height - msize
x_offset = width - msize

tl = cv2.resize(marker0, (msize,msize), interpolation = cv2.INTER_NEAREST)
res[0:0 + tl.shape[0], 0:0 + tl.shape[1]] = tl
res[0:0 + tl.shape[0], x_offset:x_offset + tl.shape[1]] = tl
res[y_offset:y_offset + tl.shape[0], 0:0 + tl.shape[1]] = tl
res[y_offset:y_offset+tl.shape[0], x_offset:x_offset+tl.shape[1]] = tl

cv2.namedWindow("Squares Overlay", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Squares Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Squares Overlay", res)

hwnd = win32gui.FindWindow(None, "Squares Overlay")
win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)  # no idea, but it goes together with transparency
win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 255, 0), 0, win32con.LWA_COLORKEY)  # black as transparent
win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, -5, -5, GetSystemMetrics(0), GetSystemMetrics(1), 0)  # always on top
win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)  # maximiced

cv2.waitKey(-1)