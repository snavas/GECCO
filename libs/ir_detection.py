import cv2
import numpy as np

def detect(irframe, colorframe):
    mask = cv2.inRange(irframe, 255, 255)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    blurred = cv2.blur(mask, (5, 5))  # TODO: VERY BASIC, TRY OTHER FILTERS
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # ret, thresholded = cv2.threshold(blurred, 50, 255, 0)  # TODO: VERY BASIC, TRY OTHER THRESHHOLDS
    ret, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    cX = -1
    cY = -1
    if (thresholded!=0).any():
        # calculate moments of binary image
        M = cv2.moments(thresholded)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return (cX,cY)