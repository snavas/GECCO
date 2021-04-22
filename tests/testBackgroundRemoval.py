import cv2
from cv2 import aruco
import numpy as np
import math
from classes.realsense import RealSense
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

def main():
    device = RealSense('../material/new_cali.bag')
    try:
        while True:
            frame = device.getcolorstream()

            fgMask = backSub.apply(frame)

            ret, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

            # cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask', fgMask)

            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    finally:
        # Stop streaming
        device.stop()
        pass


if __name__ == '__main__':
    main()