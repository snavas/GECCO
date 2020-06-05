## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
import cv2
from classes.realsense import RealSense
from classes.objloader import *
import copy
import numpy as np
#import screeninfo

def main():
    device = RealSense("1234")
    print("Color intrinsics: ", device.getcolorintrinsics())
    print("Depth intrinsics: ", device.getdepthintrinsics())
    # Initiate ORB detector
    orb = cv2.ORB_create()
    try:
        while True:
            #image2 = device.getdepthstream()
            #image2 = cv2.applyColorMap(cv2.convertScaleAbs(image2, alpha=0.03), cv2.COLORMAP_BONE)
            image1 = device.getcolorstream()
            image2 = copy.deepcopy(image1)
            image3 = device.getdepthcolormap()
            image4 = device.getsegmentedstream()
            # Feature Extraction
            kp = orb.detect(image2, None) # find the keypoints with ORB
            kp, des = orb.compute(image2, kp) # compute the descriptors with ORB
            image2 = cv2.drawKeypoints(image2, kp, image2, color=(0, 255, 0), flags=0) # draw only keypoints location,not size and orientation
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