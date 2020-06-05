## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
from classes.realsense import RealSense
from classes.objloader import *
import copy
import numpy as np
import cv2
#import screeninfo

def main():
    device = RealSense(21312312312)
    print("Color intrinsics: ", device.getcolorintrinsics())
    print("Depth intrinsics: ", device.getdepthintrinsics())
    # Initiate QR detector
    qrCodeDetector = cv2.QRCodeDetector()

    try:
        while True:
            #image2 = device.getdepthstream()
            #image2 = cv2.applyColorMap(cv2.convertScaleAbs(image2, alpha=0.03), cv2.COLORMAP_BONE)
            image1 = device.getcolorstream()
            image2 = copy.deepcopy(image1)
            image3 = device.getdepthcolormap()
            image4 = device.getsegmentedstream()
            # QR extraction
            # (single) https://techtutorialsx.com/2019/12/08/python-opencv-detecting-and-decoding-a-qrcode/
            # (multiple) https://github.com/opencv/opencv/issues/13311
            # (tips to better detect) https://github.com/MikhailGordeev/QR-Code-Extractor
            codes, image2 = reader.extract(image2)
            if codes is not None:
                print("start")
                print(codes)
                print("end")
            #if (qrCodeDetector.detectMulti(image2, points)):
            #    print(points)
            # decodedText, points, _ = qrCodeDetector.detectAndDecode(image2)
            # if points is not None:
            #    nrOfPoints = len(points)
            #    for i in range(nrOfPoints):
            #        nextPointIndex = (i + 1) % nrOfPoints
            #        cv2.line(image2, tuple(points[i][0]), tuple(points[nextPointIndex][0]), (255, 0, 0), 5)
                #print(decodedText, points)
            else:
                print("QR code not detected")
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