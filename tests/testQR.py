## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
from classes.realsense import RealSense
import copy
import numpy as np
import cv2
#import screeninfo

def main():
    device = RealSense('752112070399')
    #print("Color intrinsics: ", device.getcolorintrinsics())
    #print("Depth intrinsics: ", device.getdepthintrinsics())
    # Initiate QR detector
    qrCodeDetector = cv2.QRCodeDetector()

    try:
        while True:
            image = device.getcolorstream()
            qrCodeDetector = cv2.QRCodeDetector()
            decodedText, points, _ = qrCodeDetector.detectAndDecode(image)

            if points is not None:

                nrOfPoints = len(points)

                for i in range(nrOfPoints):
                    nextPointIndex = (i + 1) % nrOfPoints
                    cv2.line(image, tuple(points[i][0]), tuple(points[nextPointIndex][0]), (255, 0, 0), 5)

                print(decodedText)

                cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.imshow('Output Frame', image)
                cv2.waitKey(1)

            else:
                print("QR code not detected")

            #print(decodedText, points)
            # QR extraction
            # (single) https://techtutorialsx.com/2019/12/08/python-opencv-detecting-and-decoding-a-qrcode/
            # (multiple) https://github.com/opencv/opencv/issues/13311
            # (tips to better detect) https://github.com/MikhailGordeev/QR-Code-Extractor
            #codes, image2 = reader.extract(image2)
            #if codes is not None:
            #    print("start")
            #    print(codes)
            #    print("end")
            #if (qrCodeDetector.detectMulti(image2, points)):
            #    print(points)
            # decodedText, points, _ = qrCodeDetector.detectAndDecode(image2)
            # if points is not None:
            #    nrOfPoints = len(points)
            #    for i in range(nrOfPoints):
            #        nextPointIndex = (i + 1) % nrOfPoints
            #        cv2.line(image2, tuple(points[i][0]), tuple(points[nextPointIndex][0]), (255, 0, 0), 5)
                #print(decodedText, points)
            #else:
            #    print("QR code not detected")
            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    finally:
        # Stop streaming
        device.stop()

if __name__ == '__main__':
    main()