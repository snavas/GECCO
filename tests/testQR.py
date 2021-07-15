## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
from classes.realsense import RealSense
import copy
import numpy as np
import cv2
import time
#import screeninfo

def main():
    device = RealSense('821212062065', False)
    #print("Color intrinsics: ", device.getcolorintrinsics())
    #print("Depth intrinsics: ", device.getdepthintrinsics())
    # Initiate QR detector
    qrCodeDetector = cv2.QRCodeDetector()

    try:
        while True:

            image = device.getcolorstream()
            qrCodeDetector = cv2.QRCodeDetector()

            # ONLY ONE QR CODE WILL BE DETECTED AND DECODED
            timestamp = time.time()
            decoded_info, points, _ = qrCodeDetector.detectAndDecode(image)
            print(time.time() - timestamp)

            # MULTIPLE QR CODES WILL BE DETECTED AND DECODED (PERFORMANCE IS WORSE)
            #retval, decoded_info, points, straight_qrcode = qrCodeDetector.detectAndDecodeMulti(image)

            if points is not None:

                nrOfPoints = len(points)

                for i in range(nrOfPoints):
                    nextPointIndex = (i + 1) % nrOfPoints
                    cv2.line(image, tuple(points[i][0]), tuple(points[nextPointIndex][0]), (255, 0, 0), 5)

                print(decoded_info)

            else:
                pass
                #print("QR code not detected")

            cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.imshow('Output Frame', image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()

if __name__ == '__main__':
    main()