
import numpy as np
import cv2#, PIL
from cv2 import aruco
from classes.realsense import RealSense

# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/Aruco.html

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

def main():

    device = RealSense("21312312312")
    print("Color intrinsics: ", device.getcolorintrinsics())
    print("Depth intrinsics: ", device.getdepthintrinsics())
    flag = True

    try:
        while True:
            image = device.getcolorstream()
            if flag:
                crop_img = image
            #image = cv2.imread("../tests/aruco.jpg")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

            #print(corners)

            calibrationMatrix = []
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == 0:
                        c = corners[i][0]
                        #cv2.circle(frame_markers, (c[0, 0], c[0, 1]), 2, (0, 255, 0), 2)
                        #print(c[0, ])
                        calibrationMatrix.append([c[0, 0], c[0, 1]])

                print(len(calibrationMatrix))
                if len(calibrationMatrix)==4:
                    minx = min((calibrationMatrix[0][0], calibrationMatrix[1][0], calibrationMatrix[2][0], calibrationMatrix[3][0]))
                    miny = min((calibrationMatrix[0][1], calibrationMatrix[1][1], calibrationMatrix[2][1], calibrationMatrix[3][1]))
                    maxx = max((calibrationMatrix[0][0], calibrationMatrix[1][0], calibrationMatrix[2][0], calibrationMatrix[3][0]))
                    maxy = max((calibrationMatrix[0][1], calibrationMatrix[1][1], calibrationMatrix[2][1], calibrationMatrix[3][1]))
                    cv2.circle(frame_markers, (int(miny),int(miny + (maxy - miny))), 2, (0, 255, 0), 2)
                    cv2.circle(frame_markers, (c[0, 0], c[0, 1]), 2, (0, 255, 0), 2)
                    cv2.circle(frame_markers, (c[0, 0], c[0, 1]), 2, (0, 255, 0), 2)
                    cv2.circle(frame_markers, (c[0, 0], c[0, 1]), 2, (0, 255, 0), 2)
                    crop_img = image[int(miny):int(miny + (maxy - miny)), int(minx):int(minx + (maxx - minx))]
                    flag = False

                    cv2.resize(image, (848, 480))

            images_H1 = np.hstack((cv2.resize(image, (848, 480)), cv2.resize(crop_img, (848, 480))))
            images_H2 = np.hstack((cv2.resize(image, (848, 480)), cv2.resize(frame_markers, (848, 480))))
            images = np.vstack((images_H1, images_H2))
            cv2.imshow('ARUCO', images)
            if len(calibrationMatrix) == 4:
                #wait = input("Press Enter to continue.")
                pass
            cv2.waitKey(1)
            #cv2.destroyAllWindows()

    finally:
        # Stop streaming
        device.stop()








if __name__ == '__main__':
    main()

