
import numpy as np
import cv2
from cv2 import aruco

# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/Aruco.html

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

image = cv2.imread("../tests/aruco.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

for i in range(len(ids)):
    c = corners[i][0]
    #lt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
    print(c)

images_H1 = np.hstack((image, frame_markers))
images_H2 = np.hstack((image, frame_markers))
images = np.vstack((images_H1, images_H2))
cv2.imshow('ARUCO', images_H1)
cv2.waitKey(0)
cv2.destroyAllWindows()



