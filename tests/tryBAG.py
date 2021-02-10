## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#import pyrealsense2 as rs
#import numpy as np
import hashlib

import cv2
from classes.realsense import RealSense
from classes.objloader import *
import copy
import numpy as np
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import libs.utils as utils
import matplotlib.pyplot as plt
#import screeninfo
import math
import time

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle(vector1, vector2):
    length1 = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1])
    length2 = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1])
    return math.acos((vector1[0] * vector2[0] + vector1[1] * vector2[1])/ (length1 * length2))

memory = {}
def id_to_random_color(number):
    if number == -1:
        return (0,0,0)
    elif number == 0:
        return (31,120,180)
    elif number == 1:
        return(178,223,138)
    elif number == 2:
        return(51,160,44)
    elif number == 3:
        return(251,154,153)
    elif number == 4:
        return(227,26,28)
    elif number == 5:
        return(253,191,111)
    elif number == 6:
        return(255,127,0)
    elif number == 7:
        return(202,178,214)
    elif number == 8:
        return(106,61,154)
    elif number == 9:
        return (166,206,227)
    elif number == 10:
        return (31,120,180)
    elif number == 11:
        return(178,223,138)
    elif number == 12:
        return(51,160,44)
    elif number == 13:
        return(251,154,153)
    elif number == 14:
        return(227,26,28)
    elif number == 15:
        return(253,191,111)
    elif number == 16:
        return(255,127,0)
    elif number == 17:
        return(202,178,214)
    elif number == 18:
        return(106,61,154)
    else:
        return(255,255,255)

def main():
    device = RealSense("1234")
    #print("Color intrinsics: ", device.getcolorintrinsics())
    #print("Depth intrinsics: ", device.getdepthintrinsics())
    # Initiate ORB detector
    orb = cv2.ORB_create()
    flag = 500
    try:
        while True:
            image = device.getcolorstream()
            depth = device.getdepthstream()
            #image = cv2.imread("D:/Users/s_nava02/Desktop/raw_output.png")
            screenshot = image.copy()
            if flag == 0:
                cv2.imwrite("C:/GECCO/raw_output.png", screenshot)
            flag -= 1
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
            # res = cv2.bitwise_and(colorframe, colorframe, mask=mask)
            # remove noise
            # imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            blurred = cv2.blur(mask, (5, 5))  # TODO: VERY BASIC, TRY OTHER FILTERS
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
            # ret, thresholded = cv2.threshold(blurred, 50, 255, 0)  # TODO: VERY BASIC, TRY OTHER THRESHHOLDS
            ret, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
            # th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            ######################
            # return tresholded
            ######################
            #cv2.imshow('RealSense', thresholded)

            ###################################################
            # getcontours(thresholded image):
            ###################################################
            mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
            method = cv2.CHAIN_APPROX_SIMPLE
            hand_contours = []
            contours, hierarchy = cv2.findContours(thresholded, mode, method)
            # contours = sorted(contours, key=cv2.contourArea)  # TODO: is this really necessary?
            for c in contours:
                # If contours are bigger than a certain area we push them to the array
                if cv2.contourArea(c) > 3000:
                    hand_contours.append(c)
                    #print("contour found")
            #####################
            # return hand_contours
            #####################

            # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
            ###################################################
            # Get Rough Hull
            ###################################################
            # TODO: try to not compute convex hull twice
            # https://stackoverflow.com/questions/52099356/opencvconvexitydefects-on-largest-contour-gives-error
            for cnt in hand_contours:
                hull = cv2.convexHull(cnt)
                index = cv2.convexHull(cnt, returnPoints=False)
                # cv2.drawContours(image, cnt, 0, (255, 255, 0), 2)
                # cv2.drawContours(image, hull_list, i, (0, 255, 0), 2)

                # TODO: different ways of grouping hull points into neigbours/clusters
                # term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
                # _ret, labels, centers = cv2.kmeans(np.float32(hull[:,0]), 6, None, term_crit, 10, 0)
                # point_tree = spatial.cKDTree(np.float32(hull[:,0]))
                # print("total points: ",len(np.float32(hull_list[i][:,0])), " - Total groups: ", point_tree.size)
                # neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
                # output = neigh.fit(hull[:,0])
                clustering = DBSCAN(eps=10, min_samples=1).fit(hull[:,0])
                #print(len(clustering.labels_))
                #print(hull_list[i])
                #print(clustering.labels_)
                #print(clustering.components_)
                rhull = np.column_stack((hull[:,0], index[:,0]))
                centers = utils.groupPointsbyLabels(rhull, clustering.labels_)
                defects = cv2.convexityDefects(cnt, np.array(centers)[:,2])
                c = 0
                for p in hull:
                    # print("init ", p, " - ")
                    # cv2.circle(image, tuple(p[0]), 10, id_to_random_color(clustering.labels_[c]))
                    c += 1
                #for p in centers:
                    #print("init ", p[0], " - ")
                    #cv2.circle(image, (p[0],p[1]), 4, (0, 255, 255))
                    #pass
                for p in centers:
                    # cv2.circle(image, (int(p[0]), int(p[1])), 4, (0, 255, 255))
                    pass
                ###############################################################
                # getHullDefectVertices
                ###############################################################
                # get neighbor defect points of each hull point
                hullPointDefectNeighbors = [] # 0: start, 1: end, 2:defect
                print("defects.shape[0]: ",defects.shape[0])
                for x in range(defects.shape[0]):
                    s, e, f, d = defects[x, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv2.line(image, start, end, [0, 255, 0], 1)
                    cv2.circle(image, far, 4, (0, 0, 255))
                    cv2.line(image, start, far, [255, 150, 0], 1)
                    cv2.line(image, end, far, [255, 150, 0], 1)
                    hullPointDefectNeighbors.append([start, end, far]) # each defect point (red) has its neihbour points (yellow)
                ###############################################################
                # filterVerticesByAngle
                ###############################################################
                #maxAngleDeg = 60
                maxAngleDeg = math.radians(60)
                i = 0
                fingers = []
                for triple in hullPointDefectNeighbors:
                    cf = triple[0] # candidate finger
                    rd = triple[2] # right deflect
                    if i == 0:     # left deflect
                        ld = hullPointDefectNeighbors[len(hullPointDefectNeighbors)-1][2]
                    else:
                        ld = hullPointDefectNeighbors[i-1][2]
                    # alternative maths
                    v_cp_ld = (ld[0] - cf[0], ld[1] - cf[1])
                    v_cp_rd = (rd[0] - cf[0], rd[1] - cf[1])
                    beta = angle_between(v_cp_ld, v_cp_rd)
                    print(beta)
                    cv2.circle(image, (cf[0], cf[1]), 4, (0, 0, 255)) # candidate finger: red
                    cv2.circle(image, (rd[0], rd[1]), 4, (255, 0, 0)) # right defect: blue
                    cv2.circle(image, (ld[0], ld[1]), 4, (255, 0, 0)) # left defect: blue
                    if beta<maxAngleDeg:
                        fingers.append(cf)
                    # old maths
                    #if (math.atan2(cf[1] - rd[1], cf[0] - rd[0]) < maxAngleDeg) and (
                    #            math.atan2(cf[1] - ld[1], cf[0] - ld[0]) < maxAngleDeg) and len(fingers) < 5:
                    #    fingers.append(triple[0])
                    i += 1
                print(len(fingers))
                for f in fingers:
                    cv2.circle(image, (f[0], f[1]), 4, (255, 255, 255)) # identified finger: white
                    print("image size: ", image.shape)
                    print("color pixel value of ", f, ":", image[f[0]][f[1]])
                    pass

            # Show images
            cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.imshow('Output Frame', image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()
        pass

if __name__ == '__main__':
    main()