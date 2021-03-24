import cv2
import numpy as np
import copy
import libs.detectColor as color

# Source: https://medium.com/@muehler.v/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0

import math
from sklearn.cluster import DBSCAN
import libs.utils as utils

# define pink range
lower_color = np.array([110, 80, 80])
upper_color = np.array([170, 255, 255])

def angle(vector1, vector2):
    length1 = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1])
    length2 = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1])
    return math.acos((vector1[0] * vector2[0] + vector1[1] * vector2[1])/ (length1 * length2))

def getHand(colorframe, uncaliColorframe):
    def gethandmask(img):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        global lower_color, upper_color
        lower_color,upper_color = color.detectcolor3D(uncaliColorframe, lower_color, upper_color)
        # Threshold the HSV image to get only color colors
        mask = cv2.inRange(hsv, lower_color, upper_color)
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
        return thresholded

    def getcontours(tresh):
        mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        hand_contours = []
        contours, hierarchy = cv2.findContours(tresh, mode, method)
        # contours = sorted(contours, key=cv2.contourArea)  # TODO: is this really necessary?
        for c in contours:
            # If contours are bigger than a certain area we push them to the array
            if cv2.contourArea(c) > 2500:
                hand_contours.append(c)
        return hand_contours

    def getRoughHull(cnt):
        # TODO: try to not compute convex hull twice
        # https://stackoverflow.com/questions/52099356/opencvconvexitydefects-on-largest-contour-gives-error
        hull = cv2.convexHull(cnt)
        index = cv2.convexHull(cnt, returnPoints=False)
        # TODO: different ways of grouping hull points into neigbours/clusters
        # term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        # _ret, labels, centers = cv2.kmeans(np.float32(hull[:,0]), 6, None, term_crit, 10, 0)
        # point_tree = spatial.cKDTree(np.float32(hull[:,0]))
        # print("total points: ",len(np.float32(hull_list[i][:,0])), " - Total groups: ", point_tree.size)
        # neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
        # output = neigh.fit(hull[:,0])
        clustering = DBSCAN(eps=5, min_samples=1).fit(hull[:, 0])
        rhull = np.column_stack((hull[:, 0], index[:, 0]))
        centers = utils.groupPointsbyLabels(rhull, clustering.labels_)
        centers = np.array(centers)[np.array(centers)[:, 2].argsort()]
        try:
            defects = cv2.convexityDefects(cnt, centers[:, 2])
        except Exception as e:
            print("convexity defects could not be determined")
            defects = np.array([])
        return defects

    def getHullVertices(defects, contour):
        hullPointDefectNeighbors = []  # 0: start, 1: end, 2:defect
        for x in range(defects.shape[0]):
            s, e, f, d = defects[x, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            hullPointDefectNeighbors.append([start, end, far])
        return hullPointDefectNeighbors

    def filterVerticesByAngle(neihbors):
        maxAngleDeg = math.radians(60)
        i = 0
        fingers = []
        for triple in neihbors:
            cf = triple[0]  # candidate finger
            rd = triple[2]  # right deflect
            if i == 0:  # left deflect
                ld = neihbors[len(neihbors) - 1][2]
            else:
                ld = neihbors[i - 1][2]
            v_cp_ld = (ld[0] - cf[0], ld[1] - cf[1])
            v_cp_rd = (rd[0] - cf[0], rd[1] - cf[1])
            beta = utils.angle_between(v_cp_ld, v_cp_rd)
            if beta < maxAngleDeg and len(fingers) < 5:
                fingers.append(cf)
            i += 1
        return fingers

    def getFeatures(contour, max_distance):
        # TODO: NOT USED
        ####################################################
        # Extract Hand Features
        ####################################################
        # https://github.com/SouravJohar/handy/blob/master/Hand.py
        ####################################################
        #hull = cv2.convexHull(contour)
        #hull_indices = contour.vertices
        #contourPoints = hull.poi
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        contour_points = cv2.convexHull(contour, returnPoints=True)
        x = []
        y = []
        #for pt in contour_points():
        for pt in range(len(contour_points)):
            x.append(contour_points[pt][0][0])
            y.append(contour_points[pt][0][1])
        z = np.vstack((x, y))
        z = np.float32(z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(z, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        return center.transpose()
        #return contour_points

    def getcontourmask(handmask, handcontours):
        arrayOut = []
        for contour in handcontours:
            mask = np.zeros_like(handmask)  # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Draw filled contour in mask
            tempOut = np.zeros_like(handmask)  # Extract out the object and place into output image
            tempOut[mask == 255] = handmask[mask == 255]
            arrayOut.append(tempOut)
        return arrayOut

    ###################################################
    # Function body
    ###################################################

    handMask = gethandmask(colorframe)  # hand mask
    handContours = getcontours(handMask)       # hand contours
    handMasks = getcontourmask(handMask, handContours)
    handList = []
    fingerList = []
    if handContours:
        for c in handContours:
            rHull = getRoughHull(c)
            vertices = getHullVertices(rHull, c)
            points = filterVerticesByAngle(vertices)
            handList.append(c)
            fingerList.append(points)

    else:
        handList = False
        fingerList = False

    colorframes = []
    for curMask in handMasks:
        temp = colorframe.copy()
        temp = cv2.bitwise_and(temp, temp, mask=curMask)
        colorframes.append(temp)
    return colorframes, handList, fingerList
