import cv2
import numpy as np

# Source: https://medium.com/@muehler.v/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0

def getHand(colorframe, depthframe, depthscale):
    def gethandmask():
        ###################################################
        # Create hand mask
        ###################################################
        # Convert BGR to HSV
        hsv = cv2.cvtColor(colorframe, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV // (145, 100, 20), (155, 255, 255)
        lower_pink = np.array([130, 100, 100])
        upper_pink = np.array([170, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(colorframe, colorframe, mask=mask)
        # remove noise
        imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
        blurred = cv2.GaussianBlur(imgray, (5, 5), 0)  # TODO: VERY BASIC, TRY OTHER FILTERS
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        ret, thresholded = cv2.threshold(blurred, 50, 255, 0)  # TODO: VERY BASIC, TRY OTHER THRESHHOLDS
        # th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresholded

    def getcontours(img):
        ###################################################
        # Saving hand contours in an array
        ###################################################
        mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        hand_contours = []
        contours, hierarchy = cv2.findContours(img, mode, method)
        #contours = sorted(contours, key=cv2.contourArea)  # TODO: is this really necessary?
        for c in contours:
            # If contours are bigger than a certain area we push them to the array
            if cv2.contourArea(c) > 3000:
                hand_contours.append(c)
        return hand_contours

    def getFeatures(contour, max_distance):
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

    ###################################################
    # Function body
    ###################################################
    maxPointDist = 25;
    maxAngleDeg = 60;

    result = gethandmask()
    hands = getcontours(result)
    if hands:
        points = getFeatures(hands[0],maxPointDist)
    else:
        hands = False
        points = False

    return cv2.bitwise_and(colorframe, colorframe, mask = result), hands, points
