import cv2
import numpy as np
from classes.realsense import RealSense
import math
from sklearn.cluster import DBSCAN
import libs.utils as utils
import libs.detectColor as color
import libs.calibration as cal
import win32api
import win32con
import win32gui
from win32api import GetSystemMetrics
from vidgear.gears.helper import reducer


# define pink range
lower_color = np.array([110, 80, 80])
upper_color = np.array([170, 255, 255])


def angle(vector1, vector2):
    length1 = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1])
    length2 = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1])
    return math.acos((vector1[0] * vector2[0] + vector1[1] * vector2[1]) / (length1 * length2))


def getHand(colorframe, uncaliColorframe, colorspace, edges):
    def gethandmask(img):
        # Convert BGR to HSV
        colorConverted = cv2.cvtColor(img, colorspace)
        global lower_color, upper_color
        lower_color, upper_color = color.detectcolor3D(uncaliColorframe, lower_color, upper_color, colorspace)
        # Threshold the HSV image to get only color colors
        mask = cv2.inRange(colorConverted, lower_color, upper_color)
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

    def getcontourmask(handmask):
        mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        hand_contours = []
        contours, hierarchy = cv2.findContours(handmask, mode, method)
        hands = []
        for c in contours:
            # If contours are bigger than a certain area we push them to the array
            if cv2.contourArea(c) > 2500:
                hand_contours.append(c)
                mask = np.zeros_like(handmask)  # Create mask where white is what we want, black otherwise
                cv2.drawContours(mask, [c], -1, 255, -1)  # Draw filled contour in mask
                tempOut = np.zeros_like(handmask)  # Extract out the object and place into output image
                tempOut[mask == 255] = handmask[mask == 255]
                tempOut2 = cv2.dilate(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20)), iterations=3)

                rHull = getRoughHull(c)
                vertices = getHullVertices(rHull, c)
                points = filterVerticesByAngle(vertices)

                hand = {
                    "mask": tempOut,
                    "mask2": tempOut2,
                    "contour": c,
                    "fingers": points
                }
                # edge only mode
                if edges:
                    hand["dilated_masks"] = []
                    # Heavily dilated
                    tempOutDilBig = cv2.dilate(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 30)),
                                               iterations=3)
                    hand["dilated_masks"].append(tempOutDilBig)
                    # A little less dilated
                    tempOutDilSmol = cv2.dilate(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                                                iterations=1)
                    hand["dilated_masks"].append(tempOutDilSmol)
                hands.append(hand)
        return hands

    ###################################################
    # Function body
    ###################################################

    mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
    method = cv2.CHAIN_APPROX_SIMPLE
    handMask = gethandmask(colorframe)  # hand mask
    hands = getcontourmask(handMask) # divide the big mask into individual hand masks and contours
    for hand in hands:
        copy = colorframe.copy()
        # edge only mode
        if edges:
            # get a really dilated masked out hand, so that the edges dont have to be calculated for the entire image
            # hand_image = cv2.bitwise_and(copy, copy, mask=hand["dilated_masks"][0])
            # all = hand_image[:, :, 0] + hand_image[:, :, 1] + hand_image[:, :, 2]
            # hand_image[:, :, 0] = hand_image[:, :, 0] / all
            # hand_image[:, :, 1] = hand_image[:, :, 1] / all
            # hand_image[:, :, 2] = hand_image[:, :, 2] / all
            cnt, _ = cv2.findContours(hand["mask2"], mode, method)
            hand["hand_crop"] = []
            for ci in cnt:
                x, y, w, h = cv2.boundingRect(ci)
                crop_img = copy[y:y + h, x:x + w]
                crop_img = reducer(crop_img, percentage=40)  # reduce frame by 40%
                hand["hand_crop"].append({
                    "crop": crop_img,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })
            (left, width, top, height) = (hand["hand_crop"][0]["x"], hand["hand_crop"][0]["w"],
                                          hand["hand_crop"][0]["y"], hand["hand_crop"][0]["h"])

            hand_image = copy[top:top + height, left:left + width]

            # calculate edges
            canny_output = cv2.Canny(hand_image, 100, 200)
            # empty image
            hand_image = np.empty((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            hand_image.fill(255)
            # get contours of edges
            contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours into the empty image
            edge_color = np.mean(np.array([lower_color, upper_color]), axis=0)
            for i in range(len(contours)):
                cv2.drawContours(hand_image, contours, i, (254, 254, 254), 3, cv2.LINE_8, hierarchy, 0)

            hand_image = cv2.bitwise_not(hand_image)
            result = np.empty((copy.shape[0], copy.shape[1], 3), dtype=np.uint8)
            result[top:top + height, left:left + width] = hand_image
            for i in range(len(contours)):
                cv2.drawContours(result, contours, i, (254,254,254), 1, cv2.LINE_8, hierarchy, 0)
            # mask out the outer edges, that belong to the more heavily dilated mask
            # hand_image = cv2.bitwise_and(hand_image, hand_image, mask=hand["dilated_masks"][1])

            erodedMask = cv2.erode(hand["mask"], cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)), iterations=1)
            # Perform the distance transform algorithm
            dist = cv2.distanceTransform(erodedMask, cv2.DIST_L2, 3)
            # Normalize the distance image for range = {0.0, 1.0}
            # so we can visualize and threshold it
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

            # Threshold to obtain the peaks
            # This will be the markers for the foreground objects
            _, dist = cv2.threshold(dist, 0.2, 1.0, cv2.THRESH_BINARY)
            # Dilate a bit the dist image
            #kernel1 = np.ones((3, 3), dtype=np.uint8)
            #dist = cv2.dilate(dist, kernel1)

            # Create the CV_8U version of the distance image
            # It is needed for findContours()
            dist_8u = dist.astype('uint8')
            # Find total markers
            contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create the marker image for the watershed algorithm
            markers = np.zeros(dist.shape, dtype=np.int32)
            # Draw the foreground markers
            for i in range(len(contours)):
                cv2.drawContours(markers, contours, i, (i + 1), -1)
            background = cv2.bitwise_not(hand["dilated_masks"][0])
            markers = markers + background
            result = cv2.watershed(result, markers)
            result = result.astype(np.uint8)
            result = cv2.bitwise_not(result)
            #result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            result = cv2.bitwise_and(copy, copy, mask=result)

        # normal mode
        else:
            result = cv2.bitwise_and(copy, copy, mask=hand["dilated_masks"][1])
        hand["hand_image"] = result
    return hands

def main():
    device = RealSense('../material/new_cali.bag')
    try:
        screen_corners = []
        target_corners = []
        while True:
            ########################
            # Startup              #
            ########################
            # read frames
            colorframe = device.getcolorstream()
            # if fileFlag:
            #    colorframe = cv2.cvtColor(colorframe, cv2.COLOR_RGB2BGR) #Reading from BAG alters the color space and needs to be fixed

            # check if frame empty
            if colorframe is None:
                break
            # process frame
            ########################
            # Calibration           #
            ########################
            if len(target_corners) != 4:
                frame, screen_corners, target_corners = cal.calibrateViaARUco(colorframe, screen_corners,
                                                                              target_corners)
            # if transform_mat.size == 0:
            #     frame, screen_corners, target_corners = cal.calibrateViaARUco(colorframe)
            #     if len(target_corners) == 4:
            #         transform_mat = cv2.getPerspectiveTransform(target_corners, screen_corners)
            else:
                # print(depthframe[int(calibrationMatrix[0][1])][int(calibrationMatrix[0][0])])
                # print("newtabledistance = ", depthframe[calibrationMatrix[0][1]][calibrationMatrix[0][0]])

                M = cv2.getPerspectiveTransform(target_corners, screen_corners)
                # TODO: derive resolution from width and height of original frame?
                caliColorframe = cv2.warpPerspective(colorframe, M, (1280, 720))

                ########################
                # Hand Detection       #
                ########################
                frame = np.zeros(colorframe.shape, dtype='uint8')
                colorspace = cv2.COLOR_BGR2LAB
                hands = getHand(caliColorframe, colorframe, colorspace, True)

                # if hands were detected visualize them
                if len(hands) > 0:
                    # Print and log the fingertips
                    for i, hand in enumerate(hands):
                        hand_image = hand["hand_image"]
                        # Altering hand colors (to avoid feedback loop
                        # Option 1: Inverting the picture
                        # hand_image = hand_image.astype(np.uint8)
                        # hand_image = cv2.cvtColor(hand_image, cv2.COLOR_GRAY2BGR)
                        hand_image = cv2.bitwise_not(hand_image)
                        hand_image[np.where((hand_image == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

                        # add the hand to the frame
                        frame = cv2.bitwise_or(frame, hand_image)

            frame.astype(np.uint8)
            # frame = reducer(frame, percentage=40)  # reduce frame by 40%
            cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            hwnd = win32gui.FindWindow(None, "Output Frame")
            cv2.imshow('Output Frame', frame)
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                   win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
            win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0,
                                                win32con.LWA_COLORKEY)  # black as transparent
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device.stop()
        pass


if __name__ == '__main__':
    main()