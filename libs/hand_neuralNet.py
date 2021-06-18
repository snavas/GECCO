import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import DBSCAN
import libs.utils as utils
import math
from vidgear.gears.helper import reducer

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def getHand(colorframe, colorspace, edges, lower_color, upper_color, handsMP, log, min_samples, eps):
    def calculateCenter(x1, y1, x2, y2):
        x = int((x2 - x1) / 2 + x1)
        y = int((y2 - y1) / 2 + y1)
        return x, y

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

    detectionFrame = cv2.cvtColor(colorframe, colorspace)
    hands = []
    # apply mediapipe to the image
    resultsMP = handsMP.process(colorframe)
    # if hands were detected
    if resultsMP.multi_hand_landmarks:
        image_rows, image_cols, _ = colorframe.shape
        # for every hand
        for hand_landmarks in resultsMP.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            # initialize some variables
            curr_detections = []
            points = []
            maskTemp = np.zeros((colorframe.shape[0], colorframe.shape[1]), dtype=np.uint8)
            maxX = 0
            minX = image_cols
            maxY = 0
            minY = image_rows
            # get the color and extent of all detected joints
            for landmark in landmarks:
                # get the position of the detection (has to be smaller than the maximum extent)
                x = min(math.floor(landmark.x * image_cols), image_cols-1)
                y = min(math.floor(landmark.y * image_rows), image_rows-1)
                # get the color at this detection
                color_detection = detectionFrame[y, x]
                curr_detections.append(color_detection)
                # update the extent of the detections
                if maxX < x:
                    maxX = x
                elif minX > x:
                    minX = x
                if maxY < y:
                    maxY = y
                elif minY > y:
                    minY = y
                landmark.x = x
                landmark.y = y
                points.append((x,y))
            # also get the color from points in between joints
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                # get the middle between the joints
                x,y = calculateCenter(start_point.x, start_point.y, end_point.x, end_point.y)
                # get the color at this detection
                color_detection = detectionFrame[y, x]
                curr_detections.append(color_detection)
            width = maxX-minX
            height = maxY-minY
            # enlarge the extent by 10 to 20 percent
            maxX = min(image_cols, maxX + math.floor((width/image_cols)*200))
            maxY = min(image_rows, maxY + math.floor((height/image_rows)*100))
            minX = max(0, minX - math.floor((width/image_cols)*200))
            minY = max(0, minY - math.floor((height/image_rows)*100))
            crop_img = colorframe[minY:maxY, minX:maxX]

            ########################### COLOR DETECTION ####################################
            curr_detections = np.asarray(curr_detections)
            outlier_detection = DBSCAN(min_samples=min_samples, eps=eps)
            clusters = outlier_detection.fit_predict(curr_detections)
            curr_detections = curr_detections[(clusters != -1)]
            if curr_detections.size == 0:
                break
            # The hand color is usually not extremely bright or extremely dark, so the detection are thresholded
            upperThresh = curr_detections[curr_detections > 248].size / curr_detections.size
            lowerThresh = curr_detections[curr_detections < 7].size / curr_detections.size
            # for detection in curr_detections:
            #     log.write(''.join(["[", str(detection[0]), ", ", str(detection[1]), ", ", str(detection[2]), "],", "\n"]))
            # The detection is only declared valid if not too many values are extremely bright/dark
            if upperThresh < 0.3 and lowerThresh < 0.3:
                # If there are too many clusters, the detected color was not very homogenic and therefore probably not valid
                if clusters[clusters > 0].size < 3 and clusters[clusters > 1].size == 0:
                    # calculate mean and standard deviation of the detected colors
                    mean = np.mean(curr_detections, axis=0)
                    std = np.std(curr_detections, axis=0)
                    upper_color = mean + std * 3
                    lower_color = mean - std * 3
                    upper_color[upper_color > 255] = 255
                    lower_color[lower_color < 0] = 0
                    upper_color = upper_color.astype(np.uint8)
                    lower_color = lower_color.astype(np.uint8)

                ########################### HAND MASK #######################################
                colorConverted = cv2.cvtColor(crop_img, colorspace)
                tempMask = cv2.inRange(colorConverted, lower_color, upper_color)
                blurred = cv2.blur(tempMask, (5, 5))
                ret, handmask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
                maskTemp[minY:maxY, minX:maxX] = handmask
                handmask = maskTemp
                ########################### HAND CONTOUR ####################################
                mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
                method = cv2.CHAIN_APPROX_SIMPLE
                hand_contours = []
                contours, hierarchy = cv2.findContours(handmask, mode, method)
                # initialize hand
                hand = {
                    "contour": np.empty(shape=(0,1,2), dtype=np.uint8),
                    "fingers": points,
                    "mask": np.zeros_like(handmask)
                }
                for c in contours:
                    # If contours are bigger than a certain area we push them to the array
                    if cv2.contourArea(c) > 3000:
                        rHull = getRoughHull(c)
                        if rHull is not None:
                            hand_contours.append(c)
                            tempMask = np.zeros_like(handmask)  # Create mask where white is what we want, black otherwise
                            cv2.drawContours(tempMask, [c], -1, 255, -1)  # Draw filled contour in mask
                            tempOut = np.zeros_like(handmask)  # Extract out the object and place into output image
                            tempOut[tempMask == 255] = handmask[tempMask == 255]
                            # dilate and erode to remove holes
                            tempOut = cv2.dilate(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
                            tempOut = cv2.erode(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)), iterations=1)

                            if c.shape != (0,):
                                hand["contour"] = np.concatenate((hand["contour"], c))
                            hand["mask"] = cv2.bitwise_or(hand["mask"], tempOut)
                # if there is a contour there is also a hand
                if hand["contour"].shape != (0,1,2):
                    copy = colorframe.copy()
                    # edge only mode
                    if edges:
                        # Heavily dilated mask
                        heavily_dilated = cv2.dilate(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 30)),
                                                   iterations=3)
                        # A little less dilated mask
                        dilated = cv2.dilate(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                                                    iterations=1)

                        # get a really dilated masked out hand, so that the edges dont have to be calculated for the entire image
                        hand_image = cv2.bitwise_and(copy, copy, mask=heavily_dilated)
                        # calculate edges
                        canny_output = cv2.Canny(hand_image, 100, 200)
                        # empty image
                        hand_image = np.empty((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

                        # get contours of edges
                        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE,
                                                               cv2.CHAIN_APPROX_SIMPLE)
                        # draw the contours into the empty image
                        for i in range(len(contours)):
                            cv2.drawContours(hand_image, contours, i, (254, 254, 254), 3, cv2.LINE_8, hierarchy,
                                             0)

                        for i in range(len(contours)):
                            cv2.drawContours(hand_image, contours, i, (1, 1, 1), 1, cv2.LINE_8, hierarchy, 0)
                        # mask out the outer edges, that belong to the more heavily dilated mask
                        result = cv2.bitwise_and(hand_image, hand_image, mask=dilated)
                        # comment this in, to see edges and hand:
                        # hand_image_norm = cv2.bitwise_and(copy, copy, mask=curMask[0])
                        # hand_image = cv2.bitwise_or(hand_image, hand_image_norm)

                    # normal mode
                    else:
                        result = cv2.bitwise_and(copy, copy, mask=hand["mask"])
                        result = cv2.bitwise_not(result)
                    hand["hand_image"] = result
                    hands.append(hand)
    return hands, lower_color, upper_color