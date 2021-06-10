import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import DBSCAN
import libs.utils as utils
import math
from vidgear.gears.helper import reducer

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def getHand(colorframe, colorspace, edges, lower_color, upper_color, handsMP, log):
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
    resultsMP = handsMP.process(colorframe)
    if resultsMP.multi_hand_landmarks:
        image_rows, image_cols, _ = colorframe.shape
        for hand_landmarks in resultsMP.multi_hand_landmarks:
            curr_detections = []
            maskTemp = np.zeros((colorframe.shape[0], colorframe.shape[1]), dtype=np.uint8)
            landmarks = hand_landmarks.landmark
            maxX = 0
            minX = image_cols
            maxY = 0
            minY = image_rows

            for landmark in landmarks:
                x = min(math.floor(landmark.x * image_cols), image_cols-1)
                y = min(math.floor(landmark.y * image_rows), image_rows-1)
                color_detection = detectionFrame[y, x]
                curr_detections.append(color_detection)

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
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                x,y = calculateCenter(start_point.x, start_point.y, end_point.x, end_point.y)
                color_detection = detectionFrame[y, x]
                curr_detections.append(color_detection)
            width = maxX-minX
            height = maxY-minY
            maxX = min(image_cols, maxX + math.floor((width/image_cols)*200))
            maxY = min(image_rows, maxY + math.floor((height/image_rows)*100))
            minX = max(0, minX - 30)
            minY = max(0, minY - 30)
            crop_img = colorframe[minY:maxY, minX:maxX]

            ########################### COLOR DETECTION ####################################
            curr_detections = np.asarray(curr_detections)
            outlier_detection = DBSCAN(min_samples=3, eps=30)
            clusters = outlier_detection.fit_predict(curr_detections)
            curr_detections = curr_detections[(clusters!=-1)]
            upperThresh = curr_detections[curr_detections>245].size / curr_detections.size
            lowerThresh = curr_detections[curr_detections < 10].size / curr_detections.size
            # for detection in curr_detections:
            #     log.write(''.join(["[", str(detection[0]), ", ", str(detection[1]), ", ", str(detection[2]), "],", "\n"]))
            if upperThresh < 0.35 and lowerThresh < 0.35:
                if clusters[clusters > 0].size < 3 and clusters[clusters > 1].size == 0:
                    mean = np.mean(curr_detections, axis=0)
                    #if all(i < 238 for i in mean) and all(i > 17 for i in mean):
                    std = np.std(curr_detections, axis=0) * 3
                    upper_color = mean + std
                    lower_color = mean - std
                    upper_color[upper_color > 255] = 255
                    lower_color[lower_color < 0] = 0
                    upper_color = upper_color.astype(np.uint8)
                    lower_color = lower_color.astype(np.uint8)
                    # else:
                    #     break
                # else:
                #     break
                ########################### HAND MASK #######################################
                colorConverted = cv2.cvtColor(crop_img, colorspace)
                mask = cv2.inRange(colorConverted, lower_color, upper_color)
                blurred = cv2.blur(mask, (5, 5))
                ret, handmask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
                maskTemp[minY:maxY, minX:maxX] = handmask
                handmask = maskTemp
                ########################### HAND CONTOUR ####################################
                mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
                method = cv2.CHAIN_APPROX_SIMPLE
                hand_contours = []
                contours, hierarchy = cv2.findContours(handmask, mode, method)
                for c in contours:
                    # If contours are bigger than a certain area we push them to the array
                    if cv2.contourArea(c) > 3000:
                        rHull = getRoughHull(c)
                        if rHull is not None:
                            hand_contours.append(c)
                            mask = np.zeros_like(handmask)  # Create mask where white is what we want, black otherwise
                            cv2.drawContours(mask, [c], -1, 255, -1)  # Draw filled contour in mask
                            tempOut = np.zeros_like(handmask)  # Extract out the object and place into output image
                            tempOut[mask == 255] = handmask[mask == 255]
                            # dilate and erode to remove holes
                            tempOut = cv2.dilate(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)), iterations=1)
                            tempOut = cv2.erode(tempOut, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)

                            vertices = getHullVertices(rHull, c)
                            points = filterVerticesByAngle(vertices)

                            hand = {
                                "mask": tempOut,
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
                            ####################
                            copy = colorframe.copy()
                            # edge only mode # TODO: not tested
                            if edges:
                                cnt, _ = cv2.findContours(hand["mask2"], mode, method)
                                hand["hand_crop"] = []
                                for ci in cnt:
                                    x, y, w, h = cv2.boundingRect(ci)
                                    crop_img = copy[y:y + h, x:x + w]
                                    wcrop_img = reducer(crop_img, percentage=40)  # reduce frame by 40%
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
                                for i in range(len(contours)):
                                    cv2.drawContours(hand_image, contours, i, (254, 254, 254), 3, cv2.LINE_8, hierarchy, 0)

                                hand_image = cv2.bitwise_not(hand_image)
                                result = np.empty((copy.shape[0], copy.shape[1], 3), dtype=np.uint8)
                                result[top:top + height, left:left + width] = hand_image
                                # for i in range(len(contours)):
                                #     cv2.drawContours(result, contours, i, (254,254,254), 1, cv2.LINE_8, hierarchy, 0)
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
                                result = cv2.bitwise_and(copy, copy, mask=hand["mask"])
                            result = cv2.bitwise_not(result)
                            hand["hand_image"] = result
                            hands.append(hand)
    return hands, lower_color, upper_color