import cv2
import numpy as np
import math
from cv2 import aruco

def angle(A1, A2, B1, B2):
    """Calculates the signed angle between two lines (A and B)."""

    def calc_radians(ab):
        if ab > math.pi:
            return ab + (-2 * math.pi)
        else:
            if ab < 0 - math.pi:
                return ab + (2 * math.pi)
            else:
                return ab + 0

    AhAB = math.atan2((B2[0] - B1[0]), (B2[1] - B1[1]))
    AhAO = math.atan2((A2[0] - A1[0]), (A2[1] - A1[1]))

    res = calc_radians(AhAB - AhAO)

    return 180+math.degrees(res)

def detect(irframe, cm_per_pix):
    mask = cv2.inRange(irframe, 255, 255)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    blurred = cv2.blur(mask, (5, 5))  # TODO: VERY BASIC, TRY OTHER FILTERS
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # ret, thresholded = cv2.threshold(blurred, 50, 255, 0)  # TODO: VERY BASIC, TRY OTHER THRESHHOLDS
    ret, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    cX = -1
    cY = -1

    if (thresholded!=0).any():
        # calculate the contours to get the area of the detections
        mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy = cv2.findContours(thresholded, mode, method)
        for c in contours:
            # points are only valid if they are smaller than 25cm2
            if cv2.contourArea(c) < (3/(cm_per_pix*cm_per_pix)):
                # calculate moments of binary image
                M = cv2.moments(thresholded)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # only acknowledge the first valid detection
                break
    return (cX,cY)

def ir_annotations(frame, colorframe, target_corners, device, prev_point, prev_frame, current_tui_setting, tui_dict, cm_per_pix):
    # look for aruco codes (more easily detected in black and white images)
    gray = cv2.cvtColor(colorframe, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # if aruco codes have been found
    if ids is not None:
        for i in range(len(ids)):  # loop through all detections
            for key in tui_dict.keys():  # loop through all codes
                # save the position of the detected codes
                if ids[i] == key:
                    if key == 8:
                        angle_lines = angle((corners[i][0][0][0], corners[i][0][0][1]),
                                            (corners[i][0][2][0], corners[i][0][2][1]),
                                            (target_corners[0][0], target_corners[0][1]),
                                            (target_corners[2][0], target_corners[2][1]))
                        if math.isnan(angle_lines) != True:
                            tui_dict[key]['thickness'] = int(max(15 * (angle_lines / 360), 1.0))
                    tui_dict[key]["edges"] = corners[i][0].astype('int32')

    # the order of the frame corners has to be adjusted to form a rectangle
    frame_corners = target_corners.reshape((-1, 1, 2)).astype('int32')
    temp = frame_corners[2].copy()
    frame_corners[2] = frame_corners[3]
    frame_corners[3] = temp

    irframe = device.getirstream()
    point = detect(irframe, cm_per_pix)
    if len(prev_frame) < 1:
        # init prev frame as empty
        prev_frame = np.zeros_like(frame)
    # if an infrared point was detected
    if point[0] != -1:
        # check if the point was made on any of the aruco codes
        for key in tui_dict.keys():
            if key != 8:
                tuiX = tui_dict[key]["edges"][:, :1]
                tuiY = tui_dict[key]["edges"][:, 1:]
                # check if the ir detection is on an aruco code
                if (point[0] < tuiX.max() and point[0] > tuiX.min() and point[1] < tuiY.max() and point[1] > tuiY.min()):
                    # change the draw settings according to the code
                    current_tui_setting = tui_dict[key]

                    # make a brief outline around the frame or the aruco code to signify the functionality
                    color = current_tui_setting["color"]
                    # check if the aruco code is inside the frame
                    tui_dict[key]["inside"] = cv2.pointPolygonTest(frame_corners, (tui_dict[key]["edges"][0][0], tui_dict[key]["edges"][0][1]), False)
                    # if is not inside the frame use the corners of the aruco code for drawing the outline
                    if tui_dict[key]["inside"] > 0.0:
                        pts = tui_dict[key]["edges"].reshape((-1, 1, 2))
                    else:
                        pts = frame_corners.copy()
                    # the eraser has a special outline
                    if color == (0, 0, 0):
                        for p in pts:
                            cv2.circle(frame, (p[0][0], p[0][1]), tui_dict[8]["thickness"], (255, 255, 255), -1)
                    # normal outline for the other draw modes
                    else:
                        cv2.polylines(frame, [pts], True, color, tui_dict[8]["thickness"])

                    # do not draw a point or line
                    point = (-1, -1)
                    break
        # the point is checked again, because if it was on an aruco code it would now be (-1, -1)
        if point[0] != -1:
            color = current_tui_setting["color"]
            thickness = tui_dict[8]["thickness"]
            # draw point
            prev_frame[(point[1] - int(thickness/2)):(point[1] + int(thickness/2)), (point[0] - int(thickness/2)):(point[0] + int(thickness/2))] = current_tui_setting["color"]
            # draw line
            if prev_point[0] != -1:
                cv2.line(prev_frame, prev_point, point, color, thickness)
    prev_point = point
    frame = cv2.bitwise_or(frame, prev_frame)
    return cv2.bitwise_or(frame, prev_frame), prev_frame, prev_point, current_tui_setting
