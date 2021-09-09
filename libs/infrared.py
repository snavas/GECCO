import cv2
import numpy as np
import math
from cv2 import aruco


def calc_radians(ab):
    if ab > math.pi:
        return ab + (-2 * math.pi)
    else:
        if ab < 0 - math.pi:
            return ab + (2 * math.pi)
        else:
            return ab + 0


def angle(a1, a2, b1, b2):
    """Calculates the signed angle between two lines (A and B)."""

    ah_ab = math.atan2((b2[0] - b1[0]), (b2[1] - b1[1]))
    ah_ao = math.atan2((a2[0] - a1[0]), (a2[1] - a1[1]))

    res = calc_radians(ah_ab - ah_ao)

    return 180 + math.degrees(res)


def detect(irframe, cm_per_pix):
    mask = cv2.inRange(irframe, 255, 255)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    blurred = cv2.blur(mask, (5, 5))  # TODO: VERY BASIC, TRY OTHER FILTERS
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # ret, thresholded = cv2.threshold(blurred, 50, 255, 0)  # TODO: VERY BASIC, TRY OTHER THRESHHOLDS
    ret, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    c_x = -1
    c_y = -1

    if (thresholded != 0).any():
        # calculate the contours to get the area of the detections
        mode = cv2.RETR_EXTERNAL  # cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy = cv2.findContours(thresholded, mode, method)
        for c in contours:
            # points are only valid if they are smaller than 25cm2
            if cv2.contourArea(c) < (3 / (cm_per_pix * cm_per_pix)):
                # calculate moments of binary image
                m = cv2.moments(thresholded)
                # calculate x,y coordinate of center
                c_x = int(m["m10"] / m["m00"])
                c_y = int(m["m01"] / m["m00"])
                # only acknowledge the first valid detection
                break
    return (c_x, c_y)


def ir_annotations(frame, colorframe, target_corners, device, prev_point, prev_frame, current_tui_setting, tui_dict,
                   cm_per_pix, transform_mat):
    # look for aruco codes (more easily detected in black and white images)
    gray = cv2.cvtColor(colorframe, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

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
                        if not math.isnan(angle_lines):
                            tui_dict[key]['thickness'] = int(max(25 * (angle_lines / 360), 1.0))
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
                tui_x = tui_dict[key]["edges"][:, :1]
                tui_y = tui_dict[key]["edges"][:, 1:]
                # check if the ir detection is on an aruco code
                if tui_x.max() > point[0] > tui_x.min() and tui_y.max() > point[1] > tui_y.min():
                    # change the draw settings according to the code
                    current_tui_setting = tui_dict[key]

                    # make a brief outline around the frame or the aruco code to signify the functionality
                    color = current_tui_setting["color"]
                    # check if the aruco code is inside the frame
                    tui_dict[key]["inside"] = cv2.pointPolygonTest(frame_corners, (
                        int(tui_dict[key]["edges"][0][0]), int(tui_dict[key]["edges"][0][1])), False)
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
            # thickness is doubled for the eraser
            if color == (0, 0, 0):
                thickness = thickness * 2
            # draw point
            cv2.circle(prev_frame, (point[0], point[1]), int(thickness / 2), color, -1)
            # draw line
            if prev_point[0] != -1:
                cv2.line(prev_frame, prev_point, point, color, thickness)
    prev_point = point
    frame = cv2.bitwise_or(frame, prev_frame)

    color = current_tui_setting["color"]

    ##############################
    # draw switch point for knob #
    ##############################
    # calculate center of the aruco code
    x_sum = tui_dict[8]["edges"][0][0] + tui_dict[8]["edges"][1][0] + tui_dict[8]["edges"][2][0] + \
            tui_dict[8]["edges"][3][0]
    y_sum = tui_dict[8]["edges"][0][1] + tui_dict[8]["edges"][1][1] + tui_dict[8]["edges"][2][1] + \
            tui_dict[8]["edges"][3][1]
    x_center = x_sum * .25
    y_center = y_sum * .25
    # calculate vertical vector
    v_x = (target_corners[1][0] - target_corners[0][0])
    v_y = (target_corners[1][1] - target_corners[0][1])
    mag = math.sqrt(v_x * v_x + v_y * v_y)
    v_x = v_x / mag
    v_y = v_y / mag
    # calculate coordinates of switch point
    switch_x = x_center + v_x * 35
    switch_y = y_center + v_y * 35
    # draw switch point
    if color != (0, 0, 0):  # normal switch point
        cv2.circle(frame, (int(switch_x), int(switch_y)), int(tui_dict[8]["thickness"] / 2), color, -1)
    else:  # eraser switch point
        cv2.circle(frame, (int(switch_x), int(switch_y)), int(tui_dict[8]["thickness"]), (120, 120, 120), 1, -1)
    #############################

    return frame, prev_frame, prev_point, current_tui_setting
