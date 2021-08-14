import cv2
import numpy as np

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
            # points are only valid if they are smaller than 25cm2 and larger than 0.15cm2
            if cv2.contourArea(c) < (25/(cm_per_pix*cm_per_pix)) and cv2.contourArea(c) > (0.15/(cm_per_pix*cm_per_pix)):
                # calculate moments of binary image
                M = cv2.moments(thresholded)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # only acknowledge the first valid detection
                break;
    return (cX,cY)

def ir_annotations(frame, caliColorframe, device, prev_point, prev_frame, current_tui_setting, tui_dict, cm_per_pix):
    irframe = device.getirstream()
    point = detect(irframe, cm_per_pix)
    if len(prev_frame) < 1:
        # init prev frame as empty
        prev_frame = np.zeros_like(frame)
    # if an infrared point was detected
    if point[0] != -1:
        # check if the point was made on any of the aruco codes
        for key in tui_dict.keys():
            tuiX = tui_dict[key]["edges"][:, :1]
            tuiY = tui_dict[key]["edges"][:, 1:]
            # check if the ir detection is on an aruco code
            if (point[0] < tuiX.max() and point[0] > tuiX.min() and point[1] < tuiY.max() and point[1] > tuiY.min()):
                # change the draw settings according to the code
                current_tui_setting = tui_dict[key]
                # make a brief outline around the code to signify the functionality
                pts = tui_dict[key]["edges"].reshape((-1, 1, 2))
                color = current_tui_setting["color"]
                # the eraser has a special outline
                if color == (0, 0, 0):
                    for p in pts:
                        cv2.circle(frame, (p[0][0], p[0][1]), 5, (255, 255, 255), -1)
                # normal outline for the other draw modes
                else:
                    cv2.polylines(frame, [pts], True, color, 3)
                # do not draw a point or line
                point = (-1, -1)
                break
        # the point is checked again, because if it was on an aruco code it would now be (-1, -1)
        if point[0] != -1:
            color = current_tui_setting["color"]
            thickness = current_tui_setting["thickness"]
            # draw point
            prev_frame[(point[1] - int(thickness/2)):(point[1] + int(thickness/2)), (point[0] - int(thickness/2)):(point[0] + int(thickness/2))] = current_tui_setting["color"]
            # draw line
            if prev_point[0] != -1:
                cv2.line(prev_frame, prev_point, point, color, thickness)
    prev_point = point
    frame = cv2.bitwise_or(frame, prev_frame)
    return cv2.bitwise_or(frame, prev_frame), prev_frame, prev_point, current_tui_setting
