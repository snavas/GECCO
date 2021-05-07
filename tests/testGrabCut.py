import cv2
from cv2 import aruco
import numpy as np
import math
from classes.realsense import RealSense
import argparse
from matplotlib import pyplot as plt
from vidgear.gears.helper import reducer
from utils import detector_utils as detector_utils
import libs.hand as hand_lib
import libs.calibration as cal
import time

screen_corners = []
target_corners = []
#
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
#
# if args.algo == 'MOG2':
#     backSub = cv2.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv2.createBackgroundSubtractorKNN()
#
# detection_graph, sess = detector_utils.load_inference_graph()

def main():
    device = RealSense('../material/new_cali2.bag')
    try:
        while True:
            frame = device.getcolorstream()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # boxes, scores = detector_utils.detect_objects(frame,
            #                                               detection_graph, sess)
            #frame = reducer(frame, percentage=60)  # reduce frame by 40%
            global screen_corners, target_corners
            if len(target_corners) != 4:
                frame, screen_corners, target_corners = cal.calibrateViaARUco(frame, screen_corners,
                                                                              target_corners)
            else:
                # print(depthframe[int(calibrationMatrix[0][1])][int(calibrationMatrix[0][0])])
                # print("newtabledistance = ", depthframe[calibrationMatrix[0][1]][calibrationMatrix[0][0]])

                M = cv2.getPerspectiveTransform(target_corners, screen_corners)
                # TODO: derive resolution from width and height of original frame?
                caliColorframe = cv2.warpPerspective(frame, M, (1280, 720))

                im_height, im_width, _ = frame.shape
                # (left, right, top, bottom) = (int(boxes[0][1] * im_width), int(boxes[0][3] * im_width),
                #                               int(boxes[0][0] * im_height), int(boxes[0][2] * im_height))
                hands = hand_lib.getHand(caliColorframe, frame, cv2.COLOR_BGR2LAB, False)
                if len(hands) > 0:
                    frame = np.zeros(caliColorframe.shape, np.uint8)
                    for hand_obj in hands:
                        (left, width, top, height) = (hand_obj["hand_crop"][0]["x"], hand_obj["hand_crop"][0]["w"],
                                                      hand_obj["hand_crop"][0]["y"], hand_obj["hand_crop"][0]["h"])

                        hand = caliColorframe[top:top+height, left:left+width]
                        mask = np.zeros(hand.shape[:2], np.uint8)
                        bgdModel = np.zeros((1, 65), np.float64)
                        fgdModel = np.zeros((1, 65), np.float64)
                        # rect = (0, 0, right-left, bottom-top)
                        rect = (0, 0, height, width)
                        try:
                            cv2.grabCut(hand, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                            hand = hand * mask2[:, :, np.newaxis]
                        except Exception as e:
                            print("error")
                        result = frame.copy()
                        result[top:top+height, left:left+width] = hand
                        frame = np.bitwise_or(frame, result)
            cv2.imshow('window', frame)

            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    finally:
        # Stop streaming
        device.stop()
        pass


if __name__ == '__main__':
    main()