# import library
import concurrent

from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import reducer
from classes.realsense import RealSense
from classes.bcolors import bcolors
import libs.hand as hand_lib
import libs.hand_neuralNet as hand_lib_nn
import libs.ir_detection as ir_detection
import libs.calibration as cal
import libs.utils as utils
import libs.visHeight as height
import numpy as np
import cv2, asyncio
import argparse
import sys
import time
import win32api
import win32con
import win32gui
from win32api import GetSystemMetrics
import traceback
import mediapipe as mp
import tkinter as tk
import threading
from cv2 import aruco

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

HostPort = 5555
PeerAddress = "localhost"
PeerPort = 5555
continuousCalibration = False
overlay = True
DeviceSrc = "0"
finish = False

irframe = np.array([])

min_detection_confidence = 0.35
min_tracking_confidence = 0.3
min_samples = 3
eps = 30

handsMP = mp_hands.Hands(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
    max_num_hands=4)

colorspace_dict = {
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
    "ycrcb": cv2.COLOR_BGR2YCrCb,
    "rgb": cv2.COLOR_BGR2RGB,
    "luv": cv2.COLOR_BGR2LUV,
    "xyz": cv2.COLOR_BGR2XYZ,
    "hls": cv2.COLOR_BGR2HLS,
    "yuv": cv2.COLOR_BGR2YUV
}

tui_dict = {
    7: {
        "color": (0,0,0),
        "thickness": 35,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8')
    }, # eraser
    1: {
        "color": (255, 255, 255),
        "thickness": 3,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8')
    }, # black pen
    2: {
        "color": (200, 3, 3),
        "thickness": 3,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8')
    },
    3: {
        "color": (200, 200, 3),
        "thickness": 3,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8')
    },
    4: {
        "color": (3, 3, 200),
        "thickness": 3,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8')
    },
    5: {
        "color": (200, 3, 200),
        "thickness": 3,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8')
    },
    6: {
        "color": (3, 200, 200),
        "thickness": 3,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8')
    },
}

# Create a async frame generator as custom source
async def custom_frame_generator(pattern):
    try:
        tabledistance = 1200  # Default distance to table
        # Open video stream
        device = RealSense(DeviceSrc, pattern.iranno)
        # open log file and write header
        log = open("logs/log_" + str(int(time.time())) + ".log", "x")
        log.write("timestamp height class x y" + "\n")
        # initialize corners
        transform_mat = np.array([])
        # define initial pink range
        lower_color = np.array([1, 1, 1])
        upper_color = np.array([0, 0, 0])
        # translate colorspace to opencv code
        colorspace = colorspace_dict[pattern.colorspace]
        prev_frame = []
        prev_point = (-1, -1)
        current_tui_setting = tui_dict[1]

        global irframe

        global finish

        # loop over stream until its terminated
        while not finish:
            # print("start")
            ########################
            # Startup              #
            ########################
            # store time in seconds since the epoch (UTC)
            timestamp = time.time()
            # read frames
            colorframe = device.getcolorstream()

            irframe = np.zeros(colorframe.shape, dtype='uint8')

            # check if frame empty
            if colorframe is None:
                break
            # process frame
            ########################
            # Calibration           #
            ########################
            if continuousCalibration == False and transform_mat.size == 0:
                frame, screen_corners, target_corners = cal.calibrateViaARUco(colorframe)
                if len(target_corners) == 4:
                    transform_mat = cv2.getPerspectiveTransform(target_corners, screen_corners)
                    if pattern.depth:
                        depthframe = device.getdepthstream()
                        tabledistance = depthframe[int(target_corners[1][1])][int(target_corners[1][0])]
                        if tabledistance == 0:
                            tabledistance = 1200
                # frame = reducer(frame, percentage=40)  # reduce frame by 40%

            else:

                # TODO: derive resolution from width and height of original frame?
                caliColorframe = cv2.warpPerspective(colorframe, transform_mat, (1280, 720))

                frame = np.zeros(colorframe.shape, dtype='uint8')

                ##########################
                # IR Annotations + Hands #
                ##########################
                if (pattern.iranno):
                    # look for aruco codes
                    gray = cv2.cvtColor(caliColorframe, cv2.COLOR_BGR2GRAY)
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
                    parameters = aruco.DetectorParameters_create()
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    if ids is not None:
                        for i in range(len(ids)): # loop through all detections
                            for key in tui_dict.keys(): # loop through all codes
                                if ids[i] == key:
                                    tui_dict[key]["edges"] = corners[i][0].astype('int32')
                    # simultaniously detect hands and do the ir drawings
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        ir_future = executor.submit(ir_annotations, frame, caliColorframe, device, transform_mat, prev_point, prev_frame, current_tui_setting)
                        hand_future = executor.submit(hand_detection,
                                                 frame, caliColorframe, colorspace, pattern.edges, lower_color,
                                                 upper_color, handsMP, log,
                                                 tabledistance, pattern.logging, pattern.depth, timestamp, device,
                                                 transform_mat)
                        frame = hand_future.result()
                        irframe, prev_frame, prev_point, current_tui_setting = ir_future.result()
                ##############
                # Just Hands #
                ##############
                else:
                    frame = hand_detection(frame, caliColorframe, colorspace, pattern.edges, lower_color, upper_color, handsMP, log,
                               tabledistance, pattern.logging, pattern.depth, timestamp, device, transform_mat)
                ##### Mediapipe: visualize detections ###########
                # resultsMP = handsMP.process(caliColorframe)
                # if resultsMP.multi_hand_landmarks:
                #     frame.flags.writeable = True
                #     for hand_landmarks in resultsMP.multi_hand_landmarks:
                #         mp_drawing.draw_landmarks(
                #             frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # to measure time to completion
            print(time.time() - timestamp)
            yield frame
            # sleep for sometime
            await asyncio.sleep(0.00001)
        # close stream
        device.stop()
        # close file
        log.close()
    except Exception as e:
        print(bcolors.FAIL + traceback.format_exc() + bcolors.ENDC)
    finally:
        log.flush()
        print(bcolors.OKGREEN + "\n Session log saved: " + log.name + "\n" + bcolors.WARNING)
        log.close()
        device.stop()


def ir_annotations(frame, caliColorframe, device, transform_mat, prev_point, prev_frame, current_tui_setting):
    irframe = device.getirstream()
    caliIrframe = cv2.warpPerspective(irframe, transform_mat, (1280, 720))
    point = ir_detection.detect(caliIrframe, caliColorframe)
    # semi-permanent
    # prev_frame.append(ir)
    # if len(prev_frame) > 0:
    #     for prev in prev_frame:
    #         frame = cv2.bitwise_or(frame, prev)
    # if len(prev_frame) > 30:
    #     prev_frame.pop(0)

    # permanent
    if point[0] != -1:
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
                if color == (0, 0, 0):
                    color = (10, 10, 10)
                cv2.polylines(frame, [pts], True, color, 5)
                # do not draw a point or line
                point = (-1,-1)
                break
        if point[0] != -1:
            color = current_tui_setting["color"]
            thickness = current_tui_setting["thickness"]
            prev_frame[(point[1] - int(thickness/2)):(point[1] + int(thickness/2)), (point[0] - int(thickness/2)):(point[0] + int(thickness/2))] = current_tui_setting["color"]
            if prev_point[0] != -1:
                cv2.line(prev_frame, prev_point, point, color, thickness)
    prev_point = point
    if len(prev_frame) < 1:
        prev_frame = np.zeros_like(frame)
    frame = cv2.bitwise_or(frame, prev_frame)
    return cv2.bitwise_or(frame, prev_frame), prev_frame, prev_point, current_tui_setting


def hand_detection(frame, caliColorframe, colorspace, edges, lower_color, upper_color, handsMP, log, tabledistance,
                   logging, depth, timestamp, device, transform_mat):
    # hands, lower_color, upper_color = hand_lib.getHand(caliColorframe, colorframe, colorspace,
    #                                                   pattern.edges,
    #                                                   lower_color, upper_color)
    # Mediapipe
    global min_samples, eps
    hands, lower_color, upper_color = hand_lib_nn.getHand(caliColorframe, colorspace, edges, lower_color, upper_color,
                                                          handsMP, log, min_samples, eps)

    # if hands were detected visualize them
    if len(hands) > 0:
        # if the depth is enabled read out the depth frame
        if depth:
            depthframe = device.getdepthstream()
            caliDepthframe = cv2.warpPerspective(depthframe, transform_mat, (1280, 720))

        # Print and log the fingertips
        for i, hand in enumerate(hands):
            hand_image = hand["hand_image"]
            # Altering hand colors (to avoid feedback loop
            # Option 1: Inverting the picture
            if edges != True:
                hand_image = cv2.bitwise_not(hand_image, hand["mask"])
                hand_image = cv2.bitwise_and(hand_image, hand_image, mask=hand["mask"])
            # Calculate hand centre
            M = cv2.moments(hand["contour"])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # if depth is enabled also visualize and log the distance
            if depth:
                handToTableDist = (float(tabledistance) - float(caliDepthframe[cY][cX])) / 100

                if handToTableDist > 0 and handToTableDist < 10:
                    hand_image = height.visHeight(hand_image, handToTableDist)

                if logging:
                    cv2.circle(hand_image, (cX, cY), 4, utils.id_to_random_color(i), -1)
                    cv2.putText(hand_image, "  " + str(handToTableDist), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i), 1, cv2.LINE_AA)
                log.write(''.join(
                    [str(timestamp), " ", str(float(tabledistance) - float(depthframe[cY][cX])), " H ", str(cX), " ",
                     str(cY), "\n"]))

                for f in hand["fingers"]:
                    if logging:
                        cv2.circle(hand_image, f, 4, utils.id_to_random_color(i), -1)
                        cv2.putText(hand_image,
                                    "  " + str((float(tabledistance) - float(caliDepthframe[f[1]][f[0]])) / 100), f,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i),
                                    1, cv2.LINE_AA)
                    # print("color pixel value of ", f, ":", frame[f[1]][f[0]]) # <- TODO: reverse coordinates idk why
                    # print("depth pixel value of ", f, ":", depthframe[f[1]][f[0]])
                    log.write(''.join(
                        [str(timestamp), " ", str(float(tabledistance) - float(depthframe[cY][cX])), " P ", str(f[0]),
                         " ", str(f[1]), "\n"]))
            else:
                if logging:
                    cv2.circle(hand_image, (cX, cY), 4, utils.id_to_random_color(i), -1)
                # record depth as "Null"
                log.write(str(timestamp) + " Null H " + str(cX) + " " + str(
                    cY) + "\n")
                for f in hand["fingers"]:
                    if logging:
                        cv2.circle(hand_image, f, 4, utils.id_to_random_color(i), -1)
                    # record depth as "Null"
                    log.write(''.join([str(timestamp), " Null P ", str(f[0]), " ", str(
                        f[1]), "\n"]))
            # add the hand to the frame
            frame = cv2.bitwise_or(frame, hand_image)
    # frame = reducer(frame, percentage=40)  # reduce frame by 40%
    # to measure time to completion
    print(time.time() - timestamp)
    return frame


# Create a async function where you want to show/manipulate your received frames
async def client_iterator(client, pattern):
    # loop over Client's Asynchronous Frame Generator
    if not pattern.invisible:
        cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    async for frame in client.recv_generator():
        try:
            if not pattern.invisible:
                # do something with received frames here
                # print("frame recieved")
                # Show output window
                if (pattern.iranno):
                    global irframe
                    if irframe.shape != (0,):
                        frame = frame.astype("uint8")
                        frame = cv2.bitwise_or(frame, irframe)
                cv2.imshow("Output Frame", frame)
                if overlay:
                    hwnd = win32gui.FindWindow(None, "Output Frame")
                    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd,
                                                                                              win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)  # no idea, but it goes together with transparency
                    # win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0,
                    #                                     win32con.LWA_COLORKEY)  # black as transparent
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, GetSystemMetrics(0), GetSystemMetrics(1),
                                          0)  # always on top
                    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)  # maximiced
                key = cv2.waitKey(1) & 0xFF
                # await before continuing
            await asyncio.sleep(0.00001)
        except Exception as e:
            print(e)


async def netgear_async_playback(pattern):
    try:
        # define and launch Client with `receive_mode = True`
        server = NetGear_Async(address=PeerAddress, port=PeerPort, logging=pattern.logging)  # invalid protocol
        server.config["generator"] = custom_frame_generator(pattern)
        server.launch()
        # define and launch Client with `receive_mode = True` and timeout = 5.0
        client = NetGear_Async(port=HostPort, receive_mode=True, timeout=float("inf"), logging=pattern.logging).launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client, pattern)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        print(e)
        pass
    finally:
        try:
            server
        except Exception as e:
            print("server undefined")
        else:
            server.close(skip_loop=True)
        try:
            client
        except Exception as e:
            print("client undefined")
        else:
            client.close(skip_loop=True)


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="GECCO")
    parser.add_argument("-s", "--source", help="Realsense device S/N")
    parser.add_argument('-r', '--remote', nargs='+',
                        help='Set this argument to connect to another peer. '
                             'Give it the host port number, the peer IP address and the port number of the peer')
    parser.add_argument("-f", "--file", help="Simulate camera sensor from .bag file")
    parser.add_argument("-d", "--depth", help="Don't use depth camera (faster)", action='store_false')
    parser.add_argument("-i", "--invisible", help="Nothing is displayed. Only hand data is logged.",
                        action='store_true')
    parser.add_argument("-e", "--edges", help="Only visualize the edges of a hand", action='store_true')
    parser.add_argument("-c", "--colorspace",
                        help="choose the colorspace for color segmentation. Popular choice is 'hsv' but we achieved best results with 'lab'",
                        choices=['hsv', 'lab', 'ycrcb', 'rgb', 'luv', 'xyz', 'hls', 'yuv'], default='lab')
    parser.add_argument("-v", "--verbose", dest='logging', action='store_true', help="enable vidgear logging")
    parser.add_argument("-a", "--annotations", dest='iranno', action='store_true',
                        help="enable making annotations using IR light")
    options = parser.parse_args(args)
    return options


def set_valueA(val):
    global handsMP, min_detection_confidence, min_tracking_confidence
    min_detection_confidence = float(val)
    handsMP = mp_hands.Hands(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)


def set_valueB(val):
    global handsMP, min_detection_confidence, min_tracking_confidence
    min_tracking_confidence = float(val)
    handsMP = mp_hands.Hands(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)


def set_valueC(val):
    global min_sample
    min_sample = int(val)


def set_valueD(val):
    global eps
    eps = int(val)


class App(object):
    def __init__(self, master):
        master.geometry("200x200")
        master.title("My GUI Title")
        w = tk.Scale(master, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, command=set_valueA)
        w.set(0.35)
        w.pack()
        w = tk.Scale(master, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, command=set_valueB)
        w.set(0.30)
        w.pack()
        w = tk.Scale(master, from_=0, to=15, orient=tk.HORIZONTAL, command=set_valueC)
        w.set(3)
        w.pack()
        w = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, command=set_valueD)
        w.set(30)
        w.pack()


def tkinterGui():
    global finish
    mainWindow = tk.Tk()
    app = App(mainWindow)
    mainWindow.mainloop()
    # When the GUI is closed we set finish to "True"
    finish = True


if __name__ == '__main__':
    options = getOptions(sys.argv[1:])
    DeviceSrc = options.source
    # configure network
    if options.remote:
        HostPort = options.remote[0]
        PeerAddress = options.remote[1]
        PeerPort = options.remote[2]
    else:
        HostPort = 5555
        PeerAddress = "localhost"
        PeerPort = 5555

    # configure Realsense device
    if options.file:
        DeviceSrc = options.file
    if options.logging:
        GUI = threading.Thread(target=tkinterGui)
        GUI.start()
        Process = threading.Thread(target=asyncio.run(netgear_async_playback(options)))
        Process.start()
        GUI.join()
        Process.join()
    else:
        asyncio.run(netgear_async_playback(options))
