# import library
import argparse
import concurrent
import sys
import threading
import time
import tkinter as tk
import traceback

import asyncio
import cv2
import mediapipe as mp
import numpy as np
import win32api
import win32con
import win32gui
from vidgear.gears.asyncio import NetGear_Async
from win32api import GetSystemMetrics

import libs.calibration as cal
import libs.hand_neural_net as hand_lib_nn
import libs.infrared as infrared
from classes.bcolors import bcolors
from classes.realsense import RealSense
from dicts.colorspace_dict import colorspace_dict
from dicts.tui_dict import tui_dict

# init mediapipe hand detection parameters
min_detection_confidence = 0.35
min_tracking_confidence = 0.3
min_samples = 3
eps = 30
# init mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
handsMP = mp_hands.Hands(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
    max_num_hands=4)

# init ir_frame
irframe = np.array([])

# Create a async frame generator as custom source
async def custom_frame_generator(pattern):
    try:
        tabledistance = 1200  # Default distance to table
        # Open video stream
        device = RealSense(DeviceSrc, pattern.depth, pattern.iranno)
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
        # init parameters
        prev_frame = np.array([])
        prev_point = (-1, -1)
        current_tui_setting = tui_dict[5]
        cm_per_pix = -1

        global irframe, min_samples, eps

        # loop over stream until its terminated
        while True:
            ########################
            # Startup              #
            ########################
            # store time in seconds since the epoch (UTC)
            timestamp = time.time()
            # read frames
            colorframe = device.getcolorstream()
            # init the ir_frame as empty
            irframe = np.zeros(colorframe.shape, dtype='uint8')

            # check if frame empty
            if colorframe is None:
                break
            # process frame
            ########################
            # Calibration          #
            ########################
            if transform_mat.size == 0:
                frame, screen_corners, target_corners, cm_per_pix = cal.calibrateViaARUco(colorframe)
                # if all four target corners have been found create the transformation matrix
                if len(target_corners) == 4:
                    transform_mat = cv2.getPerspectiveTransform(target_corners, screen_corners)
                    # if depth mode is activated also read the depth frame and measure the table distance
                    if pattern.depth:
                        depthframe = device.getdepthstream()
                        tabledistance = depthframe[int(target_corners[1][1])][int(target_corners[1][0])]
                        if tabledistance == 0:
                            tabledistance = 1200
            # main frame processing after calibration is finished
            else:
                # init the resulting frame as empty
                frame = np.zeros(colorframe.shape, dtype='uint8')

                ##########################
                # IR Annotations + Hands #
                ##########################
                if (pattern.iranno):
                    # simultaneously detect hands and do the ir drawings
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        ir_future = executor.submit(infrared.ir_annotations, frame, colorframe, target_corners, device, prev_point,
                                                    prev_frame, current_tui_setting, tui_dict, cm_per_pix)
                        hand_future = executor.submit(hand_lib_nn.hand_detection, frame, colorframe, colorspace,
                                                      pattern, lower_color, upper_color, handsMP,
                                                      tabledistance, timestamp, device,
                                                      transform_mat, min_samples, eps, cm_per_pix)
                        frame = hand_future.result()
                        irframe, prev_frame, prev_point, current_tui_setting = ir_future.result()
                        frame = cv2.bitwise_or(frame, irframe)
                        irframe = cv2.warpPerspective(irframe, transform_mat, irframe.shape[1:None:-1])
                ##############
                # Just Hands #
                ##############
                else:
                    frame = hand_lib_nn.hand_detection(frame, colorframe, colorspace,
                                                      pattern, lower_color, upper_color, handsMP,
                                                      tabledistance, timestamp, device,
                                                      transform_mat, min_samples, eps, cm_per_pix)
                ##### Mediapipe: visualize detections for debugging ###########
                # resultsMP = handsMP.process(caliColorframe)
                # if resultsMP.multi_hand_landmarks:
                #     frame.flags.writeable = True
                #     for hand_landmarks in resultsMP.multi_hand_landmarks:
                #         mp_drawing.draw_landmarks(
                #             frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                frame = cv2.warpPerspective(frame, transform_mat, colorframe.shape[1:None:-1])

            # to measure time to completion
            # print(time.time() - timestamp)
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
                hwnd = win32gui.FindWindow(None, "Output Frame")
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd,
                                                                                          win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)  # no idea, but it goes together with transparency
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, GetSystemMetrics(0), GetSystemMetrics(1),
                                      0)  # always on top
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)  # maximiced
                if pattern.paper == False:
                    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0,
                                                        win32con.LWA_COLORKEY)  # black as transparent
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
    parser.add_argument("-p", "--paper", dest='paper', action='store_true',
                        help="switch to paper plan mode")
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
    mainWindow = tk.Tk()
    app = App(mainWindow)
    mainWindow.mainloop()


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
