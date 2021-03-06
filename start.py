# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import reducer
from classes.realsense import RealSense
from classes.bcolors import bcolors
import libs.hand as hand_lib
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

HostPort = 5555
PeerAddress = "localhost"
PeerPort = 5555
screen_corners = []
target_corners = []
continuousCalibration = False
overlay = True
DeviceSrc = "0"
#fileFlag = True

colorspacedict = {
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
    "ycrcb": cv2.COLOR_BGR2YCrCb,
    "rgb": cv2.COLOR_BGR2RGB,
    "luv": cv2.COLOR_BGR2LUV,
    "xyz": cv2.COLOR_BGR2XYZ,
    "hls": cv2.COLOR_BGR2HLS,
    "yuv": cv2.COLOR_BGR2YUV
}

# Create a async frame generator as custom source
async def custom_frame_generator(pattern):
    try:
        # Get global log variable
        global log
        tabledistance = 1200 # Default distance to table
        # Open video stream
        device = RealSense(DeviceSrc)
        # loop over stream until its terminated
        while True:
            ########################
            # Startup              #
            ########################
            # store time in seconds since the epoch (UTC)
            timestamp = time.time()
            # read frames
            colorframe = device.getcolorstream()
            #if fileFlag:
            #    colorframe = cv2.cvtColor(colorframe, cv2.COLOR_RGB2BGR) #Reading from BAG alters the color space and needs to be fixed

            # check if frame empty
            if colorframe is None:
                break
            # process frame
            ########################
            # Calibration           #
            ########################
            global screen_corners, target_corners
            if continuousCalibration == False and len(target_corners) != 4:
                frame, screen_corners, target_corners = cal.calibrateViaARUco(colorframe, screen_corners, target_corners)
                if len(target_corners) == 4 and pattern.depth:
                    depthframe = device.getdepthstream()
                    tabledistance = depthframe[int(target_corners[1][1])][int(target_corners[1][0])]
                    if tabledistance == 0:
                        tabledistance = 1200
            else:
                #print(depthframe[int(calibrationMatrix[0][1])][int(calibrationMatrix[0][0])])
                #print("newtabledistance = ", depthframe[calibrationMatrix[0][1]][calibrationMatrix[0][0]])

                M = cv2.getPerspectiveTransform(target_corners,screen_corners)
                # TODO: derive resolution from width and height of original frame?
                caliColorframe = cv2.warpPerspective(colorframe, M, (1280, 720))

                ########################
                # Hand Detection       #
                ########################
                frame = np.zeros(colorframe.shape, dtype='uint8')
                colorspace = colorspacedict[pattern.colorspace]
                hands = hand_lib.getHand(caliColorframe, colorframe, colorspace, pattern.edges)

                # if hands were detected visualize them
                if len(hands) > 0:
                    # if the depth is enabled read out the depth frame
                    if pattern.depth:
                        depthframe = device.getdepthstream()
                        caliDepthframe = cv2.warpPerspective(depthframe, M, (1280, 720))

                    # Print and log the fingertips
                    for i, hand in enumerate(hands):
                        hand_image = hand["hand_image"]
                        # Altering hand colors (to avoid feedback loop
                        # Option 1: Inverting the picture
                        if pattern.edges != True:
                            hand_image = cv2.bitwise_not(hand_image, hand["mask"])
                            hand_image = cv2.bitwise_and(hand_image, hand_image, mask=hand["mask"])
                        # Calculate hand centre
                        M = cv2.moments(hand["contour"])
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # if depth is enabled also visualize and log the distance
                        if pattern.depth:
                            handToTableDist = (float(tabledistance) - float(caliDepthframe[cY][cX])) / 100

                            if handToTableDist > 0 and handToTableDist < 10:
                                hand_image = height.visHeight(hand_image, handToTableDist)

                            if pattern.logging:
                                cv2.circle(hand_image, (cX, cY), 4, utils.id_to_random_color(i), -1)
                                cv2.putText(hand_image, "  " + str(handToTableDist), (cX, cY),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i), 1, cv2.LINE_AA)
                            log.write(str(timestamp) + " " + str(float(tabledistance) - float(depthframe[cY][cX])) + " H " + str(cX) + " " + str(cY) + "\n")

                            for f in hand["fingers"]:
                                if pattern.logging:
                                    cv2.circle(hand_image, f, 4, utils.id_to_random_color(i), -1)
                                    cv2.putText(hand_image, "  " + str((float(tabledistance) - float(caliDepthframe[f[1]][f[0]]))/100), f, cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i),
                                                1, cv2.LINE_AA)
                                #print("color pixel value of ", f, ":", frame[f[1]][f[0]]) # <- TODO: reverse coordinates idk why
                                #print("depth pixel value of ", f, ":", depthframe[f[1]][f[0]])
                                log.write(str(timestamp) + " " + str(float(tabledistance) - float(depthframe[cY][cX])) + " P " +  str(f[0]) + " " + str(f[1]) + "\n")
                        else:
                            if pattern.logging:
                                cv2.circle(hand_image, (cX, cY), 4, utils.id_to_random_color(i), -1)
                            # record depth as "Null"
                            log.write(str(timestamp) + " Null H " + str(cX) + " " + str(
                                cY) + "\n")
                            for f in hand["fingers"]:
                                if pattern.logging:
                                    cv2.circle(hand_image, f, 4, utils.id_to_random_color(i), -1)
                                # record depth as "Null"
                                log.write(str(timestamp) + " Null P " + str(f[0]) + " " + str(
                                    f[1]) + "\n")
                        # add the hand to the frame
                        frame = cv2.bitwise_or(frame, hand_image)

            # frame = reducer(frame, percentage=40)  # reduce frame by 40%
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
        print(bcolors.OKGREEN+"\n Session log saved: "+log.name+"\n"+bcolors.WARNING)
        log.close()
        device.stop()

# Create a async function where you want to show/manipulate your received frames
async def client_iterator(client):
    # loop over Client's Asynchronous Frame Generator
    cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    async for frame in client.recv_generator():
        # do something with received frames here
        # print("frame recieved")
        # Show output window
        cv2.imshow("Output Frame", frame)
        if overlay:
            hwnd = win32gui.FindWindow(None, "Output Frame")
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)  # no idea, but it goes together with transparency
            win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)  # black as transparent
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, GetSystemMetrics(0), GetSystemMetrics(1), 0)  # always on top
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)  # maximiced
        key = cv2.waitKey(1) & 0xFF
        # await before continuing
        await asyncio.sleep(0.00001)

async def netgear_async_playback(pattern):
    try:
        # define and launch Client with `receive_mode = True`
        server = NetGear_Async(address = PeerAddress, port = PeerPort, logging=pattern.logging)  # invalid protocol
        server.config["generator"] = custom_frame_generator(pattern)
        server.launch()
        # define and launch Client with `receive_mode = True` and timeout = 5.0
        client = NetGear_Async(port = HostPort,receive_mode=True, timeout=float("inf"), logging=pattern.logging).launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
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
    parser.add_argument("-r", "--realsense", help="Realsense device S/N")
    parser.add_argument("-s", "--standalone", help="Standalone Mode", action='store_true')
    parser.add_argument("-o", "--host", type=int, help="Host port number")
    parser.add_argument("-a", "--address", help="Peer IP address")
    parser.add_argument("-p", "--port", type=int, help="Peer port number")
    parser.add_argument("-f", "--file", help="Simulate camera sensor from .bag file")
    parser.add_argument("-d", "--depth", help="dont use depth camera (faster)", action='store_false')
    parser.add_argument("-e", "--edges", help="only visualize the edges of a hand", action='store_true')
    parser.add_argument("-c", "--colorspace",
                        help="choose the colorspace for color segmentation. Popular choice is 'hsv' but we achieved best results with 'lab'",
                        choices=['hsv', 'lab', 'ycrcb', 'rgb', 'luv', 'xyz', 'hls', 'yuv'], default='lab')
    parser.add_argument("-v", "--verbose", dest='logging', action='store_true', help="enable vidgear logging")
    options = parser.parse_args(args)
    return options

if __name__ == '__main__':
    options = getOptions(sys.argv[1:])
    DeviceSrc = options.realsense
    # configure network
    if options.standalone:
        HostPort = 5555
        PeerAddress = "localhost"
        PeerPort = 5555
    else:
        if options.host:
            HostPort = options.host
        if options.address:
            PeerAddress = options.address
        if options.port:
            PeerPort = options.port

    # configure Realsense device
    if options.file:
        DeviceSrc = options.file

    log = open("logs/log_"+str(int(time.time()))+".log", "x")
    log.write("timestamp height class x y" + "\n")
    asyncio.run(netgear_async_playback(options))
