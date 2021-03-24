# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import reducer
from classes.realsense import RealSense
from classes.bcolors import bcolors
import libs.hand as hand
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
DeviceSrc = "752112070204"
#fileFlag = True

# Create a async frame generator as custom source
async def custom_frame_generator(useDepth):
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
            print(time.now() - timestamp)
            #if fileFlag:
            #    colorframe = cv2.cvtColor(colorframe, cv2.COLOR_RGB2BGR) #Reading from BAG alters the color space and needs to be fixed

            # check if frame empty
            if colorframe is None:
                break
            # process frame
            ########################
            # Calibation           #
            ########################
            global screen_corners, target_corners
            if continuousCalibration == False and len(target_corners) != 4:
                frame, screen_corners, target_corners = cal.calibrateViaARUco(colorframe, screen_corners, target_corners)
                if len(target_corners) == 4 and useDepth:
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
                results, hands, points = hand.getHand(caliColorframe, colorframe)
                #drawings = draw.getDraw(colorframe)
                #frame = cv2.bitwise_or(result, drawings)

                if hands:
                    if useDepth:
                        depthframe = device.getdepthstream()
                        caliDepthframe = cv2.warpPerspective(depthframe, M, (1280, 720))
                        # Print and log the fingertips
                        for i in range(len(hands)):
                            # Altering hand colors (to avoid feedback loop
                            # Option 1: Inverting the picture
                            results[i] = cv2.bitwise_not(results[i])
                            results[i][np.where((results[i] == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
                            # Calculate hand centre
                            M = cv2.moments(hands[i])
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])

                            handToTableDist = (float(tabledistance) - float(caliDepthframe[cY][cX])) / 100

                            if handToTableDist > 0 and handToTableDist < 10:
                                results[i] = height.visHeight(results[i], handToTableDist)

                            cv2.circle(results[i], (cX, cY), 4, utils.id_to_random_color(i), -1)
                            cv2.putText(results[i], "  " + str(handToTableDist), (cX, cY),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i), 1, cv2.LINE_AA)
                            log.write(str(timestamp) + " " + str(float(tabledistance) - float(depthframe[cY][cX])) + " H " + str(cX) + " " + str(cY) + "\n")

                            for f in points[i]:
                                cv2.circle(results[i], f, 4, utils.id_to_random_color(i), -1)
                                cv2.putText(results[i], "  " + str((float(tabledistance) - float(caliDepthframe[f[1]][f[0]]))/100), f, cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i),
                                                1, cv2.LINE_AA)
                                #print("color pixel value of ", f, ":", frame[f[1]][f[0]]) # <- TODO: reverse coordinates idk why
                                #print("depth pixel value of ", f, ":", depthframe[f[1]][f[0]])
                                log.write(str(timestamp) + " " + str(float(tabledistance) - float(depthframe[cY][cX])) + " P " +  str(f[0]) + " " + str(f[1]) + "\n")
                            frame = cv2.bitwise_or(frame, results[i])
                    else:
                        for i in range(len(hands)):
                            # Altering hand colors (to avoid feedback loop
                            # Option 1: Inverting the picture
                            results[i] = cv2.bitwise_not(results[i])
                            results[i][np.where((results[i] == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
                            frame = cv2.bitwise_or(frame, results[i])

                            # Calculate hand centre
                            M = cv2.moments(hands[i])
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            log.write(str(timestamp) + " Null H " + str(cX) + " " + str(
                                cY) + "\n")
                            for f in points[i]:
                                log.write(str(timestamp) + " Null P " + str(f[0]) + " " + str(
                                    f[1]) + "\n")

            # frame = reducer(frame, percentage=40)  # reduce frame by 40%
            # to measure time to completion
            # print(time.now() - timestamp)
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
    cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
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
        options = {'compression_param': cv2.IMREAD_COLOR}
        client = NetGear_Async(
            port = HostPort, pattern=1, receive_mode=True, **options
        ).launch()
        options = {'compression_format': '.jpg', 'compression_param': [cv2.IMWRITE_JPEG_QUALITY, 50]}
        server = NetGear_Async(
            address = PeerAddress, port = PeerPort, pattern=1, **options
        )
        server.config["generator"] = custom_frame_generator(pattern.depth)
        server.launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        print(e)
        pass
    finally:
        server.close(skip_loop=True)
        client.close(skip_loop=True)

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="PyMote")
    parser.add_argument("-s", "--standalone", help="Standalone Mode", action='store_true')
    parser.add_argument("-o", "--host", type=int, help="Host port number")
    parser.add_argument("-a", "--address", help="Peer IP address")
    parser.add_argument("-p", "--port", type=int, help="Peer port number")
    parser.add_argument("-f", "--file", help="Simulate camera sensor from .bag file")
    parser.add_argument("-d", "--depth", help="dont use depth camera (faster)", action='store_false')
    #parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options

if __name__ == '__main__':
    options = getOptions(sys.argv[1:])
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
