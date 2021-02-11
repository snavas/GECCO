# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import reducer
from classes.realsense import RealSense
from classes.bcolors import bcolors
import libs.hand as hand
import libs.draw as draw
import libs.calibration as cal
import libs.utils as utils
import numpy as np
import cv2, asyncio
import argparse
import sys
import time
import win32api
import win32con
import win32gui
from win32api import GetSystemMetrics

HostPort = 5555
PeerAddress = "localhost"
PeerPort = 5555
calibrationMatrix = []
oldCalibration = False
overlay = True
DeviceSrc = "752112070204"
#fileFlag = True

# Create a async frame generator as custom source
async def custom_frame_generator():
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
            # read frames
            colorframe = device.getcolorstream()
            #if fileFlag:
            #    colorframe = cv2.cvtColor(colorframe, cv2.COLOR_RGB2BGR) #Reading from BAG alters the color space and needs to be fixed
            depthframe = device.getdepthstream()
            # store time in seconds since the epoch (UTC)
            timestamp = time.time()
            # check if frame empty
            if colorframe is None:
                break
            # process frame
            ########################
            # Calibation           #
            ########################
            global calibrationMatrix
            global oldCalibration
            frame, newcalibrationMatrix = cal.calibrateViaARUco(colorframe, depthframe, calibrationMatrix)
            if calibrationMatrix == newcalibrationMatrix:
                oldCalibration = True
            else:
                oldCalibration = False
            calibrationMatrix = newcalibrationMatrix
            if len(calibrationMatrix) == 4:
                #print(depthframe[int(calibrationMatrix[0][1])][int(calibrationMatrix[0][0])])
                #print("newtabledistance = ", depthframe[calibrationMatrix[0][1]][calibrationMatrix[0][0]])
                tabledistance = depthframe[int(calibrationMatrix[0][1])][int(calibrationMatrix[0][0])]
                # TODO: this solution is too simple, it needs better maths to create a more robust solution
                # TODO: put all this code into a function?
                minx = min((calibrationMatrix[0][0], calibrationMatrix[1][0], calibrationMatrix[2][0],
                            calibrationMatrix[3][0]))
                miny = min((calibrationMatrix[0][1], calibrationMatrix[1][1], calibrationMatrix[2][1],
                            calibrationMatrix[3][1]))
                maxx = max((calibrationMatrix[0][0], calibrationMatrix[1][0], calibrationMatrix[2][0],
                            calibrationMatrix[3][0]))
                maxy = max((calibrationMatrix[0][1], calibrationMatrix[1][1], calibrationMatrix[2][1],
                            calibrationMatrix[3][1]))
                height = (maxy - miny)
                width = (maxx - minx)
                # TODO: Are colorframe and depthframe totally aligned? E.g. same dimmensions¿?
                # print(len(colorframe), " ", len(depthframe)) #TODO: Same dimmensions, hopefully aligned
                colorframe = colorframe[int(miny):int(miny + height), int(minx):int(minx + width)]
                colorframe = cv2.resize(colorframe,(1280, 720)) #TODO: Necessary? Might affect network performance
                depthframe = depthframe[int(miny):int(miny + height), int(minx):int(minx + width)]
                depthframe = cv2.resize(depthframe,(1280, 720)) #TODO: Necessary? Might affect network performance
                #print(calibrationMatrix)

                ########################
                # Hand Detection       #
                ########################
                result, hands, points = hand.getHand(colorframe, depthframe, device.getdepthscale())
                drawings = draw.getDraw(colorframe)
                frame = cv2.bitwise_or(result, drawings)
                # Altering hand colors (to avoid feedback loop
                # Option 1: Inverting the picture
                frame = cv2.bitwise_not(frame)
                frame[np.where((frame == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

                #if (oldCalibration):
                #    cv2.putText(frame, "CALIBRATED (OLD)", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                #                (0, 255, 255), 1, cv2.LINE_AA)
                #    cv2.putText(frame, "DISTANCE TO TABLE: "+str(tabledistance), (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                #                (0, 255, 255), 1, cv2.LINE_AA)
                #else:
                #    cv2.putText(frame, "CALIBRATED (4)", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0),
                #                1, cv2.LINE_AA)
                #    cv2.putText(frame, "DISTANCE TO TABLE: "+str(tabledistance), (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0),
                #                1, cv2.LINE_AA)

                if hands:
                # Print and log the fingertips
                    for i in range(len(hands)):
                        # Calculate hand centre
                        M = cv2.moments(hands[i])
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cX, cY), 4, utils.id_to_random_color(i), -1)
                        cv2.putText(frame, "  " + str((tabledistance - depthframe[cY][cX]) / 100), (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i), 1, cv2.LINE_AA)
                        string = "T " + str(timestamp) + " DH " + str(tabledistance - depthframe[cY][cX])
                        for f in points[i]:
                            cv2.circle(frame, f, 4, utils.id_to_random_color(i), -1)
                            cv2.putText(frame, "  " + str((tabledistance - depthframe[f[1]][f[0]])/100), f, cv2.FONT_HERSHEY_SIMPLEX, 0.25, utils.id_to_random_color(i),
                                            1, cv2.LINE_AA)
                            #print("color pixel value of ", f, ":", frame[f[1]][f[0]]) # <- TODO: reverse coordinates idk why
                            #print("depth pixel value of ", f, ":", depthframe[f[1]][f[0]])
                            string += " P " + str(f)
                        log.write(string+"\n")
                else:
                    pass
                    #print("Unable to calibrate")
                    #frame = colorframe
                    #cv2.putText(frame, "CALIBRATED (4)", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
                    #cv2.putText(frame, "NOT CALIBRATED", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
            # frame = reducer(frame, percentage=40)  # reduce frame by 40%
            # yield frame
            yield frame
            # sleep for sometime
            await asyncio.sleep(0.00001)
        # close stream
        device.stop()
        # close file
        log.close()
    except Exception as e:
        print(bcolors.FAIL+e+bcolors.ENDC)
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
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, -5, -5, GetSystemMetrics(0), GetSystemMetrics(1), 0)  # always on top
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
        server.config["generator"] = custom_frame_generator()
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
    asyncio.run(netgear_async_playback(options))