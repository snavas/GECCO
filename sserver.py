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
import traceback

HostPort = 5555
PeerAddress = "localhost"
PeerPort = 5555
calibrationMatrix = []
oldCalibration = False
continuousCalibration = False
DeviceSrc = "C:/Users/s_nava02/Documents/GECCO/20210217_113804.bag"
#fileFlag = True

# initialize Server
server = NetGear_Async(logging=True)

# Create a async frame generator as custom source
async def custom_frame_generator():
    try:
        # Get global log variable
        global log
        tabledistance = 1200 # Default distance to table
        # Open video stream
        device = RealSense("C:/Users/s_nava02/Documents/GECCO/20210217_113804.bag")
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
            if continuousCalibration == False and len(calibrationMatrix) != 4:
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
                #drawings = draw.getDraw(colorframe)
                #frame = cv2.bitwise_or(result, drawings)
                frame = result
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
        print(bcolors.FAIL + traceback.format_exc() + bcolors.ENDC)
    finally:
        log.flush()
        print(bcolors.OKGREEN+"\n Session log saved: "+log.name+"\n"+bcolors.WARNING)
        log.close()
        device.stop()

if __name__ == "__main__":
    log = open("logs/log_" + str(int(time.time())) + ".log", "x")
    # set event loop
    asyncio.set_event_loop(server.loop)
    # Add your custom source generator to Server configuration
    server.config["generator"] = custom_frame_generator()
    # Launch the Server
    server.launch()
    try:
        # run your main function task until it is complete
        server.loop.run_until_complete(server.task)
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    finally:
        # finally close the server
        server.close()