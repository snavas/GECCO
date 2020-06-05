# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import reducer
from classes.realsense import RealSense
import libs.hand as hand
import libs.draw as draw
import libs.calibration as cal
import numpy as np
import cv2, asyncio
import argparse
import sys

HostPort = 5555
PeerAddress = "localhost"
PeerPort = 5555
calibrationMatrix = []

# Create a async frame generator as custom source
async def custom_frame_generator():
    # Open video stream
    device = RealSense("752112070204")
    # loop over stream until its terminated
    while True:
        # read frames
        colorframe = device.getcolorstream()
        depthframe = device.getdepthstream()
        # check if frame empty
        if colorframe is None:
            break
        # process frame
        if (True):
            result, hands, points = hand.getHand(colorframe, depthframe, device.getdepthscale())
            if hands:
                cv2.drawContours(result, hands, -1, (0, 255, 0), 2)
                for p in points:
                    cv2.circle(result, tuple(p), 2, (0, 0, 255))
            drawings = draw.getDraw(colorframe)
            inter = cv2.bitwise_or(result, drawings)
            global calibrationMatrix
            frame, calibrationMatrix = cal.getDraw(colorframe, inter, calibrationMatrix)
            if len(calibrationMatrix) == 4:
                #crop_img = img[y:y+h, x:x+w]
                #frame = frame
                print(calibrationMatrix)
        else:
            frame = colorframe
        # frame = reducer(frame, percentage=40)  # reduce frame by 40%
        # yield frame
        yield frame
        # sleep for sometime
        await asyncio.sleep(0.00001)
    # close stream
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
        key = cv2.waitKey(1) & 0xFF
        # await before continuing
        await asyncio.sleep(0.00001)

async def netgear_async_playback(pattern):
    try:
        # define and launch Client with `receive_mode = True`
        c_options = {'compression_param': cv2.IMREAD_COLOR}
        client = NetGear_Async(
            port = HostPort, pattern=1, receive_mode=True, **c_options
        ).launch()
        s_options = {'compression_format': '.jpg', 'compression_param': [cv2.IMWRITE_JPEG_QUALITY, 50]}
        server = NetGear_Async(
            address = PeerAddress, port = PeerPort, pattern=1, **s_options
        )
        server.config["generator"] = custom_frame_generator()
        server.launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        pass
    finally:
        server.close(skip_loop=True)
        client.close(skip_loop=True)

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="PyMote")
    parser.add_argument("-s", "--standalone", help="Standalone Mode")
    parser.add_argument("-o", "--host", type=int, help="Host port number")
    parser.add_argument("-a", "--address", help="Peer IP address")
    parser.add_argument("-p", "--port", type=int, help="Peer port number")
    #parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options

if __name__ == '__main__':
    options = getOptions(sys.argv[1:])
    if options.host:
        HostPort = options.host
    if options.address:
        PeerAddress = options.address
    if options.port:
        PeerPort = options.port
    if options.standalone:
        HostPort = 5555
        PeerAddress = "localhost"
        PeerPort = 5555
    asyncio.run(netgear_async_playback(options))