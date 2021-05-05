# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import reducer
from classes.realsense import RealSense
from classes.bcolors import bcolors
import libs.hand as hand_lib
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
import traceback
from torchvision import models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

deeplab = models.segmentation.fcn_resnet50(pretrained=0,
                                                 progress=1,
                                                 num_classes=2)
class HandSegModel(nn.Module):
    def __init__(self):
        super(HandSegModel,self).__init__()
        self.dl = deeplab

    def forward(self, x):
        y = self.dl(x)['out']
        return y

#can pass np array or path to image file
def SegmentHands(pathtest):

    if isinstance(pathtest, np.ndarray):
        img = Image.fromarray(pathtest)
    else :
        img = Image.open(pathtest)

    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    Xtest = preprocess(img)

    checkpoint = torch.load('checkpoints/pytorch/checkpointhandseg3.pt')
    model = HandSegModel()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        model.to(device)
        Xtest = Xtest.to(device).float()
        ytest = model(Xtest.unsqueeze(0).float())
        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()
        ytest = ypos >= yneg

    mask = ytest.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,kernel,iterations = 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask
#################################################################################
def getcoloredMask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] += mask.astype('uint8') * 250
    masked = cv2.addWeighted(image, 1.0, color_mask, 1.0, 0.0)
    return masked


HostPort = 5555
PeerAddress = "localhost"
PeerPort = 5555
screen_corners = []
target_corners = []
oldCalibration = False
continuousCalibration = False
overlay = True
DeviceSrc = "752112070204"
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
            # if fileFlag:
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
                frame, screen_corners, target_corners = cal.calibrateViaARUco(colorframe, screen_corners,
                                                                              target_corners)
            else:
                # print(depthframe[int(calibrationMatrix[0][1])][int(calibrationMatrix[0][0])])
                # print("newtabledistance = ", depthframe[calibrationMatrix[0][1]][calibrationMatrix[0][0]])

                M = cv2.getPerspectiveTransform(target_corners, screen_corners)
                # TODO: derive resolution from width and height of original frame?
                caliColorframe = cv2.warpPerspective(colorframe, M, (1280, 720))

                ########################
                # Hand Detection       #
                ########################
                frame = np.zeros(colorframe.shape, dtype='uint8')
                colorspace = colorspacedict[pattern.colorspace]
                hands = hand_lib.getHand(caliColorframe, colorframe, colorspace, pattern.edges)
                if len(hands) > 0:

                    # Print and log the fingertips
                    for i, hand in enumerate(hands):
                        hand_image = hand["hand_image"]
                        # add the hand to the frame
                        frame = cv2.bitwise_or(frame, hand_image)

                    pytorchMask = SegmentHands(hands[0]["hand_crop"][0]["crop"])
                    frame = pytorchMask#getcoloredMask(hands[0]["hand_crop"][0], pytorchMask)
            # yield frame
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
            #win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, GetSystemMetrics(0), GetSystemMetrics(1), 0)  # always on top
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
        server.config["generator"] = custom_frame_generator(pattern)
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