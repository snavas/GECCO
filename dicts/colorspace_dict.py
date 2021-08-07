import cv2

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