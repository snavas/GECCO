#from device import Device
from classes.device import Device
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSense(Device):
    #pipeline = rs.pipeline()

    def getcolorintrinsics(self):
        return self.color_intr

    def getdepthintrinsics(self):
        return self.depth_intr

    # overriding abstract method
    def __init__(self, id):
        ctx = rs.context()
        devices = ctx.query_devices()
        print("<*> Connected devices: ")
        print(*devices, sep="\n")

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        #config.enable_device(id)
        #print("<*> Using device: ", id)
        print("<*> Using device: ", devices[0])
        #for sensor in devices[0].query_sensors():
        #    print(sensor)
        if "D415" in str(devices[0]):
            #print("Resolution: 1280x720")
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        else:
            #print("Resolution: 848x480")
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        # Start streaming
        profile = self.pipeline.start(config)
        self.color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("<*> Depth Scale is: ", self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1.15  # 1 meter
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def getdepthscale(self):
        return self.depth_scale

    # overriding abstract method
    # Streaming loop
    def getstream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        aligned_color_frame = aligned_frames.get_color_frame()

        #if not depth_frame or not color_frame:
        #    continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))
        return depth_colormap

    def getcolorstream(self):
        frames = self.pipeline.wait_for_frames()
        aligned_color_frame = self.align.process(frames).get_color_frame()
        color_image = np.asanyarray(aligned_color_frame.get_data())
        return color_image

    def getdepthstream(self):
        frames = self.pipeline.wait_for_frames()
        aligned_depth_frame = self.align.process(frames).get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        #scaled_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        return depth_image

    def getdepthcolormap(self):
        depth_image = self.getdepthstream()
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        return depth_colormap

    def getsegmentedstream(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        grey_color = 255
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        return bg_removed
    def stop(self):
        self.pipeline.stop()

    def restart(self):
        self.pipeline.start()