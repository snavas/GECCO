# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import reducer
from classes.realsense import RealSense
import libs.hand as hand
import libs.draw as draw
import numpy as np
import cv2, asyncio

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
            frame = cv2.bitwise_or(result, drawings)
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
            #port = 6666, pattern=pattern, receive_mode=True, **c_options
            pattern=pattern, receive_mode=True
        ).launch()
        s_options = {'compression_format': '.jpg', 'compression_param': [cv2.IMWRITE_JPEG_QUALITY, 50]}
        server = NetGear_Async(
            #address = '10.67.50.132', port = 5555, pattern=pattern, **s_options
            pattern=pattern, **s_options
        )
        server.config["generator"] = custom_frame_generator()
        server.launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        #pytest.fail(str(e))
        pass
    finally:
        server.close(skip_loop=True)
        client.close(skip_loop=True)

if __name__ == '__main__':
    asyncio.run(netgear_async_playback(1))