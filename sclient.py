# import libraries
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio
import win32api
import win32con
import win32gui
from win32api import GetSystemMetrics

overlay = True

# define and launch Client with `receive_mode=True`
client = NetGear_Async(receive_mode=True, logging=True).launch()

# Create a async function where you want to show/manipulate your received frames
async def main():
    cv2.namedWindow("Output Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    # loop over Client's Asynchronous Frame Generator
    async for frame in client.recv_generator():
        # {do something with received frames here}
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
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass

    # close all output window
    cv2.destroyAllWindows()
    # safely close client
    client.close()