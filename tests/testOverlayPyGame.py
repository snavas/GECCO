import win32api
import win32con
import win32gui
import pygame

# https://www.reddit.com/r/Python/comments/ha5aws/a_transparent_overlay_in_pygame/
# https://stackoverflow.com/questions/550001/fully-transparent-windows-in-pygame

pygame.init()
icon = pygame.image.load("../material/raw_output.png")
screen = pygame.display.set_mode((1365, 750), 1, pygame.NOFRAME)
pygame.display.set_icon(icon)
pygame.display.set_caption("Moving Vehicles")
left = False
pink = (255, 192, 203)  # Transparency color
x = -1000


# Variables
vehicle = pygame.image.load("../material/raw_output.png")
y = 300
endpoint = 3000
speed = 0.5

# Set window transparency color
hwnd = pygame.display.get_wm_info()["window"]
win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*pink), 0, win32con.LWA_COLORKEY)
win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0,0,0,0,
win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)


def boatie(x, y) :
    screen.blit(vehicle, (x, y))
done = True
while done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = False

    screen.fill(pink)  # Transparent background
    x += speed
    if x > endpoint:
        x = -1000

    boatie(x, y)
    pygame.display.update()