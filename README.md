# PyMote

<div align="center">
 
 ![Demo](https://snavas.github.io/img/image4.gif)

</div>

## Goal
 
This project aims to develop a system for enabling hand gestures in a remote collaboration scenario, with a special focus on architectural design and using architectural paper plans.

## Dependencies
- numpy
```
pip install numpy
```
- opencv
```
pip install opencv-contrib-python
```
- [vidgear (asyncio)](https://github.com/abhiTronix/vidgear)
```
git clone https://github.com/abhiTronix/vidgear.git
cd vidgear
git checkout testing
pip install .[asyncio]           # installs all required dependencies including asyncio 
```
- pyrealsense2
```
pip install pyrealsense2
```

## Installation

## Use

### Peer Mode
```
python start.py -o [HOST PORT NUMBER] -a [PEER IP ADDRESS] -p [PEER PORT NUMBER]
```
### Standalone Mode
```
python start.py -s
```
## Known Issues

## Hardware Description

This prototype consists of two twin tabletop systems with a depth camera and a projector attached to them. The system uses the depth camera (Intel RealSense) together with Computer Vision algorithms to recognise and capture hand gestures, transmits those gestures to the other twin system, and then projects those hand gestures on the other end table, augmenting the physical space. However, the scope of the current system is quite limited, as it only detects, logs, transmit and displays a number of hand gestures to the other end, and vice versa.

![Prototype](https://raw.githubusercontent.com/snavas/snavas.github.io/master/img/prototype.png)

