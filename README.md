# GECCO
Gesture-Enabled Remote Communication and Collaboration

<div align="center">
 
 ![Demo](https://snavas.github.io/img/image4.gif)

</div>

This project aims to develop a system for enabling hand gestures in a remote collaboration scenario, with a special focus on architectural design and urban planing. GECCO is a Python port of the [DCOMM system](https://github.com/snavas/DCOMM).

## Requirements
- [pyrealsense2](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)
GECCO is implemented to work by default using Intel Real Sense D415 and D430 sensors. Pyrealsense2 package is our official wrapper which does support SDK 2.0. You can install pyrealsense via pip if using a Python version up to 3.7.
```
pip install pyrealsense2
```
Otherwise, windows users can install the RealSense SDK 2.0 from the release tab to get pre-compiled binaries of the wrapper, for both x86 and x64 architectures. Pyrealsense2 can also be compiled from source code with the CMake tool (both Python 2.7 and Python 3 are supported). 

https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#building-from-source.

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
- sklearn
```
pip install sklearn
```


## Installation
```
git clone https://github.com/snavas/GECCO.git
cd GECCO
```

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

