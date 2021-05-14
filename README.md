![GECCO](https://snavas.github.io/img/GECCO.png)
#

<div align="center">
  
  **Gesture-Enabled Remote Communication and Collaboration** 
  
  ![Demo](https://user-images.githubusercontent.com/9846759/112317413-50099180-8cac-11eb-9730-04099069b2ee.gif)
  <br>
  [longer demo video](https://snavas.github.io/img/2021-03-24%2013-31-11-1.mp4) · [debug view](https://snavas.github.io/img/ezgif-4-2da419ef8f6e.gif)

</div>

This project aims to develop a system for enabling hand gestures in a remote collaboration scenario, with a special focus on architectural design and urban planing. GECCO is a Python port of the [DCOMM system](https://github.com/snavas/DCOMM).


## Requirements
- [pyrealsense2](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)

GECCO is implemented to work by default using Intel Real Sense D415 and D430 sensors. The pyrealsense2 package is Intel's RealSense SDK 2.0 python wrapper. You can install pyrealsense via pip if using a Python version up to 3.7.
```
pip install pyrealsense2
```
Otherwise, windows users can install the RealSense SDK 2.0 from the release tab to get pre-compiled binaries of the wrapper, for both x86 and x64 architectures. Pyrealsense2 can also be compiled from source code with the CMake tool (both Python 2.7 and Python 3 are supported). 

https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#building-from-source.

## Dependencies
All dependencies are listed in the ```requirements.txt``` and can be installed from the console with the following command: 

```pip install -r requirements.txt ```


## Installation
```
git clone https://github.com/snavas/GECCO.git
cd GECCO
```

## Use

### Arguments

| short| long         | value        | default | description                                                                              |
| ---- | ------------ | ------------ | ------- | ---------------------------------------------------------------------------------------- |
| -r   | --realsense  | ```string``` | -       | Realsense Device S/N                                                                     |
| -s   | --standalone | -            | -       | Standalone mode                                                                          |
| -o   | --host       |  ```int```   | -       | Host port number                                                                         |
| -a   | --address    | ```string``` | -       | Peer IP address                                                                          |
| -p   | --port       | ```int```    | -       | Peer port number                                                                         |
| -f   | --file       | ```string``` | -       | Path to .bag-file to simulate camera. (Does not work if camera is still connected!)      |
| -d   | --depth      | -            | -       | Dont use depth image to increase performance                                             |
| -e   | --edges      | -            | -       | only visualize the edges of a hand                                                       |
| -i   | --invisible  | -            |```False```| Gestures are not displayed. Only hand data is logged                                   |
| -c   | --colorspace | ```['hsv', 'lab', 'ycrcb', 'rgb', 'luv', 'xyz', 'hls', 'yuv']``` | ```'lab'``` | Colorspace used  for color segmentation. Popular choice is 'hsv' but we achieved best results with 'lab' |
| -v   | --verbose    | -            | -       | Enable vidgear logging and visualize position of fingers and hand center                 |


### Example: Peer Mode
Run following command on first PC:
```
python start.py -o 5454 -a [IP of second PC] -p 5555
```
Run following command on second PC:
```
python start.py -o 5555 -a [IP of first PC] -p 5454
```

### Example: Standalone Mode
```
python start.py -s
```
### Example: Standalone + reading from file
The following example is very useful for testing GECCO in your own local computer without access to a Realsense camera. You need to download our [example .bag file](https://uni-muenster.sciebo.de/s/x6W2XDy0J4oUFNe)
```
python start.py -s -f file.bag
```
## Known Issues

## Description

This prototype consists of two twin tabletop systems with a depth camera and a projector attached to them. The system uses the depth camera (Intel RealSense) together with Computer Vision algorithms to recognise and capture hand gestures, transmits those gestures to the other twin system, and then projects those hand gestures on the other end table, augmenting the physical space.

<div align="center">
  
  ![Prototype](https://raw.githubusercontent.com/snavas/snavas.github.io/master/img/prototype.png)
  <br>
  _GECCO runing on a projector tabletop system. The communication partner's hand gestures are projected over a paper plan_

</div>



