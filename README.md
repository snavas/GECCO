![GECCO](https://snavas.github.io/img/GECCO.png)
#

<div align="center">
  
  **Gesture-Enabled Remote Communication and Collaboration** 
  
  ![Demo2](media/example_skin.gif)

</div>

This project aims to develop a system for enabling hand gestures in a remote collaboration scenario, with a special focus on architectural design and urban planing. GECCO is a Python port of the [DCOMM system](https://github.com/snavas/DCOMM).

## Description

The GECCO prototype consists of two twin tabletop systems equipped with a depth camera (Intel RealSense). The system uses the camera together with Computer Vision algorithms to recognise and capture hand gestures, transmits those gestures to the other twin system, and then display those hand gestures on the other end table, augmenting the physical space.

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

| short| long         | value        | default   | description                                                                              |
| ---- | ------------ | ------------ | --------- | ---------------------------------------------------------------------------------------- |
| -s   | --source     | ```string``` | -         | Realsense Device S/N                                                                     |
| -r   | --remote     | ```int``` ```string``` ```int``` | -       | Set this argument to connect to another peer. Give it the host port number, the peer IP address and the port number of the peer | |
| -f   | --file       | ```string``` | -         | Path to .bag-file to simulate camera. (Does not work if camera is still connected!)      |
| -d   | --depth      | -            | -         | Dont use depth image to increase performance                                             |
| -e   | --edges      | -            | -         | only visualize the edges of a hand                                                       |
| -i   | --invisible  | -            |```False```| Gestures are not displayed. Only hand data is logged                                   |
| -c   | --colorspace | ```['hsv', 'lab', 'ycrcb', 'rgb', 'luv', 'xyz', 'hls', 'yuv']``` | ```'lab'``` | Colorspace used  for color segmentation. Popular choice is 'hsv' but we achieved best results with 'lab' |
| -v   | --verbose    | -            | -         | Enable vidgear logging and visualize position of fingers and hand center                 |
| -a   | --annotations| -            |```False```| Enable IR annotations and drawings                 |


### Example: Peer Mode
Run following command on first PC:
```
python start.py -r 5454 [IP of second PC] 5555
```
Run following command on second PC:
```
python start.py -r 5555 [IP of first PC] 5454
```

### Example: Standalone Mode
```
python start.py
```
### Example: Standalone + reading from file
The following example is very useful for testing GECCO in your own local computer without access to a Realsense camera. You need to download our [example .bag file](https://uni-muenster.sciebo.de/s/x6W2XDy0J4oUFNe)
```
python start.py -f file.bag
```
### Example: Infrared drawing
Download this [file](https://uni-muenster.sciebo.de/s/r3PaJG2CE3F9L04) and try it out with the following command
```
python start.py -a -f ir_drawing.bag
```

<div align="center">
  
![IRpen](https://user-images.githubusercontent.com/9846759/124916746-4bff4680-dff3-11eb-805c-2c3cf2d9803b.png)
  
</div>

## Known Issues

- [Intel Realsense Error 5000](https://github.com/IntelRealSense/librealsense/issues/9270)

## Modalities

GECCO is designed to support gestural-enabled remote collaboration and communication in two different modalities: traditional paper media and "modern" tabletop (touch) displays. Both modalities require a depth camera, such as Intel Real Sense, positioned on top of the representation media workspace.

### Overlayed over a tabletop display

In this modality, GECCO is intended to be used over a pair of tabletop touch displays. GECCO will act just as an overlay running on top of any software being used in the machine. For example, any mapping system, GIS system, or even an internet browser.

<div align="center">
  
![touchdisplay](https://user-images.githubusercontent.com/9846759/124916994-954f9600-dff3-11eb-98ad-a110439c4de9.png)

</div>
  
### Projected over a paper plan

In this modality, GECCO would use a projector and mirror system to project hand gestures over a traditional paper plan. GECCO's software will display the user's hand gestures over a black background screen.

<div align="center">
  
  ![Prototype](https://raw.githubusercontent.com/snavas/snavas.github.io/master/img/prototype.png)
  <br>
  ![ezgif-2-5a3153e1f425](https://user-images.githubusercontent.com/9846759/124914772-08a3d880-dff1-11eb-949b-903a3d37e4cc.gif)
  <br>
  _GECCO runing on a projector tabletop system. The communication partner's hand gestures are projected over a paper plan_

</div>









