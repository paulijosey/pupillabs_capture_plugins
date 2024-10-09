# pupillabs_capture_plugins

## Dependecies
A working Pupil software. Recomended to get the dockerized setup:
https://github.com/paulijosey/pupil

## Setup
Clone me into your pupil/capture_settings folder. If it does not exist create it.

## Development
This repository consists of two different plugins:
   - Object Detection
   - ROS

Object detection handles integrating YOLO into the Pupillabs Capture software. 
ROS on the other hand handles translating data from pupil-labs format into a 
ROS2 message. All plugins should work independently if not otherwise stated. 
(They might be split up in later iterations of this project)

The documentation from pupillabs is mostly lacking. I tried to document those 
two plugins fairly well so they might be easy to adapt or help to guid you in creating
your own. 

While working on this project create a branch acording to following naming scheme:
   - If new feature: feature/\<name of fance new feature\>
   - If bug fix: fix/\<name of nasty bug\>
   - If refactor: refactor/\<think of something\> 

Once your feature/fix is ready create a merge request and let me 
(paul.joseph@pbl.ee.ethz.ch) know. 