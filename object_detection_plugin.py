# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    object_detection_plugin.py                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Paul Joseph <paul.joseph@pbl.ee.ethz.ch    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/24 11:52:09 by Paul Joseph       #+#    #+#              #
#    Updated: 2024/09/25 13:08:50 by Paul Joseph      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

 
#    ___ __  __ ____   ___  ____ _____ ____  
#   |_ _|  \/  |  _ \ / _ \|  _ \_   _/ ___| 
#    | || |\/| | |_) | | | | |_) || | \___ \ 
#    | || |  | |  __/| |_| |  _ < | |  ___) |
#   |___|_|  |_|_|    \___/|_| \_\|_| |____/ 
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import pathlib

from plugin import Plugin
import zmq_tools
from pyglui import ui

# logging
import logging
logger = logging.getLogger(__name__)

 
#     ___  _     _           _     ____       _            _   _             
#    / _ \| |__ (_) ___  ___| |_  |  _ \  ___| |_ ___  ___| |_(_) ___  _ __  
#   | | | | '_ \| |/ _ \/ __| __| | | | |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#   | |_| | |_) | |  __/ (__| |_  | |_| |  __/ ||  __/ (__| |_| | (_) | | | |
#    \___/|_.__// |\___|\___|\__| |____/ \___|\__\___|\___|\__|_|\___/|_| |_|
#             |__/                                                           
class Object_Detection(Plugin):

    #    ___       _ _    
    #   |_ _|_ __ (_) |_  
    #    | || '_ \| | __| 
    #    | || | | | | |_  
    #   |___|_| |_|_|\__| 
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.init_pupil()
        self.init_yolo()


    def init_pupil(self) -> None:
        """
        Initialize the Pupil Labs related settings.
        """
         # order (0-1) determines if your plugin should run before other plugins or after
        # gcvlc player uses high order since it relies on calculated gaze points
        self.order = .7
    
    def init_yolo(self) -> None:
        """
        Initialize the YOLOv8 model.
        """
        self.model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

    #    ____  _             _         _____                 _   _                 
    #   |  _ \| |_   _  __ _(_)_ __   |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
    #   | |_) | | | | |/ _` | | '_ \  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
    #   |  __/| | |_| | (_| | | | | | |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
    #   |_|   |_|\__,_|\__, |_|_| |_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
    #                  |___/                                                       
    def init_ui(self) -> None:
        """
        Init the user interface including text and settings buttons.

        (This is a function given by the Plugin class 
        --> see plugin.py
        --> does not need to be called explicitly)
        """
        try:
            # lets make a menu entry in the sidebar
            self.add_menu()
            # add a label to the menu
            self.menu.label = 'Object Detection'
            # add info text
            self.menu.append(ui.Info_Text('This plugin adds object detection to the scene camera.'))
        except:
            logger.error("Unexpected error: {}".format(sys.exc_info()))
        
    def deinit_ui(self) -> None:
        """
        (This is a function given by the Plugin class 
        --> see plugin.py
        --> does not need to be called explicitly)
        """
        self.remove_menu()

    def recent_events(self, events) -> None:
        """
        Handle incoming images from the scene/world camera and trigger the object detection.
        Also handle incoming gaze data to stir focus to the detected objects.

        (This is a function given by the Plugin class 
        --> see plugin.py
        --> does not need to be called explicitly)
        """

        # get the frame(aka world camera data) from the events
        frame = events.get("frame")
        if not frame:
            return
        self.object_detection(frame.img)
        
    def object_detection(self, image) -> None:
        """
        Perform object detection on a scene camera image using YOLOv8
        TODO: return the detected objects and their coordinates in a appropriate format
        """
        # Run batched inference on a list of images
        results = self.model(image)  # return a list of Results objects

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk
