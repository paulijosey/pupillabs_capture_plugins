# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    object_detection_plugin.py                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Paul Joseph <paul.joseph@pbl.ee.ethz.ch    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/24 11:52:09 by Paul Joseph       #+#    #+#              #
#    Updated: 2024/10/03 13:03:22 by Paul Joseph      ###   ########.fr        #
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
import seaborn as sns
import matplotlib.colors as mcolors

# pupil
from plugin import Plugin
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_circle, draw_rounded_rect
from pyglui.pyfontstash import fontstash

# custom
from event_handler import event_handler

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
        self.eventHandler = EventHandler()
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
        self.recent_objects = None

    #    ____  _             _         _____                 _   _                 
    #   |  _ \| |_   _  __ _(_)_ __   |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
    #   | |_) | | | | |/ _` | | '_ \  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
    #   |  __/| | |_| | (_| | | | | | |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
    #   |_|   |_|\__,_|\__, |_|_| |_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
    #                  |___/                                                   
    # 
    # Following are functions that are inherited by the Plugin class. Calling those 
    # functions is already handled by PubilLabs. For more information and more 
    # available functions check the pupil_src/shared_modules/plugin.py file.    
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
            
            self.glfont = fontstash.Context()
            self.glfont.add_font("opensans", ui.get_opensans_font_path())
            self.glfont.set_size(22)
            self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))
        except:
            logger.error("Unexpected error: {}".format(sys.exc_info()))
        
    def deinit_ui(self) -> None:
        """
        (This is a function given by the Plugin class 
        --> see plugin.py
        --> does not need to be called explicitly)
        """
        self.remove_menu()
        self.glfont = None

    def recent_events(self, events) -> None:
        """
        Handle incoming images from the scene/world camera and trigger the object detection.
        Also handle incoming gaze data to stir focus to the detected objects.

        (This is a function given by the Plugin class 
        --> see plugin.py
        --> does not need to be called explicitly)
        """

        # get the frame(aka world camera data) from the events
        frame = self.event_handler.get_frame(events)
        self.object_detection(frame)

        # TODO: use gaze data
        gaze = self.event_handler.get_gaze(events)

 
    def gl_display(self):
        """
        Overlay information on the image displayed in the GUI.

        (This is a function given by the Plugin class 
        --> see plugin.py
        --> does not need to be called explicitly)
        """

        fs = self.g_pool.capture.frame_size  # frame height
        # get object detection results
        self.visualize_objects()
 
    #     ____          _                    _____                 _   _                     
    #    / ___|   _ ___| |_ ___  _ __ ___   |  ___|   _ _ __   ___| |_(_) ___  _ __  ___     
    #   | |  | | | / __| __/ _ \| '_ ` _ \  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|    
    #   | |__| |_| \__ \ || (_) | | | | | | |  _|| |_| | | | | (__| |_| | (_) | | | \__ \    
    #    \____\__,_|___/\__\___/|_| |_| |_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/    
    def object_detection(self, image: np.array) -> None:
        """
        Perform object detection on a scene camera image using YOLOv8. Taken and modified 
        from: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
        TODO: return the detected objects and their coordinates in a appropriate format
        """
        # Run batched inference on a list of images
        self.recent_objects = self.model(image, verbose=False, stream=True)  # return a list of Results objects
        # TODO: add ROS publisher

    #     ____          _                   __     ___                 _ _          _   _             
    #    / ___|   _ ___| |_ ___  _ __ ___   \ \   / (_)___ _   _  __ _| (_)______ _| |_(_) ___  _ __  
    #   | |  | | | / __| __/ _ \| '_ ` _ \   \ \ / /| / __| | | |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
    #   | |__| |_| \__ \ || (_) | | | | | |   \ V / | \__ \ |_| | (_| | | |/ / (_| | |_| | (_) | | | |
    #    \____\__,_|___/\__\___/|_| |_| |_|    \_/  |_|___/\__,_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
    def visualize_objects(self) -> None:
        """
        Mark objects of the same class is the same colors and overlay image
        with bounding boxes in those colors. Additionally add the class and 
        confidence in the top left corner.

        This function should be called in pupillabs "gl_display" method.
        """
        if self.recent_objects is not None:
            # Process results list
            for obj in self.recent_objects:
                boxes = obj.boxes  # Boxes object for bounding box outputs
                cls_dict = obj.names
                # create an evenly spread color spectrum acording to the classes in cls_dict
                color_palette = sns.color_palette(None, len(cls_dict))           
                # iterate through the bounding boxes and display them with distinct colors 
                # for each class
                for box in boxes:
                    # draw bounding box
                    wh = [box.xywh[0][2], box.xywh[0][3]]
                    top_left = [box.xywh[0][0] - wh[0]/2 , box.xywh[0][1] - wh[1]/2]
                    color = color_palette[int(box.cls[0].item())]
                    draw_rounded_rect(top_left, wh, 2.0, RGBA(color[0], color[1], color[2], 0.5))
                    # draw object label
                    self.glfont.draw_text(top_left[0], top_left[1], str(cls_dict[box.cls[0].item()]) + " (" + str(box.conf[0].item()) + ")" )