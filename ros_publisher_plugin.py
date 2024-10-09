# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ros_publisher_plugin.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Paul Joseph <paul.joseph@pbl.ee.ethz.ch    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/10/03 10:48:51 by Paul Joseph       #+#    #+#              #
#    Updated: 2024/10/09 08:37:24 by Paul Joseph      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#    ___ __  __ ____   ___  ____ _____ ____  
#   |_ _|  \/  |  _ \ / _ \|  _ \_   _/ ___| 
#    | || |\/| | |_) | | | | |_) || | \___ \ 
#    | || |  | |  __/| |_| |  _ < | |  ___) |
#   |___|_|  |_|_|    \___/|_| \_\|_| |____/ 
import sys
import cv2
# import numpy as np
import pathlib

# pupil 
from plugin import Plugin
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_circle, draw_rounded_rect
from pyglui.pyfontstash import fontstash

# custom
from event_handler.event_handler import EventHandler

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from pupil_labs_ros2_msgs.msg import GazeStamped
from cv_bridge import CvBridge

# logging
import logging
logger = logging.getLogger(__name__)

class ROS_Publisher_Pugin(Plugin):

    #    ___       _ _    
    #   |_ _|_ __ (_) |_  
    #    | || '_ \| | __| 
    #    | || | | | | |_  
    #   |___|_| |_|_|\__| 
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.init_pupil()
        self.init_ros()

    def init_pupil(self) -> None:
        """
        Initialize the Pupil Labs related settings.
        """
        # order (0-1) determines if your plugin should run before other plugins or after
        # gcvlc player uses high order since it relies on calculated gaze points
        self.order = .7

        # custom made class, but still belongs to pupil stuff
        # (handles incoming pupil events)
        self.event_handler = EventHandler()
    
    def init_ros(self) -> None:
        """
        Initialize ROS and the Custom ROS Node.
        """
        rclpy.init()
        # init seperate class for ROS stuff
        self.node_name = 'PupilRosNode'
        self.ros_node = PupilRosNode(self.node_name)

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
        self.__sub_menu = None

        try:
            # lets make a menu entry in the sidebar
            self.add_menu()
            # add a label to the menu
            self.menu.label = 'ROS Translation'
            # add info text
            self.menu.append(ui.Info_Text('This plugin publishes data in form of ROS2 Topics.'))
            
            # add a sub menu  
            self.__sub_menu = ui.Growing_Menu('Settings')
            self.menu.append(self.__sub_menu)
            # add a text input to the sub menu
            self.__sub_menu.append(
                ui.Text_Input(
                    "node_name", 
                    self, 
                    label="Node Name", 
                    setter=self.restart_ros_node
                )
            )


            self.set_ui_font()
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
        Handle incoming data and call needed publishers

        (This is a function given by the Plugin class 
        --> see plugin.py
        --> does not need to be called explicitly)
        """

        # get the frame(aka world camera data) from the events
        frame   = self.event_handler.get_frame(events)
        self.publish_frame(frame)
        # get the gaze data from the events
        gaze    = self.event_handler.get_highest_conf_gaze(events)
        self.publish_gaze(gaze)
        # get the imu data from the events
        # imu     = self.event_handler.get_imu(events)
        # self.publish_imu(imu)

 
    #     ____          _                    _____                 _   _                 
    #    / ___|   _ ___| |_ ___  _ __ ___   |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
    #   | |  | | | / __| __/ _ \| '_ ` _ \  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
    #   | |__| |_| \__ \ || (_) | | | | | | |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
    #    \____\__,_|___/\__\___/|_| |_| |_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
    def publish_frame(self, frame) -> None:
        """
        Publish the frame data to the ROS2 topic.
        """
        if frame is None:
            print("No frame data available.")
            return
        #   convert the frame to ROS2 Image
        frame_msg = self.ros_node.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        #   publish the frame
        self.ros_node.pub.frame.image.publish(frame_msg)
    
    def publish_gaze(self, gaze) -> None:
        """
        Publish the gaze data to the ROS2 topic.
        """
        # check if gaze data is available
        if not gaze:
            print("No gaze data available.")
            return
        #   convert the frame to ROS2 Image
        gaze_msg = GazeStamped()
        gaze_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        gaze_msg.header.frame_id = 'gaze'
        gaze_msg.gaze.x = gaze['denorm_pos'][0] # TODO: check if this is correct
        gaze_msg.gaze.y = gaze['denorm_pos'][1] # TODO: check if this is correct
        gaze_msg.image_size.width  = gaze['frame_size'][0]
        gaze_msg.image_size.height = gaze['frame_size'][1]

        #   publish the frame
        self.ros_node.pub.gaze.publish(gaze_msg)

    def restart_ros_node(self, node_name) -> None:
        """
        Restart the ROS2 Node with the new name.
        """
        self.node_name = node_name
        self.ros_node.destroy_node()
        self.ros_node = PupilRosNode(self.node_name)

    def set_ui_font(self):
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))


#    ____   ___  ____    _   _           _      
#   |  _ \ / _ \/ ___|  | \ | | ___   __| | ___ 
#   | |_) | | | \___ \  |  \| |/ _ \ / _` |/ _ \
#   |  _ <| |_| |___) | | |\  | (_) | (_| |  __/
#   |_| \_\\___/|____/  |_| \_|\___/ \__,_|\___|
class PupilRosNode(Node):
    #    ___       _ _   
    #   |_ _|_ __ (_) |_ 
    #    | || '_ \| | __|
    #    | || | | | | |_ 
    #   |___|_| |_|_|\__|
    def __init__(self, node_name): 
        self.node_name = node_name
        super().__init__(self.node_name)
        self.init_publishers()

    def init_publishers(self) -> None:
        """
        Init publishers for:
            - Frame
            - Gaze
            - IMU
        """
        self.pub = RosPublishers()
        #   use cv bridge to handle cv2 to ROS convertion
        self.cv_bridge = CvBridge()
        #   publisher for pretty pictures
        self.pub.frame.image = self.create_publisher(Image, self.node_name + '/frame/image', 10)
        #   publisher for the camera info
        self.pub.frame.info = self.create_publisher(CameraInfo, self.node_name + '/frame/info', 10)
        #   publisher for the gaze data
        self.pub.gaze = self.create_publisher(GazeStamped, self.node_name + '/gaze', 10)
        #   publisher for the imu data
        self.pub.imu = self.create_publisher(Imu, self.node_name + '/imu', 10)

class RosPublishers():
    def __init__(self): 
        self.frame      = RosCameraPublisher()
        self.gaze       = None
        self.imu        = None

class RosCameraPublisher():
    def __init__(self):
        self.image      = None
        self.info       = None