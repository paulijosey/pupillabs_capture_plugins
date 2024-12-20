# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ros_publisher_plugin.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Paul Joseph <paul.joseph@pbl.ee.ethz.ch    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/10/03 10:48:51 by Paul Joseph       #+#    #+#              #
#    Updated: 2024/11/20 15:29:46 by Paul Joseph      ###   ########.fr        #
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
from yolo_ros2.msg import Detections, Detection
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
        self.order = .5

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

        # settings
        # TODO: use config file for settings
        self.publish_frame_bool = True
        self.publish_depth_frame_bool = True
        self.publish_gaze_bool = True
        self.publish_imu_bool = False
        self.publish_objects_bool = True

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
            # Node Name 
            self.__sub_menu.append(ui.Info_Text('Change the name of the ROS2 Node.'))
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

            # selct topics to publish
            self.__sub_menu.append(ui.Info_Text('Select the topics to publish.'))
            # use the checkbox to select the topics
            #   Those switches will flip the bool value of the vaiable (first value passed
            #   .... as a string .... don't ask me why they did it like this).
            #   So "self.__first_passed_value" will be affected.
            self.__sub_menu.append(
                ui.Switch(
                    'publish_frame_bool', 
                    self, 
                    label='Frame Image',
                )
            )
            self.__sub_menu.append(
                ui.Switch(
                    'publish_depth_frame_bool', 
                    self, 
                    label='Depth Image',
                )
            )
            self.__sub_menu.append(
                ui.Switch(
                    'publish_gaze_bool', 
                    self, 
                    label='Gaze',
                )
            )
            self.__sub_menu.append(
                ui.Switch(
                    'publish_imu_bool', 
                    self, 
                    label='IMU (Not implemented)',
                )
            )
            self.__sub_menu.append(
                ui.Switch(
                    'publish_objects_bool', 
                    self, 
                    label='Objects (Not implemented)',
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

        if self.publish_frame_bool:
            # get the frame(aka world camera data) from the events
            frame = self.event_handler.get_frame(events)
            self.publish_frame(frame)

        if self.publish_depth_frame_bool:
            # get the frame(aka world camera data) from the events
            depth_frame = self.event_handler.get_depth_frame(events)
            self.publish_depth_frame(depth_frame)

        if self.publish_gaze_bool:
            # get the gaze data from the events
            gaze = self.event_handler.get_highest_conf_gaze(events)
            self.publish_gaze(gaze)

        if self.publish_imu_bool:
            print("IMU publishing is not implemented yet.")
            # get the imu data from the events
            # imu = self.event_handler.get_imu(events)
            # self.publish_imu(imu)

        if self.publish_objects_bool:
            # get the object data from the events
            objects = self.event_handler.get_objects(events)
            self.publish_objects(objects)
        
    
    def cleanup(self) -> None:
        """
        Cleanup the ROS2 Node and shutdown the ROS2 environment.
        """
        self.ros_node.destroy_node()
        rclpy.shutdown()

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

    def publish_depth_frame(self, depth_frame) -> None:
        """
        Publish the frame data to the ROS2 topic.
        """
        if depth_frame is None:
            print("No depth_frame data available.")
            return
        #   convert the frame to ROS2 Image
        depth_frame_msg = self.ros_node.cv_bridge.cv2_to_imgmsg(depth_frame, encoding='bgr8')
        #   publish the frame
        self.ros_node.pub.frame.depth_image.publish(depth_frame_msg)
    
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

    def publish_objects(self, objects) -> None:
        """
        Publish the gaze data to the ROS2 topic.
        """
        # check if object data is available
        if not objects:
            print("No object data available.")
            return
        #   convert the objects to a custom ROS2 message
        obj_msg = Detections()
        obj_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        obj_msg.header.frame_id = 'object_detection'
        # iterate over object events
        for obj in objects:
            detection = Detection() 
            detection.prediction = obj['cls']
            detection.confidence = float(obj['conf'])
            detection.bbox.top_left.x = float(obj['xyxy'][0])
            detection.bbox.top_left.y = float(obj['xyxy'][1])
            detection.bbox.bottom_right.x = float(obj['xyxy'][2])
            detection.bbox.bottom_right.y = float(obj['xyxy'][3])
            # fit together with final message
            obj_msg.detections.append(detection)
        # TODO: add the object data to the message
        # obj_msg.detections = objects
        # print(objects)
        #   publish the message
        self.ros_node.pub.objects.publish(obj_msg)

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
        #   publisher for depth images
        self.pub.frame.depth_image = self.create_publisher(Image, self.node_name + '/frame/depth_image', 10)
        #   publisher for the camera info
        self.pub.frame.info = self.create_publisher(CameraInfo, self.node_name + '/frame/info', 10)
        #   publisher for the gaze data
        self.pub.gaze = self.create_publisher(GazeStamped, self.node_name + '/gaze', 10)
        #   publisher for the imu data
        self.pub.imu = self.create_publisher(Imu, self.node_name + '/imu', 10)
        #   publisher for the detected objects
        self.pub.objects = self.create_publisher(Detections, self.node_name + '/objects', 10)

class RosPublishers():
    def __init__(self): 
        self.frame      = RosCameraPublisher()
        self.gaze       = None
        self.imu        = None
        self.objects    = None

class RosCameraPublisher():
    def __init__(self):
        self.image      = None
        self.depth_image= None
        self.info       = None