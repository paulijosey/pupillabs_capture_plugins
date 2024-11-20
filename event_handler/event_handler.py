# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    event_handler.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Paul Joseph <paul.joseph@pbl.ee.ethz.ch    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/10/03 11:01:55 by Paul Joseph       #+#    #+#              #
#    Updated: 2024/11/19 10:46:33 by Paul Joseph      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

# pulillabs imports (in sharted_modules)
from methods import denormalize

class EventHandler():

    #    ___       _ _   
    #   |_ _|_ __ (_) |_ 
    #    | || '_ \| | __|
    #    | || | | | | |_ 
    #   |___|_| |_|_|\__|
    def __init__(self):
        self.frame_size = None # needed to denormalize gaze position
        pass

    #    _____                 _     _   _                 _ _               
    #   | ____|_   _____ _ __ | |_  | | | | __ _ _ __   __| | | ___ _ __ ___ 
    #   |  _| \ \ / / _ \ '_ \| __| | |_| |/ _` | '_ \ / _` | |/ _ \ '__/ __|
    #   | |___ \ V /  __/ | | | |_  |  _  | (_| | | | | (_| | |  __/ |  \__ \
    #   |_____| \_/ \___|_| |_|\__| |_| |_|\__,_|_| |_|\__,_|_|\___|_|  |___/
    def get_frame(self, events) -> np.array:
        """
        Return the frame from the scene camera. 
        For information on the events data type contact pupillabs ...
        """
        frame = events.get("frame")
        if not frame:
            return
        self.frame_size = frame.img.shape[:-1][::-1]
        return frame.img                                                                          

    def get_depth_frame(self, events) -> np.array:
        """
        Return the frame from the depth camera.
        Only applicable if a depth camera is connected, enabled and
        sends data on the "depth_frame" event channel.
        For information on the events data type contact pupillabs ...
        """
        depth_frame = events.get("depth_frame")
        if not depth_frame:
            return
        self.depth_frame_size = depth_frame.img.shape[:-1][::-1]
        return depth_frame.img                                                                          

    def get_highest_conf_gaze(self, events) -> np.array:
        """
        Return the gaze data with highest confidence.
        For information on the events data type contact pupillabs ...
        """
        gaze = events.get("gaze")

        # check abort conditions
        if not gaze:
            return

        if self.frame_size is None:
            print("Frame size not set")
            return

        # Gaze data is a list of possible gaze positions.
        # We are currently only interested in the gaze with the highest confidence.

        # find highest confidence
        gaze = sorted(gaze, key=lambda x: x["confidence"], reverse=True)

        # get gaze with highest confidence
        gaze = gaze[0]

        # add denormalized gaze position
        gaze['denorm_pos'] = denormalize(gaze['norm_pos'], self.frame_size, flip_y=True)
        gaze['frame_size'] = self.frame_size

        return gaze

    def get_imu(self, events) -> np.array:
        """
        Return the data from the IMU. 
        For information on the events data type contact pupillabs ...
        TODO: figure out how to get IMU data ... officially not supported
                by PupilLabs lads.
        """
        imu = events.get("imu")
        if not imu:
            print("No imu data available.")
            return
        print(imu)
        return imu
    
    def get_objects(self, events) -> np.array:
        """
        Return the detected objects in the scene.
        For information on the events data type contact pupillabs ...
        """
        objects = events.get("objects")
        if not objects:
            print("No objects data available.")
            return
        return objects

    #    ____  _        _               ___        __                                                                          
    #   / ___|| |_ __ _| |_ _   _ ___  |_ _|_ __  / _| ___                                                                     
    #   \___ \| __/ _` | __| | | / __|  | || '_ \| |_ / _ \                                                                    
    #    ___) | || (_| | |_| |_| \__ \  | || | | |  _| (_) |                                                                   
    #   |____/ \__\__,_|\__|\__,_|___/ |___|_| |_|_|  \___/                                                                    
    def list_events(self, events) -> None:
        """
        List all events in the events data type.
        For information on the events data type contact pupillabs ...
        """
        print("---------- Events: ----------")
        for key, value in events.items():
            print(key)
        print("----------------------------")
        return
                                                                                                                             
 
