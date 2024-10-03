# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    event_handler.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Paul Joseph <paul.joseph@pbl.ee.ethz.ch    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/10/03 11:01:55 by Paul Joseph       #+#    #+#              #
#    Updated: 2024/10/03 13:03:38 by Paul Joseph      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class EventHandler():

    #    ___       _ _   
    #   |_ _|_ __ (_) |_ 
    #    | || '_ \| | __|
    #    | || | | | | |_ 
    #   |___|_| |_|_|\__|
    def __init__(self):
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
        return frame.img                                                                          

    def get_gaze(self, events) -> np.array:
        """
        Return the gaze data.
        For information on the events data type contact pupillabs ...
        """
        gaze = events.get("gaze")
        if not gaze:
            return
        gaze = (
            gp for gp in gaze if gp["confidence"] >= self.g_pool.min_data_confidence
        )
        return gaze
 