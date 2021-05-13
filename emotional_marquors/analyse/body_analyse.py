#!/usr/bin/python3
# -*- coding:utf-8 -*-

from utils.function_utils import Utils

class Body_analyse(Utils):
    """Movement of body."""

    def __init__(self):
        """Constructor"""
        self.class_object_data_analyse = {}
        self.class_object_data_body = {}
        self.timer = 0

        # Number of detections for save sign.
        self.sign_detection = 3

        # Threshold beetween last detection & current time in video
        # for raise the temporary list (in seconds).
        self.threshold_timer = 0.1



    def getter_data_body(self, data_body):
        """Get body data"""
        self.class_object_data_body = data_body

    def getter_data_analyse(self, data_analyse):
        """Get analysis data"""
        self.class_object_data_analyse = data_analyse

    def getter_timer(self, timer):
        """Get timer in the video"""
        self.timer = timer

    def raise_data(self):
        """Raise data"""
        self.class_object_data_analyse = {}
        self.class_object_data_body = {}


    def arms_sign(self):
        """Save a body sign if the sign's detected a certain number of time & raise the 
        temporary list if the last detection's less a threshold time."""

        movement_of_body = self.class_object_data_body["sign"]

        if self.not_empty(movement_of_body):
            
            # Recuperate only moves.
            signs_body = [move for (move, index_time) in movement_of_body]

            # Count the number of the moves.
            counter_sign = sorted([(signs_body.count(move), move) for move in list(set(signs_body))])

            # Recuperate the max number of moves.
            max_counter_sign, sign = counter_sign[-1]

            # If movement is above the treshold save the sign.
            if max_counter_sign >= self.sign_detection:
                self.class_object_data_analyse["body_sign"] += [(sign, self.timer)]

            # Raise the list if the last detection's less 0.1 second.
            _, last_sign_time = movement_of_body[-1]      
            if self.timer - last_sign_time > self.threshold_timer:
                self.class_object_data_body["sign"] = []
