"""Track the hand movement"""


import cv2
from scipy.spatial import distance
from utils.function_utils import Utils
import numpy as np


class Hand_movements(Utils):
    """Map color in function of the hands localisations"""

    def __init__(self):
        """ """
 
        self.FACE_LENGTH = 19.3
        self.TO_PIX = 0.0265
        
        self.class_object_data_face = {}
        self.class_object_data_hand = {}

        self.timer = 0

    def getter_data_hand(self, data_hand):
        """Getter hand data"""
        self.class_object_data_hand = data_hand

    def getter_data_face(self, data_face):
        """Getter face data"""
        self.class_object_data_face = data_face

    def getter_timer(self, timer):
        """Getter time in video"""
        self.timer = timer

    def raise_data(self):
        """Raise data"""
        self.class_object_data_face = {}
        self.class_object_data_hand = {}


    def hand_dist_data(self, hand_label):
        """Recuperate the face for get the ratio of distance."""

        face_boxe = self.class_object_data_face["face_box"]
        landmarks = self.class_object_data_hand["landmarks"][hand_label]
        last_palm = self.class_object_data_hand["last_palm"][hand_label]
        return face_boxe, landmarks, last_palm

    def hand_speed_dist(self, hand_label, frame):
        """Recuperate speed, movement of the hand movement"""

        # Recuperate the face for get the ratio of distance.
        face_boxe, landmarks, last_palm = self.hand_dist_data(hand_label)

        first_detection = self.class_object_data_hand["speed"][hand_label] is None
        hand_is_detected = self.not_empty(landmarks)

        #First detection.
        if hand_is_detected and first_detection:
            self.class_object_data_hand["speed"][hand_label] = (self.timer, 0)

        elif hand_is_detected and not first_detection and last_palm is not None:

            # Recuperate the palm of the hand.
            palm_localisation = landmarks[0][0]
 
            # Recuperate the last detection (timer, and speed) of the hand.
            last_move_timer, _ = self.class_object_data_hand["speed"][hand_label]

            cv2.arrowedLine(frame, last_palm, palm_localisation, (255, 255, 255), 2)

            # Get the distance from the last & the current detections.
            ratio = self.get_ratio_distance(14.9, face_boxe[2])
            dist = self.scaling_distance_round(palm_localisation, last_palm, ratio)

            # Get the difference beetween the two times.
            timer_during = self.timer - last_move_timer
            speed_move = abs(dist // timer_during)
            self.class_object_data_hand["speed"][hand_label] = (self.timer, speed_move)

            self.class_object_data_hand["distance"][hand_label] = dist

        # Update last palm with the new palm.
        if hand_is_detected:
            self.class_object_data_hand["last_palm"][hand_label] = landmarks[0][0]



    def define_hand_localisation_mean_y(self, label, frame):
        """Recuperate (x, y) coordinates & timer of the rectangle of the hand detected
        (we don't take in count the thumb)."""

        # Recuperate the hand.
        hands = self.class_object_data_hand["landmarks"][label]
        # Recuperate point of the hand without the thumb.
        hand_boxe = self.class_object_data_hand["boxe"][label]

        if self.not_empty(hands) and hand_boxe is not None:

            x, y, w, h = hand_boxe
            self.class_object_data_hand["hand_localisation"][label] += [(x, y, self.timer)]
