#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2
from scipy.spatial import distance
import math
import numpy as np
from utils.function_utils import Utils


class Hand_position(Utils):
    """Hand position, the direction of the hand ,
    the face of the hand (palm or back)"""


    def __init__(self):
        """Constructor"""

        self.class_object_data_hand = {}
        self.timer = 0

    def getter_data_hand(self, data_hand):
        self.class_object_data_hand = data_hand

    def getter_timer(self, timer):
        self.timer = timer

    def raise_data(self):
        self.class_object_data_hand = {}


    @staticmethod
    def direction_of_angle(
        hand_angle, directionX, directionY, first_phax, last_phax):
        """The direction of the hand between the palm and the major."""

        direction_hand = ""

        # Only on Y axis (major direction's: top or bot).
        if 100 > hand_angle > 70:
            direction_hand = f"{directionY}"

        # In X & Y axis (major direction's: top right, top left, bot right ...).
        elif 70 > hand_angle > 35:
            direction_hand = f"{directionY} {directionX}"

        # Only one axis X or Y.
        elif 35 > hand_angle >= 0:

            x1, y1 = first_phax
            x2, y2 = last_phax
            direction_hand = f"plat {directionX}" if abs(x1 - x2) > abs(y1 - y2) else f"{directionY}"

        return direction_hand


    def hand_direction(self, hand_label):
        """Hand direction in function of the angulus beetween the major and the palm"""

        hand_landmarks_detected = self.class_object_data_hand["landmarks"][hand_label]

        if self.not_empty(hand_landmarks_detected):

            paumeX, paumeY = hand_landmarks_detected[0][0]
            majorX, majorY = hand_landmarks_detected[2][-1]

            sides = {"opposite": [(majorX, paumeY), (majorX, majorY)],
                    "adjacent": [(majorX, paumeY), (paumeX, paumeY)]}

            opposite_side, adjacent_side = [distance.euclidean(coordinates[0], coordinates[1])
                                            for _, coordinates in sides.items()]

            angle_major_palm = math.degrees(math.atan(opposite_side / adjacent_side))\
                               if (opposite_side is not 0 and adjacent_side is not 0) else 0

            direction_in_vertical = "top" if majorY < paumeY else "bot"
            direction_in_horizontal = "left" if majorX < paumeX else "right"

            self.class_object_data_hand["direction of the hand"][hand_label] = self.direction_of_angle(
                angle_major_palm, direction_in_horizontal, direction_in_vertical,
                (paumeX, paumeY), (majorX, majorY))



    @staticmethod
    def recuperate_points(hand_detected):
        """Recuperate the minimum coordinate in a finger of the phax on x or y.
        For example: auricular: [(1, 5), (2, 2), (3, 3), (4, 4)]
        min x coordiantes's 1 & min y coordinate's 2"""

        recuperate_min_pts = lambda hand_index, axis, range1, range2:\
            min([hand_detected[hand_index][index][axis] for index in range(range1, range2)]) 

        wrinkle = hand_detected[0][0]

        thumb_x_last_phax = recuperate_min_pts(0, 0, 1, 5)
        thumb_y_last_phax = recuperate_min_pts(0, 1, 1, 5)

        majors_x = recuperate_min_pts(2, 0, 0, 4)
        majors_y = recuperate_min_pts(2, 1, 0, 4)

        return wrinkle, (thumb_x_last_phax, thumb_y_last_phax), (majors_x, majors_y)
  

    @staticmethod
    def palm_or_back_plat(hand_label, thumb, wrinkle, major):
        """Definate hand label if the orienration of the hand is plat"""

        hand_face = "None"

        if hand_label is "right":
            # Thumb's bottom of the palm, major left of the palm
            hand_face = "back" if thumb[1] < wrinkle[1] and wrinkle[0] > major[0] else "Palm"

        elif hand_label is "left":
            # Thumb's bottom of the palm, major rigth of the palm
            hand_face = "back" if thumb[1] < wrinkle[1] and wrinkle[0] < major[0] else "Palm"

        return hand_face


    @staticmethod
    def palm_or_back_not_in_bot(hand_label, thumb, major):
        """Definate hand label if the orienration of the hand is bot"""

        hand_face = "None"

        if hand_label is "right":
            # Thumb's left of the palm
            hand_face = "Palm" if thumb[0] < major[0] else "back"

        elif hand_label is "left":
            # Thumb's right of the palm
            hand_face = "Palm" if thumb[0] > major[0] else "back"

        return hand_face


    @staticmethod
    def palm_or_back_bot(hand_label, thumb, wrinkle):
        """Definate hand label if the orienration of the hand is not bot"""

        hand_face = "None"

        if hand_label is "right":
            # Thumb's left of the palm
            hand_face = "back" if thumb[0] < wrinkle[0] else "Palm"

        elif hand_label is "left":
            # Thumb's right of the palm
            hand_face = "back" if thumb[0] > wrinkle[0] else "Palm"

        return hand_face


    def facing_hand(self, hand_label, frame):
        """Define the side of the hand in function of the localisation of the thumb the major and the palm coordinates.
        Hand can be palm or back face, & in function of the direction of the hand (mesured by the major - palm)

        for example:

                thumb index major annular auricular                                       
                              palm                                      

                    right hand & direction top    
                            Palm side
        """

        hand_detected = self.class_object_data_hand["landmarks"][hand_label]

        if self.not_empty(hand_detected):

            # Recuperate min (x & y) coordinates of phax of palm, thumb & major.
            wrinkle, thumb, major = self.recuperate_points(hand_detected)
            # Recuperate the direction of the hand.
            direction = self.class_object_data_hand["direction of the hand"][hand_label]

            # Define hand side.
            hand_face = "None"

            # Hand's plat.
            if "plat" in direction.lower():
                hand_face = self.palm_or_back_plat(hand_label, thumb, wrinkle, major)

            else:
                # Hand not bot.
                if "bot" not in direction.lower():
                    hand_face = self.palm_or_back_not_in_bot(hand_label, thumb, major)
                
                # Hand bot.
                elif "bot" in direction.lower():
                    hand_face = self.palm_or_back_bot(hand_label, thumb, wrinkle)


            self.class_object_data_hand["faceHand"][hand_label] = hand_face



class Hand_sign(Utils):
    """ """

    def __init__(self):
        """ """
        self.class_object_data_face = {}
        self.class_object_data_hand = {}

        self.timer = 0

        self.width_face = 14.9

        # Margin to add at the face boxe for know if the hand's in the
        # face area (in percent).
        self.detection_hand_in_face_area_threshold = 40

        # Speed of the hand (in a movement) can false a hand sign
        # detection. We make detection only if the speed is less (in cm/s):
        self.speed_threhsold_for_hand_sign = 50


    def getter_data_hand(self, data_hand):
        """Getter data hand"""
        self.class_object_data_hand = data_hand

    def getter_data_face(self, data_face):
        """Getter data face"""
        self.class_object_data_face = data_face

    def getter_timer(self, timer):
        """Getter timer in video"""
        self.timer = timer

    def raise_data(self):
        """Raise data"""
        self.class_object_data_face = {}
        self.class_object_data_hand = {}


    @staticmethod
    def get_length(mesure1, mesure2, ratio):
        """Recuperate distance round to minimum."""
        return math.floor(distance.euclidean(mesure1, mesure2) * ratio * 0.0265)

    @staticmethod
    def get_finger_from_hand(hand):
        """Recuperate all points in a list."""
        return [hand[0][1:]] + [hand[finger] for finger in range(1, 5)]

    def mesuration_distance_beetween_each_finger(self, hand, ratio):
        """Mesure distance beetween fingers in cm."""
        hand = self.get_finger_from_hand(hand)

        return [[self.get_length(phax1, phax2, ratio)
                for phax1, phax2 in zip(hand[finger], hand[finger + 1])]
                for finger in range(len(hand) - 1)]

    def mesuration_distance_beetween_thumb_finger(self, hand, ratio):
        """Mesure distance beetween thumb & fingers."""

        thumb, i, m, an, au = self.get_finger_from_hand(hand)
        hand = [i, m, an, au]

        return [[self.get_length(phax1, phax2, ratio)
                for phax1, phax2 in zip(thumb, hand[index])]
                for index in range(len(hand))]


    def length_of_the_finger(self, hand, ratio):
        """Recuperate length of fingers."""

        hand = self.get_finger_from_hand(hand)
        return [self.scaling_distance_round(finger[0], finger[-1], ratio)
                for finger in hand]


    @staticmethod
    def recuperate_phax_interest(index_finger, finger):
        """ Recuperate phaxs interest (1, 2, 4) thumb & (0, 1, 3) others fingers"""

        is_thumb_finger = index_finger is 0
        return finger[2 if is_thumb_finger else 0],\
               finger[3 if is_thumb_finger else 1],\
               finger[4 if is_thumb_finger else 3]


    @staticmethod
    def recuperate_distance(phaxs_coordinates, ratio):
        """ Recuperate euclidean distance of the triangle of the phaxs."""

        first_phax_coord, second_phax_coord, third_phax_coord = phaxs_coordinates

        coordinates = [
            (second_phax_coord, third_phax_coord),
            (third_phax_coord, first_phax_coord),
            (first_phax_coord, second_phax_coord)]

        return [distance.euclidean(coord_1, coord_2) * ratio * 0.0265
                for (coord_1, coord_2) in coordinates]


    def angle_of_the_fingers_from_first_and_last_phax(self, hand, ratio):
        """Recuperate angle of fingers."""
        angles_of_finger = []

        for index_finger, finger in enumerate(hand):

            coordinates_fingers = self.recuperate_phax_interest(index_finger, finger)

            a, b, c = self.recuperate_distance(coordinates_fingers, ratio)

            if (2 * c * a) > 0:
                angle_b = (c ** 2 + a ** 2 - b ** 2) / (2 * c * a)

                angle_b = -1 if angle_b < -1 else angle_b
                angle_b = 1 if angle_b > 1 else angle_b

                angle_b_degrees = round(math.degrees(math.acos(angle_b)))
                angles_of_finger += [angle_b_degrees]


        return angles_of_finger



    def angle_beetween_last_phax_finger_palm(self, hand):
        """Recuperate angle beetween fingers and palm."""
        angles_of_finger_palm = []

        paumeX, paumeY = hand[0][0]
        hand = self.get_finger_from_hand(hand)

        for finger in hand:

            last_phax_x, last_phax_y = finger[-1]

            sides = {
                "opposite": [(last_phax_x, paumeY), (last_phax_x, last_phax_y)],
                "adjacent": [(last_phax_x, paumeY), (paumeX, paumeY)]}

            opposite_side, adjacent_side = [
                distance.euclidean(coordinates[0], coordinates[1])
                for _, coordinates in sides.items()]

            if opposite_side != 0 and adjacent_side != 0:
                angles_of_finger_palm += [
                    round(math.degrees(math.atan(opposite_side / adjacent_side)))]

        return angles_of_finger_palm


    @staticmethod
    def recuperate_finger_extremities(hand):
        """Recuperate finger's extremities."""
        last_phax_of_the_hand = [hand[index][-1] for index in range(0, 5)]
        origin_of_the_hand = [hand[0][2], hand[1][1], hand[2][1], hand[3][1], hand[4][1]]

        return last_phax_of_the_hand, origin_of_the_hand


    def finger_direction_y(self, hand):
        """Recuperate direction of a fingers on vertical axis."""
        last_phax_of_the_hand, origin_of_the_hand =\
            self.recuperate_finger_extremities(hand)

        return ["top" if last_phax_y < first_phax_y else "bot"
                for (_, first_phax_y), (_, last_phax_y) in
                zip(origin_of_the_hand, last_phax_of_the_hand)]



    def finger_position(self, hand):
        """Recuperate finger direction on all axis."""
        last_phax_of_the_hand, origin_of_the_hand = self.recuperate_finger_extremities(hand)

        direction = []

        for (first_phax_x, first_phax_y), (last_phax_x, last_phax_y) in zip(
            origin_of_the_hand, last_phax_of_the_hand):

            diff_x = abs(first_phax_x - last_phax_x)
            diff_y = abs(first_phax_y - last_phax_y)

            divisor, divisible = sorted([diff_x, diff_y])
            ratio = divisible / (divisor if divisor is not 0 else 1)
            is_diagonal = ratio < 5

            y_axis = "top"  if first_phax_y - last_phax_y > 0 else "bot"
            x_axis = "left" if first_phax_x - last_phax_x < 0 else "right"

            direction += [
                f'{x_axis} {y_axis}' if is_diagonal
                else x_axis if (diff_x > diff_y) else y_axis]

        return direction



    @staticmethod
    def one_mesure_above_a_threshold(liste, finger, mesure):
        """Compare a phax with a mesure and verify it's superior"""
        return liste[finger] > mesure

    @staticmethod
    def one_mesure_less_a_threshold(liste, finger, mesure):
        """Compare a phax with a mesure and verify it's inferior"""
        return liste[finger] < mesure

    @staticmethod
    def mesure_less_a_threshold(liste, finger1, finger2, mesure_threshold):
        """Verify there are a False data in a range list (case inferior)."""
        return False if False in\
               [mesure < mesure_threshold for mesure in liste[finger1 : finger2]]\
                else True
    @staticmethod
    def mesure_above_a_threshold(liste, finger1, finger2, mesure_threshold):
        """Verify there are a False data in a range list (case superior)."""
        return False if False in\
               [mesure > mesure_threshold for mesure in liste[finger1 : finger2]]\
                else True

    @staticmethod
    def mesure_target_less_a_threshold(liste, mesure_threshold):
        """Verify there are a False data in a boolean list (case superior)."""
        return False if False in\
               [mesure < mesure_threshold for mesure in liste]\
               else True

    @staticmethod
    def condition_are_filled(sign_detected, *args):
        """Verify a sign's detected."""
        return sign_detected if False not in [i for i in args] else ""


    def in_communication_finger(self, dist_beetween_finger, angle_of_finger)  -> list():
        """Little finger up."""
        auricular_is_not_fold = self.one_mesure_above_a_threshold(angle_of_finger, 4, 130)
        others_fingers_are_fold = self.mesure_less_a_threshold(angle_of_finger, 1, -1, 100)

        return self.condition_are_filled(
            "communication", auricular_is_not_fold, others_fingers_are_fold)


    def confidence(self, angles_of_finger, distance_fingers_extremities) -> list():
        """Index up."""
        index_is_not_fold = self.one_mesure_above_a_threshold(angles_of_finger, 1, 120)
        others_fingers_are_fold = self.mesure_less_a_threshold(angles_of_finger, 2, 5, 100)

        return self.condition_are_filled(
            "confident", index_is_not_fold, others_fingers_are_fold)


    def is_approbation(self, distance_thumb_and_fingers, angle_of_finger) -> list():
        """Thumb up."""
        thumb_is_not_fold = self.one_mesure_above_a_threshold(angle_of_finger, 0, 120)
        others_fingers_are_fold = self.mesure_less_a_threshold(angle_of_finger, 1, -1, 100)

        distance_phaxs_fingers_thumb = [phax[-1] >= 4 and phax[-2] >= 4
                                        for phax in distance_thumb_and_fingers]

        distance_fingers_thumb_are_above_4_cm =\
            self.is_true_or_false_in_list(distance_phaxs_fingers_thumb)

        return self.condition_are_filled(
            "approbation", thumb_is_not_fold, others_fingers_are_fold,
            distance_fingers_thumb_are_above_4_cm)


    def approbation_palm_face(self, distance_beetween_fingers, distance_thumb_and_fingers,
                              angles_of_finger, is_palm_side):
        """Thumb up case 2."""
        fingers_are_glue = [phax[-1] <= 3 for phax in distance_beetween_fingers[1:]]
        fingers_are_glue = self.is_true_or_false_in_list(fingers_are_glue)

        thumb_finger_above_6_cm = [i[-1] > 6 for i in distance_thumb_and_fingers]
        thumb_finger_above_6_cm = self.is_true_or_false_in_list(thumb_finger_above_6_cm)

        thumb_deliate = self.one_mesure_above_a_threshold(angles_of_finger, 0, 120)
        others_fingers_pliate = self.mesure_less_a_threshold(angles_of_finger, 1, 5, 161)

        return self.condition_are_filled(
            "approbation 1", fingers_are_glue, thumb_finger_above_6_cm,
            thumb_deliate, others_fingers_pliate, is_palm_side)


    def fist_sign1(self, distance_beetween_fingers, angles_of_finger):
        """Fist sign."""
        all_finger_are_fold = self.mesure_less_a_threshold(angles_of_finger, 0, 5, 121)
        index_thumb_less_3cm = distance_beetween_fingers[0][-1] <= 3

        all_finger_except_thumb_are_less_3cm = [i[-1] <= 3 for i in distance_beetween_fingers[1:]]
        all_finger_except_thumb_are_less_3cm =\
            self.is_true_or_false_in_list(all_finger_except_thumb_are_less_3cm)

        return self.condition_are_filled(
            "fist 1", all_finger_are_fold, index_thumb_less_3cm,
            all_finger_except_thumb_are_less_3cm)


    def fist_sign2(self, distance_beetween_fingers, distance_thumb_and_fingers,
                   angles_of_finger):
        """Fist sign case 2."""
        fingers_are_glue = [i[-1] <= 2 for i in distance_beetween_fingers]
        fingers_are_glue = self.is_true_or_false_in_list(fingers_are_glue)

        fingers_except_thumb_are_less_120_degrees =\
                self.mesure_less_a_threshold(angles_of_finger, 1, 5, 121)

        return self.condition_are_filled(
            "fist 2", fingers_are_glue,
            fingers_except_thumb_are_less_120_degrees)


    def fist_sign3(self, distance_beetween_fingers, length_finger, angles_of_finger):
        """Fist sign case 3."""

        index_thumb_less_3cm = distance_beetween_fingers[0][-2] <= 3

        figners_phax_are_glue = [i[-2] <= 3 and i[-1] <= 2
                                 for i in distance_beetween_fingers[1:]]

        figners_phax_are_glue = self.is_true_or_false_in_list(
            figners_phax_are_glue)

        figners_length_are_less_6cm = self.mesure_less_a_threshold(
            length_finger, 1, 5, 6)

        fingers_except_thumb_are_less_120_degrees =\
                self.mesure_less_a_threshold(angles_of_finger, 1, 5, 121)

        return self.condition_are_filled(
            "fist 3", index_thumb_less_3cm, figners_phax_are_glue,
            figners_length_are_less_6cm, fingers_except_thumb_are_less_120_degrees)



    def index_separate_other(self, angles_of_finger, finger_all_dir, angle_fing_palm):
        """Index up."""
  
        index_is_deliate = self.one_mesure_above_a_threshold(angles_of_finger, 1, 150)
        index_is_top = "top" in finger_all_dir[1]
        other_finger_are_bot = ["bot" in i for i in finger_all_dir[3:]]
        other_finger_are_bot = False if other_finger_are_bot.count(True) < 2 else True

        return self.condition_are_filled(
            "confident 2", index_is_deliate, index_is_top, other_finger_are_bot)


    def stop_sign(self, angles_of_finger, is_palm_side, angle_finger_palm, direction):
        """Stop sign (palm face)."""
 
        finger_are_above_160_degrees =\
            self.mesure_above_a_threshold(angles_of_finger, 1, 5, 160)

        fingers_are_straight =\
            self.mesure_above_a_threshold(angle_finger_palm, 1, 4, 65)

        fingers_are_top = [i is "top" for i in direction]
        fingers_are_top = self.is_true_or_false_in_list(fingers_are_top)

        return self.condition_are_filled(
            "stop", finger_are_above_160_degrees,
            fingers_are_straight, fingers_are_top, is_palm_side)


    def finger_isnt_pliate(self, angles_of_finger, distance_beetween_fingers, direction):
        """Peace, gun & attraction."""

        detection = ""

        if len(angles_of_finger) > 4:
            thumb, index, major, annu, auri = angles_of_finger

            index_deliate = self.one_mesure_above_a_threshold(angles_of_finger, 1, 140)
            major_deliate = self.one_mesure_above_a_threshold(angles_of_finger, 2, 140)

            index_major_deliate = index_deliate and major_deliate

            index_maj_glue = distance_beetween_fingers[1][-1] <= 2 and\
                             distance_beetween_fingers[1][-2] <= 2

            are_fold = self.one_mesure_less_a_threshold(angles_of_finger, 3, 121) and\
                       self.one_mesure_less_a_threshold(angles_of_finger, 4, 121)

            are_bot = [i == "bot" for i in direction]
            are_bot = self.is_true_or_false_in_list(are_bot)

            if index_major_deliate and are_fold and index_maj_glue:
                detection = "gun"

            elif index_major_deliate and are_fold and not index_maj_glue and not are_bot:
                detection = "peace"

            elif index_major_deliate and are_fold and not index_maj_glue and are_bot:
                detection = "attraction"

        return detection


    def h_sign(self, angles_of_finger, direction):
        """Index, auricular up."""

        detection = ""

        if len(angles_of_finger) == 5:

            t, i, m, an, au = angles_of_finger

            are_not_fold = self.mesure_target_less_a_threshold([t, i, au], 120)
            are_fold = self.mesure_target_less_a_threshold([m, au], 100)

            t, i, m, an, au = direction

            there_are_top = [i == "top" for i in [t, i, au]]
            there_are_top = self.is_true_or_false_in_list(there_are_top)

            if are_not_fold and are_fold and there_are_top:
                detection = "h"

        return detection


    def drawing_skeletton(self, frame, hand):
        """ """

        first_phaxs = [hand[0][0], hand[0][1]] + [hand[finger][0] for finger in range(1, 5)]

        [cv2.line(frame, first_phaxs[i], first_phaxs[i + 1], (0, 0, 255), 2)
         for i in range(len(first_phaxs) - 1)]

        cv2.line(frame, first_phaxs[0], first_phaxs[-1], (0, 0, 255), 2)

        color = [(255, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

        for index_finger, finger in enumerate(hand):

            first_phax = 1 if index_finger is 0 else 0

            [cv2.line(frame, finger[i], finger[i + 1], color[index_finger], 2)
             for i in range(first_phax, len(finger) - 1)]

            if index_finger is 0:
                cv2.circle(frame, finger[0], 2, (0, 255, 0), -1)


    def a_sign_is_detected(self, frame, boxe_hand, is_palm_side, data_features, label):
        """Verify if a sign complete conditions."""

        distance_beetween_each_fingers, distance_thumb_and_others_fingers,\
        fingers_length_beetween_extremities, angles_of_the_finger,\
        angle_of_the_finger_and_the_palm, direction_on_y_of_the_finger,\
        finger_direction_on_y_and_x = data_features

        sign_detected = [
            self.in_communication_finger(distance_beetween_each_fingers, angles_of_the_finger),
            self.is_approbation(distance_thumb_and_others_fingers, angles_of_the_finger),
            self.fist_sign1(distance_beetween_each_fingers, angles_of_the_finger),

            self.fist_sign2(distance_beetween_each_fingers, distance_thumb_and_others_fingers, angles_of_the_finger),
            self.approbation_palm_face(distance_beetween_each_fingers, distance_thumb_and_others_fingers,angles_of_the_finger, is_palm_side),
            self.confidence(angles_of_the_finger, fingers_length_beetween_extremities),
 
            self.finger_isnt_pliate(angles_of_the_finger, distance_beetween_each_fingers, direction_on_y_of_the_finger),
            self.index_separate_other(angles_of_the_finger, finger_direction_on_y_and_x, angle_of_the_finger_and_the_palm),
            self.stop_sign(angles_of_the_finger, is_palm_side, angle_of_the_finger_and_the_palm, direction_on_y_of_the_finger),

            self.h_sign(angles_of_the_finger, direction_on_y_of_the_finger),
            self.fist_sign3(distance_beetween_each_fingers, fingers_length_beetween_extremities, angles_of_the_finger)
        ]


        x, y, w, h = boxe_hand

        font =  cv2.FONT_HERSHEY_SIMPLEX
        detection = list(set([i.split()[0] for i in sign_detected if i is not ""]))

        [cv2.putText(frame, i, (x, y + nb * 15),font, 0.4, (255, 255, 255)) for nb, i in enumerate(detection)]

        for sign_detected in detection:
            self.class_object_data_hand["sign"][label] += [(sign_detected, self.timer)]



    def signs_of_the_hand(self, copy, frame, label):
        """Recuperate all mesures."""

        boxe_face = self.class_object_data_face["face_box"]
        hand = self.class_object_data_hand["landmarks"][label]
        # Speed of the hand can make false detection.
        # if the speed less 100 cm/s we can make a detection.
        hand_speed = self.class_object_data_hand["speed"][label]

        if self.not_empty(hand) and hand_speed is not None and hand_speed[1] < self.speed_threhsold_for_hand_sign:

            hand_speed = hand_speed[1]

            boxe_hand = self.class_object_data_hand["boxe"][label]
            palm = self.class_object_data_hand["faceHand"][label]
            is_palm_side = True if palm is "Palm" else False

            self.drawing_skeletton(copy, hand)

            ratio = self.get_ratio_distance(self.width_face, boxe_face[2])

            distance_beetween_each_fingers = self.mesuration_distance_beetween_each_finger(hand, ratio)
            distance_thumb_and_others_fingers = self.mesuration_distance_beetween_thumb_finger(hand, ratio)
            fingers_length_beetween_extremities = self.length_of_the_finger(hand, ratio)

            angles_of_the_finger = self.angle_of_the_fingers_from_first_and_last_phax(hand, ratio)
            angle_of_the_finger_and_the_palm = self.angle_beetween_last_phax_finger_palm(hand)
            direction_on_y_of_the_finger = self.finger_direction_y(hand)
            finger_direction_on_y_and_x = self.finger_position(hand)

            if len(angles_of_the_finger) is 5:

                data_features = [distance_beetween_each_fingers, distance_thumb_and_others_fingers,
                                fingers_length_beetween_extremities, angles_of_the_finger,
                                angle_of_the_finger_and_the_palm, direction_on_y_of_the_finger, finger_direction_on_y_and_x]

                self.a_sign_is_detected(copy, boxe_hand, is_palm_side, data_features, label)



    # TO DO hand on face.
    def recuperate_landmarks_face(self, face_landmarks):
        """Recuperate the coordinates of the area interest from DLIB."""

        # Points transforme to coordinate of the face on the picture.
        dico_part = {
            "nose_right": [(27, 27), (47, 47), (47, 35), (33, 33)],
            "nose_left": [(27, 27), (40, 40), (40, 31), (33, 33)],
            "mouse": [(48, 31), (54, 35), (54, 57), (48, 57)],
            "joue_right": [(47, 47), (15, 15), (13, 13), (47, 13)],
            "joue_left": [(40, 40), (1, 1), (3, 3), (40, 3)],
            "menton_right": [(54, 56), (12, 12), (9, 9), (55, 55)],
            "menton_left": [(48, 59), (4, 4), (7, 7), (59, 59)],
            "menton": [(59, 57), (55, 57), (10, 8), (6, 8)],
        }

        points_to_landmarks = lambda landmarks, points:\
                              [(landmarks[x][0], landmarks[y][1]) for (x, y) in points]

        for part, points_interest in dico_part.items():
            dico_part[part] = points_to_landmarks(face_landmarks, points_interest)

        return dico_part


    def recuperate_area_from_head_box(self):
        """From the face, we try to recuperate area out of the face."""

        

        x, y, w, h = self.class_object_data_face["face_box"]
        percent = lambda per, length: (per * length) // 100

        dico_part_area = {

            "tempe_right": [(x+w, y-percent(50, h)),
                            (x+w+percent(25, w), y-percent(50, h)),
                            (x+w+percent(25, w), y),(x+w, y)],

            "tempe_left": [(x, y-percent(50, h)),(x-percent(25, w), y-percent(50, h)),
                           (x-percent(25, w), y), (x, y)],

            "forehead_right": [(x+w-percent(25, w), y-percent(50, h)),
                               (x+w, y-percent(50, h)),
                               (x+w, y), (x+w-percent(25, w), y)],

            "forehead_left": [(x, y-percent(50, h)), (x+percent(25, w), y-percent(50, h)),
                              (x+percent(25, w), y), (x, y)],

            "forehead": [(x+percent(25, w), y-percent(50, h)),
                         (x+w-percent(25, w), y-percent(50, h)),
                        (x+w-percent(25, w), y),
                         (x+percent(25, w), y)],

            "on_head": [(x+percent(25, w), y-percent(120, h)),
                        (x+w-percent(25, w), y-percent(120, h)),
                        (x+w-percent(25, w), y-percent(80, h)),
                        (x+percent(25, w), y-percent(80, h))],
        }

        return dico_part_area


    def adding_marge(self, dico_part_face):
        """Define area from dlib points. Add some margin in function of the head dimensions"""

        x, y, w, h = self.class_object_data_face["face_box"]
        percent = lambda per, length: (per * length) // 100

        joue_right = dico_part_face["joue_right"]
        joue_right[1] = (joue_right[1][0]+percent(20, w), joue_right[1][1])
        joue_right[2] = (joue_right[2][0]+percent(20, w), joue_right[2][1])
        joue_right[3] = (joue_right[3][0], joue_right[3][1]+percent(8, h))
        dico_part_face["joue_right"] = joue_right

        joue_left = dico_part_face["joue_left"]
        joue_left[1] = (joue_left[1][0]-percent(20, w), joue_left[1][1])
        joue_left[2] = (joue_left[2][0]-percent(20, w), joue_left[2][1])
        joue_left[3] = (joue_left[3][0], joue_left[3][1]+percent(8, h))
        dico_part_face["joue_left"] = joue_left

        menton = dico_part_face["menton"]
        menton[2] = (menton[2][0]+percent(10, h), menton[2][1] + percent(10, h))
        menton[3] = (menton[3][0]-percent(10, h), menton[3][1] + percent(10, h))
        dico_part_face["menton"] = menton

        menton_right = dico_part_face["menton_right"]
        menton_right[1] = (menton_right[1][0]+percent(15, w), menton_right[1][1])
        menton_right[2] = (menton_right[2][0]+percent(15, w), menton_right[2][1])
        dico_part_face["menton_right"] = menton_right

        menton_left = dico_part_face["menton_left"]
        menton_left[1] = (menton_left[1][0]-percent(15, w), menton_left[1][1])
        menton_left[2] = (menton_left[2][0]-percent(15, w), menton_left[2][1])
        dico_part_face["menton_left"] = menton_left




    def define_if_hands_are_in_face_area(self, label, frame):
        """Verify if hands points are  in area_face"""

        frame_height, frame_width = frame.shape[:2]

        faceX, faceY, faceW, faceH = self.class_object_data_face["face_box"]
        hand = self.class_object_data_hand["landmarks"][label]

        fingers = self.recuperate_all_phax_in_the_hand(hand)

        margin = self.percent_of(faceW, self.detection_hand_in_face_area_threshold)

        # Verify margin doesn't exceeds the frame.
        x = faceX - margin if faceX - margin >= 0 else 0
        y = faceY - margin if faceY - margin >= 0 else 0
        w = faceX + faceW + margin if faceX + faceW + margin <= frame_width else frame_width
        h = faceY + faceH + margin if faceY + faceH + margin <= frame_height else frame_height

        # Verify x & y are in the face boxe.
        is_in_range = lambda ptsRange1, ptsRange2, hand, index: [ptsRange1 <= finger[index] <= ptsRange2 for finger in hand]
        phax_in_face_x = is_in_range(x, w, fingers, 0)
        phax_in_face_y = is_in_range(y, h, fingers, 1)

        hand_in = [phax_in_face_y[n] and phax_in_face_x[n] for n in range(len(phax_in_face_x))]
        hand_in = self.is_true_in_liste_else_false(hand_in)

        return hand_in, (x, y, w, h)


    def update_data_face_area_hand_in_area_face(self, dico_part_face, label, frame):
        """Update if hand touchs an area & if hand are in face area (hand points in a percent of the face);
        it can avoid some detection (face detection) or create false detection (beetween wrinkle).
        Update in database face area points. 
        For example, in the next frame if face isn't detected but hand were in the area face,
        don't lunch the recognition facial and recuperate the last area of the face and the current hands
        points (for the touching face part)."""

        # Verify phaxs of the hand are in the face area.
        hand_in_face_area, (x, y, w, h) = self.define_if_hands_are_in_face_area(label, frame)

        # We have two hands so the last hand can false the detection (Was True pass to false).
        # Reinitialise it with the left label (Right label always first).
        hand_and_area = self.class_object_data_face["hand_is_in_head_zone"]

        if label is "right" or (label is "left" and not hand_and_area):
            self.class_object_data_face["hand_is_in_head_zone"] = hand_in_face_area

        # Display on the copy for knows if hand are in area face.
        #draw = (0, 0, 255) if hand_and_area or hand_in_face_area else (255, 0, 0)
        #cv2.rectangle(frame, (x, y), (w, h), draw, 2)

        # Save area face in the database for the next frame.
        for area_name, coordinates in dico_part_face.items():
            self.class_object_data_face["face_area"][area_name] = np.array(coordinates)



    def fingers_in_area_face(self, frame, virgin, label):
        """Recuperate face landmarks. Recuperate hands landmarks. Recuperate the mean of each last point of the hand
        and the last point of the finger.
        Verify if the last two points are on the area face."""

        hand_landmarks = self.class_object_data_hand["landmarks"][label]
        face_landmarks = self.class_object_data_face["face_landmarks"]
        face_area_to_touch = self.class_object_data_face["face_area"]

        there_is_landmarks = len(hand_landmarks) > 0
        if there_is_landmarks:

            # Recuperate landmarks and areas.
            dico_part_face = self.recuperate_landmarks_face(face_landmarks)
            dico_part_area = self.recuperate_area_from_head_box()

            # Merge the two dictionnaries.
            for extra_face, coordinates in dico_part_area.items():
                dico_part_face[extra_face] = coordinates

            # Add some marge to intra face areas.
            self.adding_marge(dico_part_face)

            # Update.
            self.update_data_face_area_hand_in_area_face(dico_part_face, label, frame)

            # Draw.
            [cv2.drawContours(frame, [v], -1, (0, 255, 0), 2)
             for k, v in face_area_to_touch.items() if len(v) > 0]

            is_touching = []

            is_hand_is_palm = self.class_object_data_hand["faceHand"][label] is "Palm"

            # Run area face
            for name_part_face, area in face_area_to_touch.items():

                there_is_area = len(area) > 0
                if there_is_area:

                    for index_finger, finger in enumerate(hand_landmarks[:-1]):
                        phaxs_in_area = [cv2.pointPolygonTest(area, phaxs, True) > 0
                                         for phaxs in finger]

                        if True in phaxs_in_area and not is_hand_is_palm:
                            cv2.drawContours(frame, [area], -1, (0, 255, 255), -1)
                            is_touching += [name_part_face]
  
            self.class_object_data_hand["touchingFace"][label] = [(i, self.timer) for i in is_touching]

