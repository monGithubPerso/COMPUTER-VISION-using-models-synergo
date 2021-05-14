#!/usr/bin/python3
# -*- coding:utf-8 -*-


import numpy as np
import cv2
from utils.function_utils import Utils


class Hand_analyse(Utils):
    """ """

    def __init__(self):
        """ """

        self.part_face_score = {"menton_right": 1, "menton_left": 1, "menton": 2,
                                "mouse": 5, "joue_right": 3, "joue_left": 3,
                                "nose_right": 4, "nose_left": 4, "tempe_right": 2,
                                "tempe_left": 2, "forehead_right": 3, "forehead_left": 3,
                                "forehead": 4, "on_head": 1,
            }

        self.class_object_data_analyse = {}
        self.class_object_data_hand = {}
        self.timer = 0

        # hand_localisation_analyse
        # If movement's are less 90 % of the average because bottom in video is top in real.
        self.movement_hight_threshold = 80


        # Remove hand sign after a certain delay (in frame) of the last detection.
        self.remove_hand_sign_threshold = 0.04

        # Right or left hand.
        self.hand_label = ["right", "left"]

        # Grouping timer of a gesture if gesture detection if less of 0.4.
        # For example: [(1.1, 1.2, 1.3, 5, 5.1, 5.6)]
        # Regruoping by range of 0.4:
        # [(1.1, 1.3), (5, 5.1) (5.6)] be cause 5 - 1.3 > 0.4 seconds.
        self.treshold_group_hand_localisation = 0.4

        # Treshold for the speed hand above the mean of hands speeds.
        self.threshold_hand_speed = 150


    def getter_data_hand(self, data_hand):
        """Get hand data"""
        self.class_object_data_hand = data_hand

    def getter_data_analyse(self, data_analyse):
        """Get analyse data"""
        self.class_object_data_analyse = data_analyse

    def getter_timer(self, timer):
        """Get timer in the video"""
        self.timer = timer

    def raise_data(self):
        """Raise data"""
        self.class_object_data_analyse = {}
        self.class_object_data_hand = {}


    def recuperate_distance(self, label_data):
        """Recuperate arm distance movements beetween last & current frame.."""
        # What's process is highter in the video.
        dico_data = {"right": [], "left": []}

        for label in ["right", "left"]:
            movement = self.class_object_data_analyse["arm_movement"][label]
            [dico_data[label].append(i) for i in movement["distance"]]

        return dico_data


    def analyse_process_use_hand(self):
        """Anlyse the the distance of hand used."""

        # Dominant hand uses.
        dico_analyse_process_more = self.recuperate_distance("distance")

        right, left = [sum(dico_analyse_process_more[label]) for label in ["right", "left"]]
        self.class_object_data_analyse["process_hand"] = "analyse" if right > left else "empathy"

        # Hand use in the video (timer). For example: hand right's use beetween (1 second to 2 seconds).
        dico_process_in_video = self.recuperate_distance("timer")

        for hand_label, data in dico_process_in_video.items():
            data = self.utils_groupe_timer_by_range(data, threshold_time=0.3)
            self.class_object_data_analyse["process_hand_timer"][hand_label] = data


    def hand_speed_analyse(self, label):
        """Recuperate speed & distance of the hands"""

        hand_speed = self.class_object_data_hand["speed"][label]
        hand_distance = self.class_object_data_hand["distance"][label]

        if hand_speed is not None and hand_distance is not None:

            movement = self.class_object_data_analyse["arm_movement"][label]
            movement["timer"] += [self.timer]
            movement["speed"] += [hand_speed[1]]
            movement["distance"] += [hand_distance]


    def verify_movement_were_fast(self, detection):
        """Verification if the hand movement was
        above the mean of the hand movements and less a certain threshold."""

        # arm_movement : "right": {"speed": [], "distance": [], "direction": [], "timer":[]}
        gesture = self.class_object_data_analyse["arm_movement"]

        for label in ["right", "left"]:
            speed_save = [i for i in gesture[label]["speed"]]
            timer_save = [i for i in gesture[label]["timer"]]

        mean_speed = np.mean(np.array(speed_save))
        threshold = self.percent_of(mean_speed, self.threshold_hand_speed)

        [detection.remove((begin, end)) for (begin, end) in detection
        for speed, timer in zip(speed_save, timer_save)
        if begin <= timer <= end and speed < threshold and (begin, end) in detection]
 
        return detection


    def drawing_line_threshold(self, draw_frame, y_mean):
        """For help in works, draw lines of condidence"""

        height, width = draw_frame.shape[:2]
        y_mean = int(y_mean)

        cv2.line(draw_frame, (0, y_mean), (width, y_mean), (255, 255, 255), 2)

        y_threshold = int(self.percent_of_not_round(y_mean, self.movement_hight_threshold))
        cv2.line(draw_frame, (0, y_threshold), (640, y_threshold), (0, 0, 255), 2)


    def hand_localisation_analyse(self, draw_frame):
        """Recuperate the average location of the hands. 
        Revovery the minimum value because Top & Bottom are inverse (Bot of the picture is the top for us).
        If the localisation's lower of the minimum value, could significate a confidence."""

        hand_liste = self.class_object_data_hand["hand_localisation"]

        # Recuperate the two hands points.
        two_hands_points = [hand_liste[label] for label in self.hand_label]
        join_list = [pt for pts in two_hands_points for pt in pts]

        # Recuperate the y coordinates.
        y_list = [y for (x, y, timer) in join_list]
        # Recuperate the average.
        y_mean = np.mean(np.array(y_list))

        if self.not_empty(y_list):

            # Verify if the min value is lower of the mean.
            maximums_highter_mean = min(y_list) < self.percent_of(y_mean, self.movement_hight_threshold)

            self.drawing_line_threshold(draw_frame, y_mean)

            if maximums_highter_mean:
                # Recuperate all movements less of the mean.
                liste = sorted([timer for (x, y, timer) in join_list if y < self.percent_of(y_mean, self.movement_hight_threshold)])

                # We recuperate timer in video if hand are less of the mean.
                # For example we get time in video of hand less of the mean: [1, 1.1, 1.2, 1.3, 10, 10.1].
                # Regroupe timer by range of 0.3 sec intervals.
                # [(1, 1.3), (10, 10.1)] because beetween 1.3 & 10 there is move 0.3.
                liste = self.utils_groupe_timer_by_range(liste, self.treshold_group_hand_localisation)

                # If an interval of move is egal to 0 or highter of 2 secondes remove it.
                liste = [(begin, end) for (begin ,end) in liste if end - begin < 1]
                # 
                liste = self.verify_movement_were_fast(liste)
 
                # Update hand marquors (the average can changes).
                self.class_object_data_analyse["marquors"]["marquor_hand"] = liste


    def decide_area_touch(self, areas_touching):
        """to do"""

        area_touch_score = [(self.part_face_score[part_name], part_name)
                            for (part_name, _) in areas_touching
                            if part_name in self.part_face_score]

        touched = sorted(area_touch_score)[-1][-1] if self.not_empty(area_touch_score) else None

        return touched


    def recuperate_sign_of_the_hand(self, label_hand):
        """to do"""

        sign_historic = self.class_object_data_hand["signHistoric"][label_hand]

        there_is_sign_in_couse = None

        if self.not_empty(sign_historic):
            last_sign, last_detection = sign_historic[-1]
            there_is_sign_in_couse = last_sign if self.timer is last_detection else None

        return there_is_sign_in_couse


    def recuperate_sign_in_database(self, area_choice, sign_in_couse, hand_label):
        """to do"""

        global TOUCH_FACE

        signification_part_touch = None

        there_is_sign = sign_in_couse is not None
        area_is_in_data = area_choice in TOUCH_FACE

        is_sign_or_touch = "sign" if there_is_sign else "touch"

        #print(area_choice, is_sign_or_touch, sign_in_couse, hand_label)


        if is_sign_or_touch is "sign":
            sign_in_couse = sign_in_couse.split(".")[1]\
                            if sign_in_couse is not None else ""

        area_has_only_touch = TOUCH_FACE[area_choice]["only_touch"] is True
        is_sign_or_touch = is_sign_or_touch if not area_has_only_touch else "touch"

        if area_is_in_data:

            if is_sign_or_touch is "sign":

                if sign_in_couse in TOUCH_FACE[area_choice][is_sign_or_touch][hand_label]:
                    signification_part_touch =\
                        TOUCH_FACE[area_choice][is_sign_or_touch][hand_label][sign_in_couse]

                else:
                    signification_part_touch =\
                        TOUCH_FACE[area_choice]["touch"][hand_label]

            elif is_sign_or_touch is "touch":
                signification_part_touch =\
                        TOUCH_FACE[area_choice][is_sign_or_touch][hand_label]

        return signification_part_touch



    def raising_work_list_after_a_delay(self, timer, hand_touching, hand_label):
        """Raise temporaty list after 0.1 seconds of the last detection."""

        _, last_timer = hand_touching[-1]

        delay_touch_raise = timer - last_timer > 0.1

        if delay_touch_raise:
            self.class_object_data_hand["touchingFace"][hand_label] = []
            self.class_object_data_hand["hand_touch"][hand_label] = []



    def is_touching_face_historique_analyse(self, touching_area, hand_label, okiok):
        """to do"""

        hand_face_touching = self.class_object_data_analyse["touchFaceHisto"][hand_label]

        if len(touching_area) >= 5:

            last_data = [part_name for (part_name, timer) in touching_area][-5:]

            is_same_data = len(set(last_data)) == 1

            if is_same_data:
                hand_face_touching += [(touching_area[-1])]

                #print("ICIIIIIIIIIIIIIIIIIIIIIII", hand_face_touching[-1])

    def touch_face_analyse(self, face_id, hand_label):
        """Recuperate part of the face touch by the hand."""

        # Recuperate data
        hand_touching = self.class_object_data_hand["touchingFace"][hand_label]
        touching_area = self.class_object_data_hand["hand_touch"][hand_label]

        if self.not_empty(hand_touching):

            # Recuperate the sign of the hand in course.
            sign_in_couse = self.recuperate_sign_of_the_hand(hand_label)

            # Choice the area touch in case mulitpl signs is touching.
            area_choice = self.decide_area_touch(hand_touching)

            if area_choice is not None:

                # Recuperate signification of the face touched.
                signification_part_touch = self.recuperate_sign_in_database(area_choice, sign_in_couse, hand_label)

                #print(signification_part_touch)

                touching_area += [(signification_part_touch, self.timer)]

                self.is_touching_face_historique_analyse(touching_area, hand_label)

                # Raise data
                self.raising_work_list_after_a_delay(hand_touching, hand_label)


    def raise_liste_hand_sign(self, hand_sign, label):
        """remove sign list after x frames."""

        if self.not_empty(hand_sign):
            if self.timer - hand_sign[-1][-1] > self.remove_hand_sign_threshold:
                self.class_object_data_hand["sign"][label] = []



    def marquors_sign_hand(self):
        """Recuperate marquor of sign detected of the hand (fist & confidence)."""

        # Recuperate sign of the two hands.
        liste = [j for i in [self.class_object_data_analyse["sign"][label] 
                 for label in ("right", "left")] for j in i]

        # Recuperate sign interest.
        sign_detected = [timer for (sign, timer) in liste if sign in ["fist", "confidence", "communication", "stop"]]

        # Regroup timer by range of 0.3.
        sign_detected = self.utils_groupe_timer_by_range(sign_detected, 0.3)

        # If an interval of move is egal to 0 or highter of 1 secondes remove it.
        sign_detected = [(begin, end) for (begin ,end) in sign_detected if end - begin < 1]

        # Save
        self.class_object_data_analyse["marquors"]["hand_sign_marquor"] = sign_detected


    def analyse_hand_sign(self):
        """Sign analyse."""

        dico = {"peace": 1, "confident": 0, "approbation": 1, "fist": 0, "stop":0, "communication":1, "gun":0}

        for label in self.hand_label:

            hand_sign = self.class_object_data_hand["sign"][label]

            # Recuperate last sign
            last_sign = [sign.split()[0] for (sign, timer) in hand_sign if self.timer is timer]

            # Update hand_sign (temporary list) with only one sign (if there is the same sign many times).
            hand_sign += [(i, self.timer) for i in last_sign]

            # Detection of the sign. If a sign's detected more of threshold add
            # the sign & the time of the detection in the database.
            only_sign = [sign for (sign, timer) in hand_sign]
            counter = [(only_sign.count(i), i) for i in list(set(only_sign))]

            max_counter = [i[0] for i in counter]

            if self.not_empty(max_counter):

                if max(max_counter) > 3:
                    # In case multipl detection, recuperate hierchy of signs.
                    sign = [sign for (c, sign) in counter if c == max(max_counter) and sign in dico]
                    priority = sorted([(dico[s], s) for s in sign])

                    # Save data.
                    self.class_object_data_analyse["sign"][label] += [(priority[-1][1], self.timer)]

            # realease temporary list.
            self.raise_liste_hand_sign(hand_sign, label)

        # Search marquor from hand sign.
        self.marquors_sign_hand()
