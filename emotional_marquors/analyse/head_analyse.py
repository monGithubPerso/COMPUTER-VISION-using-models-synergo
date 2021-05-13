#!/usr/bin/python3
# -*- coding:utf-8 -*-

from utils.function_utils import Utils
import numpy as np

class Head_analyse(Utils):
    """Hean & face movements analysis."""

    def __init__(self):
        """Constructor"""

        # Head leaning
        self.leaning_max_threhsold = 1.35
   
        self.leaning_definate = {
            "right": "Analyse", "left": "Creativity, Empathy",
            "center": "None", None: "None"}


        self.class_object_data_analyse = {}
        self.class_object_data_face = {}
        self.class_object_data_body = {}
        self.timer = 0


        # Threshold for raise detection
        # or threshold do not pass or needed
        # for example "do not pass" marquor can't be more 1 sec
        # "needed" need more 0.1 for be a detection

        # Speed (cm / s) of a head movement.
        # Need face_move_threshold % above the mean of head movement
        # first is the minimum, second is the maximum
        self.face_move_threshold = (300, 400)


    def getter_data_face(self, data_face):
        """Get face data"""
        self.class_object_data_face = data_face

    def getter_data_analyse(self, data_analyse):
        """Get analyse data"""
        self.class_object_data_analyse = data_analyse

    def getter_timer(self, timer):
        """Get timer in the video"""
        self.timer = timer

    def getter_data_body(self, data_body):
        """Get data data"""
        self.class_object_data_body = data_body

    def raise_data(self):
        """Raise data"""
        self.class_object_data_analyse = {}
        self.class_object_data_face = {}
        self.class_object_data_body = {}


    def is_a_false_detection(self, detection_time):
        """Recuperate duration of the marquor and duration of the detection's person."""
        detection = self.class_object_data_face["timer_detection"]

        marquor_duration = sum([end - begin for (begin, end) in detection_time])
        duration = sum([detection[n + 1] - detection[n] for n in range(0, len(detection) - 1, 2)])

        if len(detection) % 2 != 0:
            duration += self.timer - detection[-1]

        return marquor_duration, duration


    def is_a_marquor_if_presence_is_lower_of(self, marquors, label):
        """If marquors > duration of the detection of the person / 4, isn't a marquor."""
        if self.not_empty(marquors):
            time_detection_marquor, duration = self.is_a_false_detection(marquors)
            if time_detection_marquor < duration / 4:
                self.class_object_data_analyse["marquors"][label] = marquors


    @staticmethod
    def movement_is_only_one_direction(leaning_head) -> list():
        """Recuperate leaning of the head (right or left). Verify there are only one side
        in the relevant of data (only left or only right)."""
        movements_of_the_head = [leaning_direction for (leaning_direction, _) in leaning_head]
        there_are_no_right_and_left_moves = list(set(movements_of_the_head))
        verify = False if ("right" and "left" in there_are_no_right_and_left_moves) else True
        return verify


    def head_analyse_leaning(self):
        """Definate & verify a leaning movement. If a leaning's detect, wait the end
        of the movement for save it (signification & time in video of the movement)."""

        leaning_head = self.class_object_data_face["leaning_head"] # Temporaty List

        if self.not_empty(leaning_head):

            # Verify there is only one side (only right or only left).
            its_not_a_movement = self.movement_is_only_one_direction(leaning_head)

            last_process, last_timer = leaning_head[-1]

            if its_not_a_movement:

                # Definate the movement.
                process = self.leaning_definate[last_process]
                # Recuperate the last detection of the leaning.
                end_of_leaning = self.timer - last_timer > 0

                if end_of_leaning:
                    # Recuperate the duration, the signification & the timer of the leaning.
                    duration = (leaning_head[0], leaning_head[-1])
                    self.class_object_data_analyse["head_leaning_definate"] += [(process, duration, self.timer)]
                    self.class_object_data_face["leaning_head"] = []

                else:
                    # If the duration of the leaning's highter of 0.5 seconds considere it"s a leaning.
                    duration = leaning_head[-1][-1] - leaning_head[0][-1]
                    if duration > 0.5:
                        self.class_object_data_analyse["head_leaning_definate"] += [(process, duration, self.timer)]


    def head_analyse_vertical(self, vertical):
        """Definate if it's a bottom or top movement of the head."""

        vertical_movement = self.class_object_data_face["face_direction_y"].lower()
        vertical_definate = self.class_object_data_analyse["face_direction_y_definate"]

        # if movement corresponding with vertical parameter.
        if vertical_movement == vertical:
            vertical_definate += [(vertical_movement, self.timer)]

        # Recuperate only timer.
        marquors = [timer for (move, timer) in vertical_definate if move == vertical]

        # Group timers for example timers are = [0.1, 0.2, 0.4, 0.8, 0.9]
        # If we group timer with a range of two: it give: [(0.1, 0.4), (0.8, 0.9)]
        marquors = self.utils_groupe_timer_by_range(marquors, threshold_time=0.2)
        #
        marquors = [(begin, end) for (begin, end) in marquors if 1 > end - begin > 0]
        #
        self.is_a_marquor_if_presence_is_lower_of(marquors, label=f"marquors_head_{vertical}")


    def analyse_marquors_face_wrinkles(self, marquor_label, data_face_label):
        """Verify detection of the wrinkle."""

        data = self.class_object_data_face[data_face_label]
        #
        detection_time = self.utils_groupe_timer_by_range(data, 0.3)
        #
        detection_time = [(begin, end) for (begin, end) in detection_time if end - begin > 0]
        #
        self.is_a_marquor_if_presence_is_lower_of(detection_time, label=marquor_label)


    def lips_analyse(self):
        """Detection of the lips analysis."""
 
        mouth_data = self.class_object_data_face["mouth_movement"] # Work - temporary list.

        # Recuperate only the time and not the signification.
        recuperate_timer = [timer for (signification, timer) in mouth_data]

        if self.not_empty(recuperate_timer):

            signification, last_move = mouth_data[-1]

            # Verify isn't false detection.
            duration = recuperate_timer[-1] - recuperate_timer[0] > 0.1

            # Verify movement's finish.
            if self.timer - last_move > 0.4:

                if duration:
                    # Save in the analyse data.
                    data = [(recuperate_timer[0], recuperate_timer[-1])]
                    self.class_object_data_analyse["marquors"][f"mouth_marquors_{signification}"] += data

                # Raise work list.
                self.class_object_data_face["mouth_movement"] = []



    def verify_face_facing_from_face_speed(self):
        """Verify isn't a movement."""
       
        # Isn't a movement.
        movement = self.class_object_data_face["face_movement_speed"]
        # Isn't in a marquor of the face movement (because it's a quick movement).
        data_analyse = self.class_object_data_analyse["face_moves"]
        # Marquors to verify
        marquors = self.class_object_data_analyse["marquors"]["face_facing"]

        # Recuperate movement speed of the face and thresholds.
        faces_movements = self.class_object_data_face["face_movement_speed"]
        thresholds = self.recuperate_threshold_face_of_movement(faces_movements)

        [marquors.remove((begin, end)) for (begin, end) in marquors 
        for (speed, timer) in faces_movements
        if end >= timer >= begin and speed >= thresholds[0] and (begin, end) in marquors]



    def can_communicate_with(self, face_showing, face_direction, person):
        """Person can talk with person in case face facing."""

        person_around = None

        # Verify if there are persons around the persin
        face_isnt_center = face_showing is face_direction
        person_direction = "left" if face_direction is "right" else "right"

        if face_isnt_center and self.not_empty(person):
            person_around = [face_id for (direction, face_id) in person if direction is person_direction]

        return person_around


    def marquors_face_showing(self):
        """Recuperate only if the face is to right."""

        face_facings_savegarde = self.class_object_data_analyse["face_facing"]
        face_facings = [timer for (sign, timer) in face_facings_savegarde if sign is "Mefiance"]

        detection_time = self.utils_groupe_timer_by_range(face_facings, 0.3)
        detection_time = [(begin, end) for (begin, end) in detection_time if 1 > end - begin > 0]

        self.is_a_marquor_if_presence_is_lower_of(detection_time, label=f"face_facing")


    def analyse_face_face(self):
        """Definate signification of the side of the face"""

        face_showing = self.class_object_data_face["face_showing"] # Temporaty List
        # Recuperate id persons around person if there are person.
        person_side = self.class_object_data_body["in_social_area"]

        person = [(direction, face_id) for i in person_side for (face_id, direction) in i]

        with_id = [self.can_communicate_with(face_showing, label, person) for label in ["right", "left"]]
        with_id = [with_id.remove(i) for i in with_id if i is None]

        there_is_detected_id = not self.not_empty(with_id) 

        if face_showing is not None:

            if there_is_detected_id:
                process = "Communication" if face_showing is "right" else "Mefiance"

            elif not there_is_detected_id:
                id_face_detected = str([i for i in with_id])[0][1:-1]
                process = f"In Communication with {id_face_detected}"

            self.class_object_data_analyse["face_facing"] += [(process, self.timer)]

        self.marquors_face_showing()
        self.verify_face_facing_from_face_speed()


    def recuperate_threshold_face_of_movement(self, liste_movement):
        """Make the average of the movement speed of the head.
        Recuperate the time in the video if a movement is beetween a certain percent of the mean."""

        # Face movement's compose of a speed movement and the time in video.
        only_speed = [speed for (speed, timer) in liste_movement]
        mean_of_speed = np.mean(np.array(only_speed))

        # Treshold from the average.
        movements_range = [self.percent_of_not_round(mean_of_speed, threshold)
                           for threshold in self.face_move_threshold]

        return movements_range


    def analyse_emotion(self):
        """Recuperate emotion detected if the emotion isn't neutral, surprise or happy."""

        emotion_detected = self.class_object_data_face["emotions"] # Work list.
        emotion_analyse_list = self.class_object_data_analyse["emotion_analyse"] # Save list.

        # Recuperate the last emotion detected.
        emotion_analyse_list += [(emotion_detected, self.timer)]

        # Recuperate data interest by categories.
        dico = {"emotion_fearful": [], "emotion_angry": [], "emotion_disgusted": [], "emotion_sad": []}

        # Recuperate only the timer.
        [dico[f"emotion_{emotion.lower()}"].append(timer) for (emotion, timer) 
        in emotion_analyse_list if f"emotion_{emotion.lower()}" in dico]

        for k, v in dico.items():
            self.class_object_data_analyse["marquors"][k] = []

        # Save marquors.
        for label, liste in dico.items():
            # Regroup marquor by range.
            liste = self.utils_groupe_timer_by_range(liste, threshold_time=0.3)
            liste = self.utils_group_timers(liste)

            liste = [(begin, end) for (begin, end) in liste if end - begin > 0]

            if len(liste) > 0:
                self.is_a_marquor_if_presence_is_lower_of(liste, label)



    def face_movement_analyse(self):
        """Getter face movement speed."""

        movement = self.class_object_data_face["face_movement_speed"]
        data_analyse = self.class_object_data_analyse["face_moves"]

        data_analyse = []

        if self.not_empty(movement):

            # Recuperate thresholds of the face movements.
            fast_movement, to_fast_movement = self.recuperate_threshold_face_of_movement(movement)

            for nb, (speed, timer) in enumerate(movement):
                if to_fast_movement > speed > fast_movement:
                    data_analyse += [timer]


        """
        data_analyse = sorted(data_analyse)
        marquors = self.utils_groupe_timer_by_range(data_analyse, 0.3)
        self.class_object_data_analyse["marquors"]["face_movement"] = marquors
        """