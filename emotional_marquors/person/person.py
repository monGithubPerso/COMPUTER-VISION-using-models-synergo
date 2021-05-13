#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np


class Person:
    """ """

    def __init__(self, save_path_picture_recognition):
        """ """
        
        # Database.
        self.face_person = {}
        self.hand_person = {}
        self.eye_person = {}
        self.body_person = {}
        self.analyse = {}

        # Path picture facial recognition.
        self.save_path_picture_recognition = save_path_picture_recognition


    def create_face_person_dictionary(self, face_person, id_face, timer):
        """ Create face person dictionnary. """

        face_person[id_face] = {

            # Tracking.
            "timer_detection": [], # Range of time of detection in video,
            "is_detected":True, # if face's on the frame,

            "face_nose_repear": None,
            "last_face_nose_repear": None,

            "last_epaul_right":[],
            "last_epaul_left":[],

            "face_moving": [],

            # Informations on the face.
            "skin_color": None,
            "gender": [],
            "older": [],
            "emotions": None,

            # Face movements.
            "face_landmarks":None,
            "face_box": None,
            "face_direction": None,
            "face_showing": None,
            "leaning_head": [],
            "face_direction_y": None,
            "last_face_box": None,
            "face_coordinate_historic": [], # direction, distance & speed.

            "beetween_wrinkle": [],
            "foreheahd": [],

            # save area face because
            # hand in face area can avoid face detection,
            # but we want know if hand touch face.
            "hand_is_in_head_zone": False,

            "face_area": {
                "beetween_eye": [], "nose_right": [], "nose_left": [],
                "mouse": [], "under_nose": [], "menton_right": [],
                "menton_left": [], "joue_right": [], "joue_left": [],
                "menton": [], "tempe_right": [],  "tempe_left": [],
                "forehead_right": [], "forehead_left": [],
                "forehead": [], "on_head": [],
                },
            
            "mouth_movement": [],
            "face_movement_speed": [],
            "last_timer": timer,
        }




    def create_hand_person_dictionary(self, hand_person, id_face):
        """ Create hand person dictionnary. """

        hand_person[id_face] = {

            # Hand tracking
            "hands": {"right": None, "left": None},
            "has_been_detected": {"right": None, "left": None},
 
            "landmarks": {"right": [], "left": []},

            "boxe": {"right": None, "left": None},
            "lastBoxe": {"right": None, "left": None},

            "last_palm": {"right": None, "left": None},

            # Hand movements
            "direction of the hand": {"right": None, "left": None}, # major - palm angulus
            "faceHand": {"right": "None", "left": "None"},
            "speed": {"right": None, "left": None}, # last - present movement

            "sign": {"right": [], "left": [],
                     "detections_right": None, "detections_left": None}, 
            "signHistoric": {"right": [], "left": []},

            "distance": {"right": None, "left": None},
            "sequence": {"right": [], "left": []},

            "touchingFace": {"right": [], "left": []}, # hand in face area
            "hand_touch": {"right": [], "left":[]},
            "hand_localisation": {"right": [], "left":[]},
        }


    def create_eye_person_dictionary(self, eye_person, id_face):
        """ """

        eye_person[id_face] = {

            # Eyes tracking
            "eyes" : (None, None),
            "last_eyes": (None, None),

            # Open eye - blink
            "open" : True,
            "closing" : [],
            "closing_historic": [],
            "during_eyes_closing": [],

            "frequency_closing": {"from_mean": "==", "by_min": 0},
            "time_closing": [],
            "eyes_move": [],
            "is_closing": [],
        }


    def create_body_person_dictionnary(self, body_person, id_face):
        """ """

        body_person[id_face] = {

            "social_area": [],
            "in_social_area": None,
            "color": None,
            "pixels_colors": [],
            "contour_body": None,
            "landmarks": [],

            "arm_position": {
                "right": {
                    "coordinates": None,
                },
                             
                "left": {
                    "coordinates": None,
                },
            },

            "last_arm_position": {
                "right": {
                    "coordinates": None,
                    "timer": None,
                },
                             
                "left": {
                    "coordinates": None,
                    "timer": None,
                },
            },

            "arm_movement": {
                "right": {"speed": [], "distance": [], "direction": [], "timer":[]},
                "left": {"speed": [], "distance": [], "direction": [], "timer":[]},
                },

            "sign": []
        }


    def create_analyse_person_dictionnary(self, analyse, id_face):
        """ """
 
        analyse[id_face] = {

            "closing_eye_definate": [],
            "closing_eye_react": [],

            "during_eyes_closing": [],

            "head_leaning_definate":[],

            "head_direction": [],
            "head_direction_marquor": [],

            "face_emotions": [],

            "face_direction_y_definate": [],
            "face_direction_y": [],

            "head_move_dist": [],

            "marquors": {"marquors_eyes": [], "marquors_head_bot": [],
                         "marquors_face": [], "marquors_forehead": [],
                         "marquors_eyes_time": [], "mouth_marquors_hide": [],
                         "mouth_marquors_honey":[],
                         "marquor_hand": [], "marquors_head_top": [],
                         "face_movement": [], "face_facing":[], 
                         "emotion_fearful": [], "emotion_angry": [],
                         "emotion_disgusted": [], "emotion_sad": [],
                         "marquors_eyes_anxiety":[], "hand_sign_marquor":[],
                         },


            "face_facing":[],
            "touchFaceHisto": {"right": [], "left": []},

            "sign": {"right" : [], "left": []},

            "profil": {"process": None, "mood": None},
            "eyes_movement": [],

            "arm_movement": {
                "right": {"speed": [], "distance": [], "direction": [], "timer":[]},
                "left": {"speed": [], "distance": [], "direction": [], "timer":[]},
            },

            "body_sign": [],
            "leaning_head": [],
            "hand_process": [],
            "hand_confidence": [],
            "face_moves":[],
            "emotion_analyse": [],
            "process_hand": [],
            "process_hand_timer": {"right":[], "left":[]}
        }



    def getter_database_person(self):
        """Getter data on the person"""

        data_person = [
            self.face_person, self.hand_person,
            self.eye_person, self.body_person, self.analyse
        ]

        return data_person


    def fill_data(self, frame, id_face, repear_coordinate, landmarks, database, timer):

        face_person, hand_person, eye_person, body_person, analyse = database

        # Add label & data to dictonaries.
        self.create_face_person_dictionary(face_person, id_face, timer)
        self.create_hand_person_dictionary(hand_person, id_face)
        self.create_eye_person_dictionary(eye_person, id_face)
        self.create_body_person_dictionnary(body_person, id_face)
        self.create_analyse_person_dictionnary(analyse, id_face)

        # Data need for a first apparition from recognition on the video.
        data_label_for_tracking_the_person = {
             "face_nose_repear": repear_coordinate,
            "face_landmarks": landmarks, "face_box": cv2.boundingRect(np.array(landmarks))
            }

        for label, data in data_label_for_tracking_the_person.items():
            face_person[id_face][label] = data

        face_person[id_face]["timer_detection"] += [timer]


    def build_data(self, id_face, repear_coordinate, landmarks, timer):

        self.create_face_person_dictionary(self.face_person, id_face, timer)
        self.create_hand_person_dictionary(self.hand_person, id_face)
        self.create_eye_person_dictionary(self.eye_person, id_face)
        self.create_body_person_dictionnary(self.body_person, id_face)
        self.create_analyse_person_dictionnary(self.analyse, id_face)

        # Data need for a first apparition from detection on the video.
        self.face_person[id_face]["timer_detection"] += [timer]
        self.face_person[id_face]["face_nose_repear"] = repear_coordinate
        self.face_person[id_face]["face_landmarks"] = landmarks
        self.face_person[id_face]["face_box"] = cv2.boundingRect(np.array(landmarks))


    def create_person(self, faces_detected, timer):

        """If new ID: create new dictionnary.
        *face[33] = nose."""

        [self.build_data(id_face, face[33], face, timer)
         for id_face, face in enumerate(faces_detected)
         if id_face not in self.face_person]


    def reinit_hand_data(self):
        """ """

        for id_face, data_hand in self.hand_person.items():

            data_hand["boxe"]["right"] = None
            data_hand["boxe"]["left"] = None
            data_hand["landmarks"]["right"] = []
            data_hand["landmarks"]["left"] = []
            data_hand["has_been_detected"]["right"] = None
            data_hand["has_been_detected"]["left"] = None

    def reinit_body_data(self):
        """ """
        for id_face, data_body in self.body_person.items():
            data_body["arm_movement"] = {
                "right": {"speed": [], "distance": [], "direction": [], "timer":[]},
                "left": {"speed": [], "distance": [], "direction": [], "timer":[]},
                }

    def reinit_eyes_data(self):
        """ """
        for id_face, data_eye in self.eye_person.items():
            data_eye["open"] = True

    def reinit_face_data(self, timer):
        """ """
        for id_face, data_face in self.face_person.items():
            data_face["last_face_nose_repear"] = data_face["face_nose_repear"]
            data_face["last_timer"] = timer
            data_face["last_face_box"] = data_face["face_box"]
            data_face["face_box"] = None


    def put_false_detections(self, timer):
        """ """

        # Reinitialise hand data for the next frame.
        self.reinit_hand_data()
        # Reinitialise body data for the next frame.
        self.reinit_body_data()
        # Reinitialise eyes data for the next frame.
        self.reinit_eyes_data()
        # Reinitialise faces data for the next frame.
        self.reinit_face_data(timer)




        
