#!/usr/bin/python3
# -*- coding:utf-8 -*-


import os
from scipy.spatial import distance
import cv2
import dlib
import numpy as np
from imutils import face_utils
import random
from person.person import Person
from utils.function_utils import Utils


class Face_tracking(Person, Utils):
    """Tracking the face along the video."""

    def __init__(self, path_predictor, path_recognition, save_path_picture_recognition, path_haar):
        """Constructor"""

        # DLIB faces detection
        self.detector_dlib = dlib.get_frontal_face_detector() # Face detection.
        self.predictor_dlib = dlib.shape_predictor(path_predictor) # Landmarks in the face.

        # DLIB face recognition
        self.face_encoder_dlib = dlib.face_recognition_model_v1(path_recognition)
        self.known_face_encodings_recognition_face = [] #
        self.known_face_names_recognition_face = [] #

        self.tolerance_recognition_face = 0.6 # Min onfidence for a recognition. 

        # Save path for facial recognition.
        self.save_path_picture_recognition = save_path_picture_recognition

        # If the face detection changes, lunch a facial recognition.
        self.last_face_number_detected = None
        self.recognition_facial_call = False

        # Haarcascade parameters.
        self.face_cascade = cv2.CascadeClassifier(path_haar)

        self.scaleFactor = 1.3 
        self.minNeighbors = 5
        self.minSize = (60, 60)

        self.class_object_data_face = {}
        self.timer = 0

        # With the tracking of distance we only recuperate the minimum.
        # Association of two head (last frame, new frame)
        # so we always associate face. We need to define a minimum value
        # of association. Else lunch a facial recognition.
        self.threshold_face_distance_recognition = 25

    def getter_timer(self, timer):
        """ Recuperate time in video"""
        self.timer = timer


    def recuperate_crop_of_the_face(self, frame, landmarks):
        """Verify crop isn't out of the frame."""
        
        frame_height, frame_width = frame.shape[:2]

        x, y, w, h = cv2.boundingRect(np.array(landmarks))

        # Verify crop isn't out of the frame.
        top = y - 20 if y - 20 >= 0 else 0
        bot = y + h + 20 if y + h + 20 <= frame_height else frame_height
        left = x - 20 if x - 20 >= 0 else 0
        right = x + w + 20 if x + w + 20 <= frame_width else frame_width

        return left, top, right, bot

    def savegarde_picture_recognition_facial(self, frame, id_face, landmarks):
        """ Savegarde picture of the face of the person, in the case it's the first apparition"""

        # Folder picture face recognition.
        picture_recog_facial = os.listdir(self.save_path_picture_recognition)
        picture_recog_facial = [picture[:-4] for picture in picture_recog_facial]
        id_face = str(id_face)

        # Person face not in Folder.
        if id_face not in picture_recog_facial:
            
            # Recuperate crop of the face.
            region_face = self.recuperate_crop_of_the_face(frame, landmarks)
            # Verify crop isn't out of the frame else modify points.
            x, y, w, h = self.utils_points_not_out_of_frame(frame, region_face, 0)

            crop = frame[y:h, x:w]

            # Save picture.
            picture_name = self.save_path_picture_recognition + f"/{str(id_face)}.png"
            cv2.imwrite(picture_name, crop)


    def detect_faces(self, img):
        """Detecting face with a haarcascade. It's an old technic but it improve speed."""

        faces_detected = self.face_cascade.detectMultiScale(
            img, scaleFactor=self.scaleFactor, 
            minNeighbors=self.minNeighbors, minSize=self.minSize)

        if len(faces_detected) is 0:
            faces_detected = []
        else:
            faces_detected[:, 2:] += faces_detected[:, :2]

        return faces_detected


    def face_landmarks_detection(self, gray, frame):
        """Recuperate face landmarks and nose point."""

        face_landmarks = []

        te = self.detect_faces(gray)
        for (x, y, w, h) in te:
            face_rect = dlib.rectangle(left = x, top = y, right = w, bottom = h)

            landmarks = self.predictor_dlib(gray, face_rect)
            face_landmarks += [[(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]]
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 1)

        return face_landmarks

    @staticmethod
    def recuperate_minimal_distance(data_face, coordinate_nose_face_detected):
        """Recuperate distance beetween the last & the current point"""
        last_nose = data_face["face_nose_repear"]
        distance_face_detected_current_face = [
            (distance.euclidean(last_nose, coordinate), coordinate)
            for coordinate in coordinate_nose_face_detected]

        return distance_face_detected_current_face


    def update_date_face_tracking_distance(self, data_face, current_coordinate, landmarks_face):
        """Update data"""
        data_face["face_box"] = cv2.boundingRect(np.array([landmarks_face]))
        data_face["face_landmarks"] = landmarks_face
        data_face["face_nose_repear"] = current_coordinate
        data_face["is_detected"] = True
        

    def need_a_facial_recognition_face_detected_to_far(self, distance_head_move):
        """The distance beetween the last & the current coordinates are to far"""

        head_move_distance = [move > 25 for move in distance_head_move]
        there_is_to_long_distance = True in head_move_distance
        need_a_facial_recognition = True if there_is_to_long_distance else False
        return need_a_facial_recognition

    def tracking_face_via_distance(self, frame, faces_detected, face_person):
        """We are trying to locate a face throughout the video.
        For this we recover the coordinates of the nose detected between 2 images and
        we recover the smallest distance between 2 noses"""

        # An effect can move the head like a slinding.
        distance_head_move = []

        # All noses of face detected.
        coordinate_nose_face_detected = [landmarks[33] for landmarks in faces_detected]

        for face_id, data_face in face_person.items():

            # Verify face's detected.
            if data_face["is_detected"]:

                # Search minimal distance.
                distance_face_detected_current_face = self.recuperate_minimal_distance(data_face, coordinate_nose_face_detected)

                if self.not_empty(distance_face_detected_current_face):

                    # Minimal istance of nose beetween last and current frame.
                    distance_beetween_frame, current_coordinate = min(distance_face_detected_current_face)

                    # Update data of face detected.
                    landmarks_face = [landmark for landmark in faces_detected if landmark[33] == current_coordinate][0]
                    self.update_date_face_tracking_distance(data_face, current_coordinate, landmarks_face)

                    # Remove coordinate (avoid a second detection - assigment)
                    coordinate_nose_face_detected.remove(current_coordinate)

                    # Savegarde face for facial recognition.
                    self.savegarde_picture_recognition_facial(frame, face_id, landmarks_face)
                    distance_head_move += [distance_beetween_frame]


        # If a distance is to far.
        need_a_facial_recognition = self.need_a_facial_recognition_face_detected_to_far(distance_head_move)
        return need_a_facial_recognition



    def encode_face_for_recognition(self, frame, channel):
        """Detecting and encoding faces in an image"""

        detection_frame = frame
        face_encodings_list = []
        faces_detected = []

        # Searching face channel 3 or channel 1
        channel_is_gray = channel == 0

        if channel_is_gray:
            detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection.
        faces_detection = self.detector_dlib(detection_frame, channel)
        for face in faces_detection:

            landmarks = self.predictor_dlib(detection_frame, face)
            face_encodings_list += [np.array(self.face_encoder_dlib.compute_face_descriptor(frame, landmarks, 1))]

            landmarks_coordinates = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            faces_detected += [landmarks_coordinates]

        return face_encodings_list, faces_detected



    def training_recognition(self):
        """Recuperate label and encode the picture"""

        # Path folder picture recognition facial.
        label_picture_in_folder_recognition = os.listdir(self.save_path_picture_recognition)

        self.known_face_names_recognition_face = []
        self.known_face_encodings_recognition_face = []

        # Image in folder.
        for label in label_picture_in_folder_recognition:

            path_picture = self.save_path_picture_recognition + "/" + label
            image = cv2.imread(path_picture)

            # Encode and detecte face in picture.
            face_encoded, _ = self.encode_face_for_recognition(image, 1)

            # Recuperate encode and name of picture.
            is_encoded = len(face_encoded) > 0
            if is_encoded:

                # Face id is name picture without extension
                face_id = label[:-4]
                self.known_face_encodings_recognition_face += [face_encoded[0]]
                self.known_face_names_recognition_face += [face_id]


    def facial_recognition(self, frame):
        """ Make differences beetween encodage beetween face in frame and face in
        folder picture facial recognition"""

        rgb_small_frame = frame[:, :, ::-1]

        # Encoding face
        face_encodings, face_locations = self.encode_face_for_recognition(rgb_small_frame, 0)
        there_is_knows_face = len(self.known_face_encodings_recognition_face) > 0

        # Recuperate face recognized or put Unknown.
        face_names = []
        for face_encoding in face_encodings:

            try:
                # Check distance beetween know face and face detected.
                vectors = np.linalg.norm(
                    self.known_face_encodings_recognition_face - face_encoding, axis=1)

                result = [True if vector <= self.tolerance_recognition_face
                          and vector == min(vectors) else False
                          for vector in vectors]

                name = self.known_face_names_recognition_face[result.index(True)]\
                       if True in result else "Unknown"
                face_names += [name]

            except:
                face_names += ["Unknown"]

        return face_locations, face_names




    def define_range_time_detection_face(self, face_person, face_detected):
        """ If face not detected in the current frame. Put detection to false.
        If the face just disapear, put the time of the video of the disapear."""

        for id_face, data in face_person.items():

            if id_face not in face_detected:

                apparition = data["timer_detection"]
                if len(apparition) % 2 != 0:

                    data["timer_detection"] += [self.timer]

                data["is_detected"] = False



    def tracking_face_with_recognition(self, frame, faces_detected, face_person,
                                       hand_person, eye_person, body_person, analyse_person):
        """Number of face detected changed. Use recognition for tracking face
            self.tolerance_recognition_face = 0.6
        """

        face_locations, face_names = self.facial_recognition(frame)
        recognition = [(name, landmarks) for landmarks, name in zip(face_locations, face_names)]


        are_detected = []
        to_create = []
        for (id_face, landmarks) in recognition:

            # Tracking the face.
            if id_face is not "Unknown":

                id_face = int(id_face)

                # Put the video time of the appartition of the face if detection's off.
                wsnt_detected_in_last_frame = face_person[id_face]["is_detected"] == False
                if wsnt_detected_in_last_frame:
                    face_person[id_face]["timer_detection"] += [self.timer]


                data_to_savegarde = {
                    "face_localisation": landmarks[33], 
                    "face_landmarks": landmarks,
                    "face_box": cv2.boundingRect(np.array(landmarks)),
                    "is_detected": True
                }

                for label_data, data in data_to_savegarde.items():
                    face_person[id_face][label_data] = data

                are_detected += [id_face]


            # Create a new face.
            elif id_face is "Unknown":

                # Recuperate all label of face detected.
                last_face_id = [key for key in face_person.keys()]

                # Define a new face id.
                there_is_face_already_detected = len(last_face_id) > 0
                new_face_id = max(last_face_id) + 1 if there_is_face_already_detected else 0

                to_create += [(new_face_id, landmarks[33], landmarks, self.timer)]

                # Fill database with new id.
                database = face_person, hand_person, eye_person, body_person, analyse_person
                self.fill_data(frame, new_face_id, landmarks[33], landmarks, database, self.timer)
                # Save picture for face recognition.
                self.savegarde_picture_recognition_facial(frame, new_face_id, landmarks)

                are_detected += [new_face_id]

        # Put to false face none detected.
        self.define_range_time_detection_face(face_person, are_detected)

        have_found = False
        if recognition == []:
            have_found = True
        else:
            have_found = False


        return have_found

    def hand_in_area_face(self, face_person):
        """ Hand can avoid face detection, don't start a face recognition
        if hand in face area"""

        hand_face_area = [data_face["hand_is_in_head_zone"] for _, data_face in face_person.items()]
        cant_detecte_face_cause_hand = hand_face_area.count(True)

        return cant_detecte_face_cause_hand



    def no_detection_but_coord_face_boxe_are_same(self, face_person, face_detected):


        detected_faces = [i[33] for i in face_detected]

        for face_id, data_face in face_person.items():

            last_face_nose = data_face["last_face_nose_repear"]
            is_detected = data_face["is_detected"]

            if last_face_nose is not None and is_detected:

                for current_nose in detected_faces:

                    x = abs(current_nose[0] - last_face_nose[0]) < 50
                    y = abs(current_nose[1] - last_face_nose[1]) < 50

                    if x and y:
                        detected_faces.remove(current_nose)


        has_been_detected = []
        if len(detected_faces) == 0:

            for landmarks in face_detected:

                current_nose = landmarks[33]

                for face_id, data_face in face_person.items():

                    last_face_nose = data_face["last_face_nose_repear"]
                    is_detected = data_face["is_detected"]

                    if last_face_nose is not None and is_detected:

                        x = abs(current_nose[0] - last_face_nose[0]) < 50
                        y = abs(current_nose[1] - last_face_nose[1]) < 50

                        if x and y:
                            has_been_detected += [face_id]
                            data_face["face_localisation"] = landmarks[33]
                            data_face["face_landmarks"] = landmarks
                            data_face["face_box"] = cv2.boundingRect(np.array(landmarks))
                            data_face["is_detected"] = True


        if len(has_been_detected) > 0:
            for face_id, data_face in face_person.items():
                if face_id not in has_been_detected:
                    data_face["is_detected"] = False

        return False if len(detected_faces) > 0 else True



    def tracking_face(self, frame, face_detected, database):
        """ Tracking face"""

        face_person, hand_person, eye_person, body_person, analyse_person = database

        # Face detected
        number_face_detected = len(face_detected)
        there_are_face_detects = number_face_detected > 0

        # Hand in area face
        cant_detecte_face_cause_hand = self.hand_in_area_face(face_person)
        there_are_hand_in_face_area = cant_detecte_face_cause_hand > 0

        if there_are_face_detects:

            number_faces_has_changed = self.last_face_number_detected is not None and\
                                       self.last_face_number_detected != number_face_detected


            changed_but_hand_avoid_detection = self.last_face_number_detected !=\
                                               (number_face_detected + cant_detecte_face_cause_hand)


            if number_faces_has_changed and changed_but_hand_avoid_detection:
                self.recognition_facial_call = True


            # TRACKING: via distance.
            if self.recognition_facial_call is False:
                to_long_distance = self.tracking_face_via_distance(frame, face_detected, face_person)
                
                if to_long_distance:
                    self.recognition_facial_call = True


            # TRAKING: via face recognition.
            if self.recognition_facial_call or to_long_distance:

                print("Face identification in course...")

                have_found = self.no_detection_but_coord_face_boxe_are_same(face_person, face_detected)
                self.recognition_facial_call = False

                if not have_found:

                    self.training_recognition()
                    no_matching = self.tracking_face_with_recognition(frame, face_detected, face_person,
                                        hand_person, eye_person, body_person, analyse_person)

                    self.recognition_facial_call = no_matching


            if number_face_detected is not 0:
                self.last_face_number_detected = number_face_detected
            else:
                self.recognition_facial_call = True

        elif not there_are_face_detects and not there_are_hand_in_face_area and self.last_face_number_detected is not None:
            self.recognition_facial_call = True
