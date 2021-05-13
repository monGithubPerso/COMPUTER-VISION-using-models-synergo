#!/usr/bin/python3
# -*- coding:utf-8 -*-


from scipy.spatial import distance
import cv2
import numpy as np
import math
from face.skin_color import skin_color_main_function
from utils.function_utils import Utils


class Face_movements(Utils):
    """ """
    def __init__(self):
        """ """

        self.width, self.height = (640, 360)
        self.width_head = 14.9
        self.face_length = 19.3

        # Leaning.
        self.LEARNING_RATE = 1.11
        self.pairs_points = np.array([(36, 45), (36, 30), (45, 30)])

        # Movement.
        self.movement_threshold = 0.0

        # Facing
        self.face_points  = np.array([0, 16, 33])

        self.dico_sign = {
            "no": "left right left",
            "no1": "right left right",
            "yes": "top bot top",
            "yes1": "bot top bot"
            }

        self.liste_mouth_points = np.array([(48, 68), (55, 60), (65, 68)])

        self.class_object_data_face = {}
        self.class_object_data_hand = {}
        self.class_object_data_body = {}
        self.timer = 0
        self.timer_deplacement = 0 

        # Percent to add of the top of the face (eyesbrows)
        self.forehead_from_face = 40
        # Min and max threshold of wrinkle area.
        self.threhsold_detection_wrinkle_forehead = [2, 20]

        # For mesuring the face facing (show left or right joue)
        # we compare distance beetween right eye & left eyes with the nose.
        # If one of these distances representes threshold_face_facing of the other
        # there is a face facing.
        self.threshold_face_facing = 150

        # Treshold for know if the head's going top or bot.
        self.treshold_bottom_top = (-10, 20)

        # Threshold of the difference require beetween two mesures.
        self.threshold_face_leaning = (88, 70)
        # Threshold for know if the shoulder has move in cm
        self.treshold_shoulder_move = 2

    def getter_data_hand(self, data_hand):
        """Get hand data"""
        self.class_object_data_hand = data_hand


    def getter_data_face(self, data_face):
        """Get face data"""
        self.class_object_data_face = data_face


    def getter_data_body(self, data_body):
        """Get body data"""
        self.class_object_data_body = data_body


    def getter_timer(self, timer):
        """Get timer in video"""
        self.timer = timer


    def getter_frame_deplacement(self, timer_deplacement):
        """Get fps"""
        self.timer_deplacement = timer_deplacement


    def raise_data(self):
        """Raise class variables."""
        self.class_object_data_face = {}
        self.class_object_data_hand = {}
        self.class_object_data_body = {}


    def face_facing(self, draw_frame):
        """Face of the face (distance of the eyes & nose). If one distance represents
        a certain % of the other, face is showing one part."""

        x, y, w, h = self.class_object_data_face["face_box"]
        landmarks = self.class_object_data_face["face_landmarks"]

        # Ratio real & video distance.
        ratio_distance = self.get_ratio_distance(self.width_head, mesure=w)
        # Recuperate eyes & nose landmarks.
        eye_right, eye_left, nose = [landmarks[i] for i in self.face_points]
        #[cv2.line(draw_frame, eye, nose, (0, 0, 255), 2) for eye in [eye_right, eye_left]]

        # Real distance eyes - nose to video distance.
        eyes = [self.scaling_distance_round(eye, nose, ratio_distance) for eye in [eye_right, eye_left]]

        distFace = lambda distance1, distance2: self.percent_of(distance1, self.threshold_face_facing) < distance2
        # Define side of the face.
        right_eye_nose, left_eye_nose = eyes
        face_part = "left" if distFace(right_eye_nose, left_eye_nose) else None
        face_part = "right" if distFace(left_eye_nose, right_eye_nose) else face_part

        print("face part", face_part)
        self.class_object_data_face["face_showing"] = face_part


    def face_leaning(self, frame):
        """Define the leaning of the face. For that we comparing the two
        extremums points of the head on the x axis & recuperate the leaning ratio."""

        x, y, w, h = self.class_object_data_face["face_box"]
        landmarks = self.class_object_data_face["face_landmarks"]

        # Point of DLIB needed.
        side_rightY = landmarks[0][1] # right of the head
        side_leftY  = landmarks[16][1] # left of the head

        # Recuperate point of the side of head & the chin.
        right_chin = ( (0, landmarks[0][1]),  (0, landmarks[8][1]) )
        left_chin = ( (0, landmarks[16][1]), (0, landmarks[8][1]) )

        ratio = self.get_ratio_distance(14.9, w)

        # Recuperate distance on Y axis beetween the side of head & the chin.
        side_rightY = self.scaling_distance_round(right_chin[0], right_chin[1], ratio)
        side_leftY = self.scaling_distance_round(left_chin[0], left_chin[1], ratio)

        print(side_rightY, side_leftY)

        # Recuperate the maximum & the minimum distance & compare them
        # if one of them is at a certain percent of the other
        # head's leaning.
        min_value, max_value = sorted([side_rightY, side_leftY])

        threhsold_max, threhsold_min = self.threshold_face_leaning

        print(self.percent_of(max_value, threhsold_min), self.percent_of(max_value, threhsold_max), min_value)

        if self.percent_of_not_round(max_value, threhsold_min) < min_value and min_value < self.percent_of_not_round(max_value, threhsold_max):

            leaning_x = "left" if side_rightY < side_leftY else "right"
            print("leaning", leaning_x)

            self.class_object_data_face["leaning_head"] += [(leaning_x, self.timer)]


    def face_vertical_movement(self):
        """Vertical movement i don't understand how it work but 
        you can check the github in the citation in the part of bottom & top movement of the head"""

        boxe_face = self.class_object_data_face["face_box"]
        landmarks = self.class_object_data_face["face_landmarks"]

        ratio = self.get_ratio_distance(self.width_head, boxe_face[2])

        landmarks = [(landmarks[i], landmarks[j]) for (i, j) in self.pairs_points]

        d_eyes, d1, d2 = [self.scaling_distance_round(pts1, pts2, ratio) for (pts1, pts2) in landmarks]
        coeff = d1 + d2

        cosb = min((math.pow(d2, 2) - math.pow(d1, 2) + math.pow(d_eyes, 2)) / (2*d2*d_eyes), 1)
        lean = int(250*(d2*math.sin(math.acos(cosb))-coeff/4)/coeff)

        print("degres vertical bot top", lean)

        position = "Top" if lean < self.treshold_bottom_top[0] else "Center"
        position = "bot" if lean > self.treshold_bottom_top[1] else position

        self.class_object_data_face["face_direction_y"] = position


    @staticmethod
    def crop_on_the_beetween(gray, frame, draw_frame, boxe_beetween):
        """Recuperate 3 crops of the region beetween the eyes.
        crop: gray crop for recuperate the contours.
        crop_virgin: Verify there are only color of the skin.
        crop_draw: For works."""

        x, y, w, h = boxe_beetween
        crop, crop_virgin, crop_draw = [i[y:h, x:w] for i in [gray.copy(), frame, draw_frame]]
        crop = cv2.GaussianBlur(crop,(5, 5), 0)
        #cv2.rectangle(draw_frame, (x, y), (w, h), (255, 255, 255), 1)

        return crop, crop_virgin, crop_draw


    @staticmethod
    def recuperate_contours_of_crop(crop):
        """Recuperate contours of the beetween eyes."""
        th3 = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]


    def area_of_contour_authorized_beetween(self, boxe_beetween):
        """Recuperate threshold of contours (min & max authorized)."""
        x, y, w, h = boxe_beetween
        surface = (w - x) * (h - y)
        return [self.percent_of(surface, percent) for percent in [20, 8]]


    def face_condition(self):
        """For recuperate contours beetween eye; face must 
        be center and hand cannot be in the face area."""

        face_is_center = self.class_object_data_face["face_showing"] not in ("right", "left")
        hand_in_area_face = not self.class_object_data_face["hand_is_in_head_zone"]
        conditions = False not in [face_is_center, hand_in_area_face]
        return conditions


    def recuperate_boxe_beetween(self, boxe_face, landmarks):
        """Resize the beetween boxe (Point are Dlib point)."""

        x, y, w, h = boxe_face

        x = landmarks[21][0] + self.percent_of(w, 2) # Utils function
        w = landmarks[22][0] - self.percent_of(w, 2)
        y = landmarks[21][1] - self.percent_of(h, 12)
        h = landmarks[42][1] - self.percent_of(h, 5)

        return x, y, w, h


    def beetween_eye(self, draw_frame, frame, gray):
        """Try to detect contour beetween eye as wrinkle."""

        landmarks = self.class_object_data_face["face_landmarks"]

        if self.face_condition():

            boxe_face = self.class_object_data_face["face_box"]
            # Recuperate boxe of the beetween eyes.
            boxe_beetween = self.recuperate_boxe_beetween(boxe_face, landmarks)
            #Verify crop isn't out of the frame
            boxe_beetween = self.utils_points_not_out_of_frame(frame, boxe_beetween, 0)

            # Recuperate crops.
            crop, crop_virgin, crop_draw = self.crop_on_the_beetween(gray, frame, draw_frame, boxe_beetween)

            # Verify percent of black pixels in the crop.
            there_is_no_black_pixel = skin_color_main_function(crop_virgin, 15)

            # Recuperate contours and define max & min contours area.
            contours = self.recuperate_contours_of_crop(crop)
            perimeter_of_crop, minimum = self.area_of_contour_authorized_beetween(boxe_beetween)

            [self.class_object_data_face["beetween_wrinkle"].append(self.timer) for cnts in contours
            if minimum <= cv2.contourArea(cnts) < perimeter_of_crop and there_is_no_black_pixel]

            #[cv2.drawContours(crop_draw, cnts, -1, (0, 0, 0), cv2.FILLED) for cnts in contours
            #if minimum <= cv2.contourArea(cnts) < perimeter_of_crop and there_is_no_black_pixel]


    def get_forehad_from_face(self, boxe_face, face_movement_on_y):
        """Recuperate the boxe of the forehead."""

        x, y, w, h = boxe_face
        height = y - self.percent_of(h, self.forehead_from_face)

        b = height if height > 0 else 0
        c = x + w
        return x, b, c, y


    def define_contours_avaible_forehead(self, boxe_forehead):
        """From the forehead boxe, define the minimum & maximum contours for a detection."""
        # w & h are already add (x + w)
        x, y, w, h = boxe_forehead
        surface = (w - x) * (h - y)
        return [self.percent_of(surface, percent) for percent in self.threhsold_detection_wrinkle_forehead]


    @staticmethod
    def recuperate_border_of_contours(cnt):
        """Extremums of a contours (right, left, top , bot)"""
        xextremum, yextremum = [tuple(cnt[cnt[:, :, nb%2].argmin()][0]) for nb in range(2)] #left, right
        wextremum, hextremum = [tuple(cnt[cnt[:, :, nb%2].argmax()][0]) for nb in range(2)] # top, bottom
        return xextremum, yextremum,  wextremum, hextremum


    def get_mesure_of_contours(self, width_face, xtremums):
        """Get length of the contour."""

        xextremum, yextremum,  wextremum, hextremum = xtremums
        ratio = self.get_ratio_distance(self.width_head, width_face)
        pairs = [((wextremum[0], 0), (xextremum[0], 0)), ((0, hextremum[1]), (0, yextremum[1]))]
        return [self.scaling_distance_round(i, j, ratio) for (i, j) in pairs]


    def verify_contour_color(self, frame, contours, percent):
        """Sometimes person can has a cap for example. Verify the contour's skin color."""
        cnt = self.get_crop(frame, cv2.boundingRect(np.array(contours)), width_height_already_add=False)
        only_skin_color = skin_color_main_function(cnt, percent)
        return only_skin_color


    def recuperate_crops(self, gray, frame, copy, boxe_forehead):
        """Recuperate differents crops needed."""
        return [self.get_crop(picture, boxe_forehead, width_height_already_add=True) for picture in [gray, copy, frame]]


    @staticmethod
    def verify_is_horizontal(xtremums_of_contour):
        """Verify width of the contour's highter of the height."""
        _, _,  wextremum, hextremum = xtremums_of_contour
        return wextremum > hextremum


    def foreheahd(self, copy, frame, gray):
        """Try to detect contours in the forehead as wrinkles."""
        boxe_face = self.class_object_data_face["face_box"]
        face_movement_on_y = self.class_object_data_face["face_direction_y"]
        hand_in_face_area = self.class_object_data_face["hand_is_in_head_zone"]

        # Get forehead (not include in the model) from face coordinates.
        boxe_forehead = self.get_forehad_from_face(boxe_face, face_movement_on_y)
        # Verify points aren't out of the crop.
        boxe_forehead = self.utils_points_not_out_of_frame(frame, boxe_forehead, 0)

        # Finger in face localisation can makes falses detections.
        if not hand_in_face_area:

            # Recuperate crops need of the forehead.
            crop, crop_col, crop_virgin = self.recuperate_crops(gray, frame, copy, boxe_forehead)

            # Recuperate contours (th3) of crop.
            contours = self.recuperate_contours_of_crop(crop)

            # Define mini & maxi contours area.
            mini, maxi = self.define_contours_avaible_forehead(boxe_forehead)

            for i in contours:
                if maxi >= cv2.contourArea(i) > mini:

                    # Recuperate contours extremities
                    xtremums_of_contour = self.recuperate_border_of_contours(i)
                    # Width / height says if the contours is almost flat
                    width, height = self.get_mesure_of_contours(boxe_face[2], xtremums_of_contour)
                    # Verify color of the contours (skin detection)
                    there_is_no_black_pixel = self.verify_contour_color(crop_virgin, i, 10)
                    #
                    is_horizontal = self.verify_is_horizontal(xtremums_of_contour)
                    #
                    if width / height > 3.5 and there_is_no_black_pixel and is_horizontal:
                        cv2.drawContours(crop_col , i, -1, (0, 0, 255), 1)
                        self.class_object_data_face["foreheahd"] += [self.timer]



    def ratio_for_is_uni_or_bi_axis(self, x1, y1, x2, y2, ratio_dist):
        """The ratio say us if the direction is uni or bi-axial."""

        # Recuperate distance.
        on_x = self.scaling_distance_not_round((x1, 0), (x2, 0), ratio_dist)
        on_y = self.scaling_distance_not_round((0, y1), (0, y2), ratio_dist)

        # Sorted them
        min_value, max_value = sorted([on_x, on_y])

        ratio = 5 if min_value == 0 else max_value / min_value, max_value, on_x
        return ratio


    def face_sign_move(self, nose, last_face_coord, ratio_dist):
        """Direction of the face movement."""
        move_face = None

        if last_face_coord is not None:

            dist = self.scaling_distance_not_round(last_face_coord, nose, ratio_dist)

            x1, y1 = nose
            x2, y2 = last_face_coord

            if dist >= 0.3:
                ratio, max_value, on_x = self.ratio_for_is_uni_or_bi_axis(x1, y1, x2, y2, ratio_dist)

                if ratio > 3:
                    if on_x == max_value:
                        move_face = ["left" if x1 < x2 else "right"] * 2
                    else:
                        move_face = ["top" if y1 < y2 else "bot"] * 2
                else:
                    move_face = ["left" if x1 < x2 else "right", "top" if y1 < y2 else "bot"]

                move_face = " ".join(move_face)

        return move_face



    def shoulder_has_moved(self, epaul1, epaul2, ratio):
        """Verify shoulder hasn't move. Recuperate last & current
        points and compare the distance with a threshold."""

        has_move = True
        print(epaul1, epaul2)
   
        if epaul1 is not None and len(epaul1) > 0:
            move = self.scaling_distance_not_round(epaul1[:2], epaul2[:2], ratio)
            has_move = False if move < self.treshold_shoulder_move else True

        return has_move


    def shoulder_move(self, epaul_right, epaul_left, ratio_dist):
        """Verify if the shoulder's has move."""

        last_epaul_right = self.class_object_data_face["last_epaul_right"]
        last_epaul_left = self.class_object_data_face["last_epaul_left"]

        # Make the distance beetween last & current shoulder points.
        epaul_right_has_move = self.shoulder_has_moved(last_epaul_right, epaul_right, ratio_dist)
        epaul_left_has_move = self.shoulder_has_moved(last_epaul_left, epaul_left, ratio_dist)

        return epaul_right_has_move, epaul_left_has_move


    @staticmethod
    def recuperate_movement_of_face(face_moving_list, axis):
        """ """

        sign_list = []

        last_movement = ""
        counter_movement = 0
 
        for (movement, time) in face_moving_list:
 
            movement = movement.split()[axis]

            if last_movement is "":
                last_movement = movement
                counter_movement += 1

            else:
                if last_movement == movement:
                    counter_movement += 1

                else:
                    sign_list += [(last_movement, counter_movement)]
                    last_movement = movement
                    counter_movement = 1

        sign_list += [(last_movement, counter_movement)]

        only_sign_significant = " ".join([
            move for (move, nb_apparition) in sign_list if nb_apparition >= 3])


        return only_sign_significant


    def remove_data(self):
        """ """
 
        face_moving_list = self.class_object_data_face["face_moving"]

        if len(face_moving_list) > 0:
            last_timer = face_moving_list[-1][-1]

            if self.timer - last_timer > self.timer_deplacement:
                self.class_object_data_face["face_moving"] = []


    def update_data_sign(self, nose, shoulder_right, shoulder_left):
        """ """

        #self.class_object_data_face["last_face_nose_repear"] = nose
        self.class_object_data_face["last_epaul_right"] = shoulder_right[:2]
        self.class_object_data_face["last_epaul_left"] = shoulder_left[:2]


    def get_data_face_sign(self):
        """ """
        label = ["face_nose_repear", "last_face_nose_repear", "face_moving"]
        return [self.class_object_data_face[data] for data in label]


    def face_sign(self):
        """ """

        boxe_face = self.class_object_data_face["face_box"]
        body_landmarks = self.class_object_data_body["landmarks"]

        if self.not_empty(body_landmarks):

            nose, last_face_coord, face_moving_list = self.get_data_face_sign()
            shoulder_right, shoulder_left = body_landmarks[11:13]
 
            ratio_dist = self.get_ratio_distance(self.width_head, boxe_face[2])

            if last_face_coord is not None:

                move_face = self.face_sign_move(nose, last_face_coord, ratio_dist)

                epaul_right_has_move, epaul_left_has_move\
                        = self.shoulder_move(shoulder_right, shoulder_left, ratio_dist)

                body_hasnt_move = epaul_right_has_move == epaul_left_has_move == False


                if move_face is not None and body_hasnt_move:

                    face_moving_list += [(move_face, self.timer)]

                    # No
                    only_sign_significant = self.recuperate_movement_of_face(face_moving_list, 0)

                    for sign_name, representation in self.dico_sign.items():
                        if only_sign_significant.find(representation) >= 0:
                            print(sign_name)
                            self.class_object_data_face["face_moving"] = []


                    # Yes
                    only_sign_significant = self.recuperate_movement_of_face(face_moving_list, 1)

                    for sign_name, representation in self.dico_sign.items():
                        if only_sign_significant.find(representation) >= 0:
                            print(sign_name)
                            self.class_object_data_face["face_moving"] = []

            self.remove_data()
            self.update_data_sign(nose, shoulder_right, shoulder_left)


    def mouth_aspirate(self, boxe_face, boxe_mouth, face):
        """Lips in kiss form"""
 
        xFace, yFace, wFace, hFace = boxe_face
        xMouth, yMouth, wMouth, hMouth = boxe_mouth

        # Width & height thresholds.
        width_aspiration = wMouth < self.percent_of(wFace, 30)
        height_aspitation = self.percent_of(hFace, 25) >= hMouth > self.percent_of(hFace, 18)

        # Distance height mouth.
        ratio = self.get_ratio_distance(self.width_head, wFace)
        openure = self.scaling_distance_not_round(face[51], face[57], ratio)
        is_not_open_but_aspirate = 3.2 > openure >= 2.8

        if width_aspiration and height_aspitation and is_not_open_but_aspirate:
            self.class_object_data_face["mouth_movement"] += [("honey", self.timer)]


    def hide_something(self, boxe_face, boxe_mouth, face):
        """Lips tucked into the mouth"""

        xFace, yFace, wFace, hFace = boxe_face
        xMouth, yMouth, wMouth, hMouth = boxe_mouth

        #
        height_hide = hMouth <= self.percent_of(hFace, 12)

        pts_top_mouse = face[51]
        pts_bot_mouse = face[62]

        #
        ratio = self.get_ratio_distance(self.width_head, wFace)
        openure_top = self.scaling_distance_not_round(pts_top_mouse, pts_bot_mouse, ratio)
        openure_bot = self.scaling_distance_not_round(face[66], face[57], ratio)

        #
        top_indoor = openure_top < 0.75
        bot_indoor = openure_bot < 0.6

        #
        if height_hide and top_indoor and bot_indoor:
            self.class_object_data_face["mouth_movement"] += [("hide", self.timer)]


    def lips_movements(self, frame):
        """ """

        boxe_face         = self.class_object_data_face["face_box"]
        landmarks_face    = self.class_object_data_face["face_landmarks"]
        hand_in_face_area = self.class_object_data_face["hand_is_in_head_zone"]

        if not hand_in_face_area:

            # Boxe of mouse.
            boxe_mouth = cv2.boundingRect(np.array([landmarks_face[i] for i in range(48, 68)]))

            # Representations of mouth.
            self.mouth_aspirate(boxe_face, boxe_mouth, landmarks_face)
            self.hide_something(boxe_face, boxe_mouth, landmarks_face)


    def update_data_face_movement(self, direction, movement_in_cm, current_coord):
        """Update direction distance & coordinates of the face"""

        dir_move_time = (" ".join(direction), movement_in_cm, self.timer)

        self.class_object_data_face["face_coordinate_historic"] += [dir_move_time]
        self.class_object_data_face["last_face_moves_coordinates"] = current_coord


    def face_movement(self):
        """Face movements. Recuperate direction, distance 
        with last & currents coordinates."""

        boxe_face = self.class_object_data_face["face_box"]
        last_nose = self.class_object_data_face["last_face_nose_repear"]
        current_nose = self.class_object_data_face["face_nose_repear"]

        if last_nose is not None:

            # Convert real distance to video distance.
            ratio_distance = self.get_ratio_distance(self.width_head, boxe_face[2])
            movement_in_cm = self.scaling_distance_round(current_nose, last_nose, ratio_distance)
            direction = self.define_a_direction(last_nose, current_nose)
            self.update_data_face_movement(direction, movement_in_cm, current_nose)


    def face_in_movement_in_left_or_right(self):
        """Recuperate speed of the face movement
        beetween the last & the current coordinates of the nose & the time in the video."""

        boxe_face = self.class_object_data_face["face_box"]
        last_nose = self.class_object_data_face["last_face_nose_repear"]
        current_nose = self.class_object_data_face["face_nose_repear"]
        last_timer = self.class_object_data_face["last_timer"]

        if last_nose is not None:

            ratio = self.get_ratio_distance(14.9, boxe_face[2])
            # Recuperate distance beetween last & current nose.
            dist = self.scaling_distance_not_round(last_nose, current_nose, ratio)
            # Make difference beetween the last & the current timer.
            time = self.timer - last_timer
            # Make speed.
            speed = dist / time

            self.class_object_data_face["face_movement_speed"] += [(speed, self.timer)]
