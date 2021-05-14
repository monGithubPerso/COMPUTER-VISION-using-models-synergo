

from utils.function_utils import Utils

from cv2 import rectangle, boundingRect, resize
from math import ceil, atan2, degrees
from PIL import Image
from body.color import ColorNames
from collections import defaultdict
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import cv2

class Body_distance(Utils):
    """Detection of the body, tracking body, space around body, color of the body."""
    def __init__(self):
        """Constructor"""

        Utils.__init__(self)

        # Area around the body. 50 cm, 1m50 & 3m in pixels.
        self.areas = [1889.765, 4535.436, 11338.59]

        # Mediapipe body detection.
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5,
            upper_body_only = True,
            static_image_mode=False,
        )

        self.class_object_data_face = {}
        self.class_object_data_hand = {}
        self.class_object_data_body = {}
        self.timer = 0

        # Average of the width of a face.
        self.width_face = 14.9

        # Points of the landmarks body.
        self.points_dico = {"right_epaul": [12, 14, 16], "left_epaul": [11, 13, 15],
                            "right_taille": [12, 24], "left_taille": [11, 23]
            }

        self.arm_points = {"left": [12, 14, 16], "right": [11, 13, 15]}
        self.shoulders_points = [[12, 11], [23, 24]]
        self.arm_label = ["right", "left"]

        # Threshold of a face in the body.
        self.treshold_body_face_distance = 50


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
        """Getter timer of the video"""
        self.timer = timer

    def raise_data(self):
        """Raise data"""
        self.class_object_data_face = {}
        self.class_object_data_hand = {}
        self.class_object_data_body = {}



    def body_detection(self, rgb_frame):
        """Body detection in a resize frame."""
        frame_resize_for_searching, (a, b) = self.utils_get_resize_frame_and_return_ratio(
            rgb_frame, xResize=400, yResize=225)

        results = self.pose.process(frame_resize_for_searching)

        landmarks_body = []
        if results.pose_landmarks:

            int_point = lambda pts, shape, ratio: int(pts * shape * ratio)
            landmarks_body = [(int_point(data_point.x, rgb_frame.shape[1], a), 
                            int_point(data_point.y, rgb_frame.shape[0], b), 
                            data_point.visibility) for data_point in results.pose_landmarks.landmark]

        return landmarks_body


    def body_association(self, face_pers, body_pers, body_detected):
        """Associate the body to a face"""

        if self.not_empty(body_detected):

            # Get first landmarks of body from the Mediapipe body model (nose on the face.).
            nose_body_landmarks = body_detected[0][:2]

            # Get all nose repear of face detected from DLIB model.
            face_detected = [(data["face_nose_repear"], face_id) for face_id, data in face_pers.items()]

            # Recuperate the minimum distance beetween last two points (nose Dlib & nose Mediapipe).
            mini_distance = [(distance.euclidean(nose_body_landmarks, repear), face_id)
                            for (repear, face_id) in face_detected]

            if self.not_empty(mini_distance):
                min_dist, face_id = min(mini_distance)
                if min_dist <= self.treshold_body_face_distance:
                    body_pers[face_id]["landmarks"] = body_detected


    def body_spaces(self, frame):
        """Recuperate 3 spaces around the current person (50cm, 1m20 & 3m)."""

        x, y, w, h = self.class_object_data_face["face_box"]

        human_height_in_function_of_height_face = 7 * h
        center_x_face = (x + x + w) // 2

        ratio_length = self.get_ratio_distance(self.width_face, w)
        spaces_around_person = [ceil(i / ratio_length) for i in self.areas]

        # Search a face in one of the areas.
        space_around_person = [(center_x_face - rayon, y, center_x_face + rayon, 
                                human_height_in_function_of_height_face)
                                for rayon in spaces_around_person]

        self.class_object_data_body["social_area"] = space_around_person



    @staticmethod
    def recuperate_face_detected_in_the_frame(face_person, current_face):
        return [(face_id, data_face["face_nose_repear"])
                for face_id, data_face in face_person.items()
                if data_face["face_nose_repear"] is not current_face and data_face["is_detected"]]


    def other_body_in_space(self, face_person):
        """Déctection d'autres visages autours du visage."""

        current_face_detected = self.class_object_data_face["face_nose_repear"]

        other_faces_detected_in_video =\
        self.recuperate_face_detected_in_the_frame(face_person, current_face_detected)

        area_face = [[] for i in range(3)]

        if self.not_empty(other_faces_detected_in_video):

            # Recuperate space around body.
            areas_around_body = self.class_object_data_body["social_area"]

            # In spaces around body,
            for index_space, (x, y, w, h) in enumerate(areas_around_body):

                # In face detected,
                for face_id, (fx, fy) in other_faces_detected_in_video:

                    # If face in space,
                    if abs(w) > fx > abs(x) and h > fy > y:

                        # Definate side.
                        side = "right" if fx > current_face_detected[0] else "left"
                        area_face[index_space] += [(face_id, side)]

        self.class_object_data_body["in_social_area"] = area_face



    def get_sorted_axis(self):
        """For raise somes unknows errors sort x & w, y and h."""
        xb, yb, wb, hb = self.class_object_data_body["contour_body"]
        return sorted([yb, hb]), sorted([xb, wb])


    def verify_hand_arent_in_body_area(self):
        """Verify hand arm aren't in body area, because the main color of the
        cloth can be skin color"""

        body_landmarks = self.class_object_data_body["landmarks"]
        body_boxe = self.class_object_data_body["contour_body"]

        hand_are_in_body_boxe = True

        if self.not_empty(body_landmarks) and body_boxe is not None:

            x, y, w, h = body_boxe
            arms = [body_landmarks[points] for points in np.array([16, 22, 18, 20, 21, 15, 19, 17])]
            arm_in_boxe = [self.utils_point_is_in_boxe(pts, x, y, w, h) for pts in arms]
            hand_are_in_body_boxe = True if True in arm_in_boxe else False

        return hand_are_in_body_boxe



    def body_color(self, frame, display):
        """Recuperate an approximative color of top cloth in the area below the face."""

        # if the color of the body isn't already detected.
        if self.class_object_data_body["color"] is None:

            # Arm in body area can false the color detection of the cloth.
            # Verify they aren't in the area.
            arm_in_body = self.verify_hand_arent_in_body_area()
            if not arm_in_body:

                (y, h), (x, w) = self.get_sorted_axis()

                crop = Image.fromarray(frame[y:h, x:w])

                # Recuperate value of the pixels in the crop
                dico_pixels = defaultdict(int)
                for value in crop.getdata():
                    dico_pixels[value] += 1

                # Recuperate the 100 firsts colors in the crop
                r_g_b_pixels = list({k: v for k, v in sorted(dico_pixels.items(), key=lambda item: item[1])})

                # Recuperate the name of the 100 firsts colors from ColorNames class (color.py).
                name_color = [ColorNames.findNearestWebColorName((r, g, b)) for (b, g, r) in r_g_b_pixels]

                # Count the presence of the name colors.
                dico_colors = {i:0 for i in name_color}
                for i in name_color:
                    dico_colors[i] += 1

                # Recuperate the name color the most presents.
                color = list({k: v for k, v in sorted(dico_colors.items(), key=lambda item: item[1])})[-1]

                self.class_object_data_body["color"] = color



    @staticmethod
    def not_out_supp(value, threshold):
        """Value doesn't superior the threshold"""
        return value if value > threshold else threshold

    @staticmethod
    def not_out_less(value, threshold):
        """Value doesn't inferior the threshold"""
        return value if value < threshold else threshold


    def recuperate_shoulders(self, landmarks_body):
        """Body part from body skeletton."""

        shoulders = []
        body = []

        if self.not_empty(landmarks_body):

            # Recuperate shoulders landmarks if the visibility is highter than 0.5.
            # Recuperate only the coordinates. landmarks_body = (coordinate x, coordinates y, visibility)
            get_landmarks = lambda liste: [landmarks_body[i][:2] for i in liste if landmarks_body[i][-1] >= 0.5]
            shoulders = get_landmarks([12, 11])
            body = get_landmarks([23, 24])

        return shoulders, body


    @staticmethod
    def recuperate_points_of_the_body_no_landmarks(boxe_face, human_width, human_height):
        """Body part from face mesures."""
        xFace, yFace, wFace, hFace = boxe_face

        x = xFace - human_width
        y = yFace + hFace
        w = xFace + human_width + wFace
        h = yFace + hFace + human_height

        return x, y, w, h

    @staticmethod
    def recupeate_points_shoulder_points(shoulders, human_height, boxe_face):
        """Body part from body skeletton."""

        xFace, yFace, wFace, hFace = boxe_face

        x = shoulders[0][0]
        y = shoulders[0][1]
        w = shoulders[1][0]
        h = yFace + hFace + human_height

        return x, y, w, h


    def updata_body_contour_in_function_of_data(self, rgb_frame, boxe_face, there_is_shoulder, there_is_body, shoulders, body):
        """For the display and the body color, recuperate part of the body.
        With body skeletton or from the face mesures."""

        human_width = int(0.9 * boxe_face[2])
        human_height = 7 * boxe_face[3]

        body_boxe = None

        # No landmarks detected.
        if not there_is_shoulder:
            # Recuperate boxe of the body & verify border aren't out side the frame.
            boxe = self.recuperate_points_of_the_body_no_landmarks(boxe_face, human_width, human_height)
            body_boxe = self.utils_points_not_out_of_frame(rgb_frame, boxe, margin_if_out_frame=20)

        # Only shoulder points detected.
        elif there_is_shoulder and not there_is_body:
            boxe = self.recupeate_points_shoulder_points(shoulders, human_height, boxe_face)
            body_boxe = self.utils_points_not_out_of_frame(rgb_frame, boxe, margin_if_out_frame=0)

        # All body detected.
        elif there_is_shoulder and there_is_body:
            x, y, w, h = boundingRect(np.array(shoulders + body))
            body_boxe = (x, y, x+w, y+h)

        return body_boxe


    def body_contours(self, rgb_frame, draw_frame):
        """Define body in function of the face dimensions."""

        boxe_face = self.class_object_data_face["face_box"]
        body_landmarks = self.class_object_data_body["landmarks"]

        if self.not_empty(boxe_face):

            there_is_shoulder = False
            there_is_body = False

            shoulders, body = self.recuperate_shoulders(body_landmarks)
            there_is_shoulder, there_is_body = [len(i) is 2 for i in [shoulders, body]]

            body_boxe = self.updata_body_contour_in_function_of_data(rgb_frame, boxe_face, there_is_shoulder, there_is_body, shoulders, body)

            if body_boxe is not None:

                a, b, c, d = body_boxe
                rectangle(draw_frame, (a, b), (c, d), (0, 0, 255), 2)

                self.class_object_data_body["contour_body"] = body_boxe
                self.class_object_data_body["landmarks"] = [i[:2] for i in self.class_object_data_body["landmarks"]]


    def recuperate_data_last_arm_movement(self, label, landmarks_body):
        """Recuperate the position of the arms, last position and last time in video.
        self.arm_points = {"left": [12, 14, 16], "right": [11, 13, 15]}
        """

        current_position = [landmarks_body[nb] for nb in self.arm_points[label]]
        last_data = [self.class_object_data_body["last_arm_position"][label][i] for i in ["coordinates", "timer"]]
        last_coordinate, last_timer = last_data

        return current_position, last_coordinate, last_timer


    def arm_informations(self, boxe_face, coordinates, last_coordinates, last_timer):
        """Recuperate last & current arm data for mesure the distance, the speed and the
        direction of the arm."""

        # Get the width face as mesure for the ratio of the distance.
        ratio = self.get_ratio_distance(self.width_face, boxe_face[2])
        distance_scaled = self.scaling_distance_not_round(coordinates, last_coordinates, ratio)
        speed = distance_scaled / (self.timer - last_timer)

        # Recuperate direction (top, left, top left ...) of the arm if the movement is > 2 cm.

        direction_arm = self.define_a_direction(coordinates, last_coordinates) if distance_scaled > 2 else "motionless"

        return speed, distance_scaled, direction_arm


    def update_data(self, speed, distance, direction_arm, label):
        """Update arm data"""

        data_label = {
            "speed": round(speed, 1),
            "distance": round(distance, 1),
            "direction": direction_arm
            }

        for data_name, data in data_label.items():
            self.class_object_data_body["arm_movement"][label][data_name] += [data]


    def arm_movements(self):
        """Recuperate coordinate and timer of the last arm movement
        for mesure the distance, speed and direction + update data."""

        landmarks_body = self.class_object_data_body["landmarks"]
        boxe_face = self.class_object_data_face["face_box"]

        if self.not_empty(landmarks_body):

            for label in self.arm_label:

                # Recuperate the last & the current position of the arm.
                data = self.recuperate_data_last_arm_movement(label, landmarks_body)
                current_position, last_movement, last_timer = data


                if last_movement is not None:
                    # Update data speed, direction and distance.
                    for coordinates, last_coordinates in zip(current_position, last_movement):
                        speed, distance, direction_arm = self.arm_informations(boxe_face, coordinates, last_coordinates, last_timer)
                        self.update_data(speed, distance, direction_arm, label)

                # Replace last data by the new for next speed and direction mesures.
                self.class_object_data_body["last_arm_position"][label]["coordinates"] = current_position
                self.class_object_data_body["last_arm_position"][label]["timer"] = self.timer



    def recuperate_body_landmarks(self, landmark):
        """"""
        recuperate_landmarks =\
        lambda points: [(int(x), int(y)) for (x, y) in [landmark[pts] for pts in points]]
        return [recuperate_landmarks(points) for name_points, points in self.points_dico.items()]


    def make_angle(self, arms_points, boxe_face):
        """ """
        (x1, y1), (x2, y2), (x3, y3) = [arms_points[i] for i in range(3)]

        ratio = self.get_ratio_distance(self.width_face, boxe_face[2]) * 0.0265

        angles = degrees(
            atan2((y3 - y2) * ratio, (x3 - x2) * ratio) -
            atan2((y1 - y2) * ratio, (x1 - x2) * ratio)
        )

        return angles + 360 if angles < 0 else angles


    def hand_in_area(self, frame, hand, point_area, boxe_face, percent):
        """ """
        width_face, height = boxe_face[2:]

        px, py = point_area

        a = px - self.percent_of(percent, width_face)
        b = py - self.percent_of(percent, height)
        c = px + self.percent_of(percent, width_face)
        d = py + self.percent_of(percent, height)

        rectangle(frame, (a, b), (c, d), (0, 255, 255), 1)

        hand_in_epaul = [True if a < x < c and b < y < d else False for (x, y) in hand]

        return self.is_true_or_false_in_list(hand_in_epaul)


    def hand_are_close_on_y(self, wrist_right, wrist_left, taille1, taille2, boxe_face):
        """ """
        ratio = self.get_ratio_distance(self.width_face, boxe_face[2])

        zero = 0
        list_pairs = [
            ( (zero, taille1[0][1]), (zero, taille1[1][1])  ),
            ( (zero, taille2[0][1]), (zero, taille2[1][1])  ),
            ( (zero, taille1[0][1]), (zero, wrist_right[1]) ),
            ( (zero, taille2[0][1]), (zero, wrist_left[1])  )
        ]

        dist_taille_right, dist_taille_left, dist_wrist_elbow1, dist_wrist_elbow2 =\
        [self.scaling_distance_not_round(coord1, coord2, ratio) for (coord1, coord2) in list_pairs]

        good_dist1 = self.percent_of(dist_taille_right, 55) > dist_wrist_elbow1
        good_dist2 = self.percent_of(dist_taille_left, 55) > dist_wrist_elbow2

        return self.is_true_or_false_in_list([good_dist1, good_dist2])


    def recuperate_data_for_arm_sign(self):
        """Recuperate the boxe of the face, body and hand landmarks."""

        face_boxe = self.class_object_data_face["face_box"]
        landmarks_body = self.class_object_data_body["landmarks"]
        landmarks_hand = self.class_object_data_hand["landmarks"]

        return face_boxe, landmarks_body, landmarks_hand


    def track_hand_on_shouder(self, frame, hand, shoulder_right, shoulder_left, face_boxe):
        """Verify hand in shoulder area"""

        hand_in_shoulder_right = self.hand_in_area(frame, hand, shoulder_right, face_boxe, 30)
        hand_in_shoulder_left = self.hand_in_area(frame, hand, shoulder_left, face_boxe, 30)

        return self.is_true_or_false_in_list([hand_in_shoulder_right, hand_in_shoulder_left])

    def track_hand_in_elbow(self, frame, hand, elbow_right, elbow_left, face_boxe):
        """Verify hand in elbow area"""

        hand_in_elbow_right = self.hand_in_area(frame, hand, elbow_right, face_boxe, 40)
        hand_in_elbow_left = self.hand_in_area(frame, hand, elbow_left, face_boxe, 40)

        return self.is_true_or_false_in_list([hand_in_elbow_right, hand_in_elbow_left])


    def definate_movement(self, angle_left, angle_right, is_cross, hand_in_epaul, hand_in_coude, hand_in_poitrine):
        """"""
        dico_arm_gesture = {
            "defensive": [55 <= angle_left <= 80, 270 <= angle_right <= 300, is_cross, hand_in_poitrine],
            "defensive_profil": [15 <= angle_left <= 45, 230 <= angle_right <= 255, is_cross, hand_in_poitrine],
            "tension": [hand_in_epaul],
            "aggressivity": [hand_in_coude, 280 <= angle_right <= 320]
        }

        for sign_name, feature in dico_arm_gesture.items():
            if False not in feature:
                self.class_object_data_body["sign"] += [(sign_name, self.timer)]


    def arm_signs(self, frame):
        """"""
        face_boxe, landmarks_body, landmarks_hand = self.recuperate_data_for_arm_sign()

        if self.not_empty(landmarks_body):

            arm_left, arm_right, taille_right, taille_left = self.recuperate_body_landmarks(landmarks_body)
            arms = [arm_right, arm_left]

            shoulder_right, shoulder_left, elbow_right, elbow_left,\
            wrist_right, wrist_left = [i[n] for n in range(3) for i in arms]

            # Angle
            right_angle, left_angle = [self.make_angle(arm, face_boxe) for arm in arms]

            hand_in_shoulder = False
            hand_in_elbow = False
            for label in self.arm_label:

                hand = [phax for finger in landmarks_hand[label] for phax in finger]

                if self.not_empty(hand):
                    hand_in_shoulder = self.track_hand_on_shouder(frame, hand, shoulder_right, shoulder_left, face_boxe)\
                    if hand_in_shoulder is False else hand_in_shoulder

                    hand_in_elbow = self.track_hand_in_elbow(frame, hand, elbow_right, elbow_left, face_boxe)\
                    if hand_in_elbow is False else hand_in_elbow

            # Hand are cross
            hand_are_cross = True if wrist_left[0] < wrist_right[0] else False
            hand_in_chest = self.hand_are_close_on_y(wrist_right, wrist_left, taille_right, taille_left, face_boxe)
            self.definate_movement(left_angle, right_angle, hand_are_cross,hand_in_shoulder, hand_in_elbow, hand_in_chest)
