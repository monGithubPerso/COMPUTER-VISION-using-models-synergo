import cv2
import numpy as np
from utils.function_utils import Utils

class Feature_display(Utils):
    """Features displaying (skeletton of the hands and of the body and the face.)"""

    def __init__(self):
        """ """

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_color = "white"
        self.color_mode = (255, 255, 255)

        self.line_points_one_coordinate = np.array([
            (5, 3), (11, 13), (31, 33), (35, 33), (19, 24), (31, 28), (35, 28), (5, 57),
            (11, 57), (5, 8), (11, 8), (8, 57), (24, 16), (19, 0), (13, 16), (3, 0), (51, 33), 
            (42, 33)
        ])

        self.line_points_one_coord_two_coord = [
            (3,  (48, 50)), (13, (54, 52)), (5,  (48, 50)), (11, (54, 52)), 
            (57, (8,  50)), (57, (54, 52)), (51, (48, 50)), (51, (54, 52)), 
            (31, (48, 50)), (35, (54, 52)), (28, (27, 19)), (19, (39, 40)),
            (19, (36, 41)), (24, (45, 46)), (24, (42, 47)), (3,  (36, 41)),
            (13, (45, 46)), (28, (36, 41)), (28, (45, 46)), (33, (36, 41)),
            (33, (45, 46)), (33, (39, 40)), (33, (45, 46)), (33, (45, 46)),
            (33, (36, 41)), (57, (48, 50)),
        ]

        self.line_points_modify_four_coord = np.array([
            ((27, 19), (39, 40)), ((27, 19), (42, 47)),
        ])

        self.circle_points_one_coord = np.array([
            33, 35, 31,51, 3, 13, 24, 19, 8, 11, 5, 19, 0, 16, 28,
        ])

        self.circle_two_coord = np.array([
            (45, 46), (36, 41),(42, 47), (39, 40),(48, 50),(54, 52),(27, 19),
        ])


    def make_rectangle(self, frame, landmarks, w):
        """ """

        add = (w * 2) // 100

        x = (landmarks[0] - add, landmarks[1] - add)
        y = (landmarks[0] + add, landmarks[1] + add)

        cv2.rectangle(frame, x, y, self.color_mode, 1)


    def drawing_line_face(self, frame, landmarks, face_boxe):
        """ """

        [cv2.line(frame, landmarks[coord1], landmarks[coord2], self.color_mode, 1)
        for coord1, coord2 in self.line_points_one_coordinate]

        [cv2.line(frame, landmarks[c1], (landmarks[c2x][0], landmarks[c2y][1]), self.color_mode, 1)
        for c1, (c2x, c2y) in self.line_points_one_coord_two_coord]

        [cv2.line(frame, (landmarks[c1x][0], landmarks[c1y][1]), 
        (landmarks[c2x][0], landmarks[c2y][1]), self.color_mode, 1)
        for (c1x, c1y), (c2x, c2y) in self.line_points_modify_four_coord]

        width_face = face_boxe[2]

        [self.make_rectangle(frame, landmarks[intersection], width_face) for intersection in self.circle_points_one_coord]
        [self.make_rectangle(frame, (landmarks[x][0], landmarks[y][1]), width_face) for (x, y) in self.circle_two_coord]



    @staticmethod
    def recuperate_double_arrow_display_face(landmarks):
        """ Beside face, double arrow draw : '<->' """

        twenty = 20

        return [( (landmarks[0][0] - twenty, landmarks[0][1]), (landmarks[0][0] - twenty, landmarks[9][1]) ),
                ( (landmarks[0][0] - twenty, landmarks[9][1]), (landmarks[0][0] - twenty, landmarks[0][1]) ),
                ( (landmarks[0][0], landmarks[24][1] - twenty), (landmarks[16][0], landmarks[24][1] - twenty) ),
                ( (landmarks[16][0], landmarks[24][1] - twenty), (landmarks[0][0], landmarks[24][1] - twenty) )]


    def drawing_face_dimension(self, frame, landmarks_face):
        """ """

        arrow = self.recuperate_double_arrow_display_face(landmarks_face)

        [cv2.arrowedLine(frame, coord1, coord2, self.color_mode, 1) for (coord1, coord2) in arrow]

        coordinates = (landmarks_face[0][0] - 60, (landmarks_face[9][1] + landmarks_face[0][1]) // 2)
        cv2.putText(frame, "~19 cm", coordinates, self.font, 0.3, self.color_mode)

        text_width_face_x = landmarks_face[0][0] + ( (landmarks_face[16][0] - landmarks_face[0][0]) // 2 )
        text_width_face_x = int(text_width_face_x - (2.5 * 8))
        text_width_face_y = landmarks_face[24][1] - 30

        cv2.putText(frame, "~15 cm", (text_width_face_x, text_width_face_y), self.font, 0.3, self.color_mode)



    def drawing_face_feature(self, frame, data_face):
        """Drawing direction, movement and speed beetween 
        the last & the current coodinates of the face"""

        face_boxe = data_face["face_box"]
        info_face = data_face["face_coordinate_historic"]
        speed = data_face["face_movement_speed"]

        if face_boxe is not None and self.not_empty(speed):
            x = face_boxe[0] + face_boxe[2] + 20
            speed = round(speed[-1][0], 1)
            text = str(speed) + " cm/s"
            cv2.putText(frame, text, (x, face_boxe[1]), self.font, 0.3, self.color_mode)

            if self.not_empty(info_face):
                info_face = info_face[-1]




    def drawing_line_of_arms(self, frame, line_of_body):
        """ """

        [[cv2.line(frame, part_of_body[i], part_of_body[i + 1], self.color_mode, 1)
          for i in range(len(part_of_body) - 1)] for part_of_body in line_of_body]


    def drawing_features_of_arms(self, frame, part_of_body, data_body, label, boxe_face):
        """ """

        w, h = boxe_face[-2:]
        percenof = lambda percent, mesure: (percent * mesure) // 100

        for nb, (x, y) in enumerate(part_of_body[:2]):

            data_distance = data_body["arm_movement"][label]["distance"]

            if len(data_distance) is not 0:

                data_to_display = [
                    str(round(data_body["arm_movement"][label]["distance"][nb], 1)) + " cm",
                    str(int(data_body["arm_movement"][label]["speed"][nb])) + " cm/s",
                    str(data_body["arm_movement"][label]["direction"][nb]).capitalize()]

                if label is "right":
                    [cv2.putText(frame, text, (x + percenof(25, w), y + (n * 10)), self.font, 0.25, self.color_mode)
                    for n, text in enumerate(data_to_display)]
                else:
                    [cv2.putText(frame, text, (x - percenof(25, w), y + (n * 10)), self.font, 0.25, self.color_mode)
                    for n, text in enumerate(data_to_display)]


    def drawing_arm(self, frame, data_body, boxe_face):
        """ """

        landmarks_body = data_body["landmarks"]

        arm_left = [landmarks_body[i] for i in [12, 14, 16]]
        arm_right = [landmarks_body[i] for i in [11, 13, 15]]
        body = [landmarks_body[i] for i in [12, 24, 23, 11, 12]]

        self.drawing_line_of_arms(frame, [arm_left, arm_right, body])

        self.drawing_features_of_arms(frame, arm_right, data_body, "right", boxe_face)
        self.drawing_features_of_arms(frame, arm_left, data_body, "left", boxe_face)
         


    def drawing_hand_in_mode_feature(self, frame, hand):
        """ """

        [cv2.line(frame, finger[i], finger[i + 1], self.color_mode, 1)
         for finger in hand
         for i in range(0, len(finger) - 1)]


    def drawing_hand(self, frame, data_hand):
        """ """

        for label in ["right", "left"]:

            boxe = data_hand["boxe"][label]

            if boxe is not None:

                hand = data_hand["landmarks"][label]
                self.drawing_hand_in_mode_feature(frame, hand)

                x, y, w, h = boxe

                speed = data_hand["speed"][label]

                data_hand_display = [
                    str(speed[1] if speed is not None else "0") + " cm/s",
                    str(data_hand["distance"][label]) + " cm",
                    data_hand["direction of the hand"][label]]

                [cv2.putText(frame, text, (x+w, y + (nb * 10)), self.font, 0.25, self.color_mode)
                 for nb, text in enumerate(data_hand_display)]


    def change_color_font(self, color_mode):
        """ """
        self.color_mode = color_mode


    def features_mode(self, frame, data_body, data_hand, data_face, boxe_face):
        """ """

        landmarks_face = data_face["face_landmarks"]

        self.drawing_line_face(frame, landmarks_face, boxe_face)
        self.drawing_face_dimension(frame, landmarks_face)
        self.drawing_face_feature(frame, data_face)
        self.drawing_hand(frame, data_hand)

        if self.not_empty(data_body["landmarks"]):
            self.drawing_arm(frame, data_body, boxe_face)