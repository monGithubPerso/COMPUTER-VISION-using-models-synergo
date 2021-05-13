import cv2
import numpy as np
from utils.function_utils import Utils

class Info_display(Utils):
    """Displaying data around the frame and beside the body. 
    Here we recuperate emplacement of each rectangle to display
    in function of the face & body emplacement."""

    def __init__(self):
        """Constructor"""

        self.font = cv2.FONT_HERSHEY_SIMPLEX # Font default.
        self.color_mode = (255, 255, 255) # Color rgb default.
        self.font_size = 0.3 # Font size default.

        # Differents emplacements for the display of the features.
        # In function of the localisation of the face.
        self.boxe_center = {
            "left_top": (20, 20),
            "left_mid": (20, 80),
            "left_bot": (20, 160),
            "left_bot2" : (20, 200),
            "right_mid" : (500, 200),
            }

        self.boxe_right = {
            "right_top" : (500, 20),
            "right_mid" : (500, 80),
            "right_bot" : (500, 160),
            "right_bot2": (500, 200),
            "left_bot": (20, 200),
            }

        self.boxe_left = {
            "left_top": (20, 20),
            "left_mid": (20, 80),
            "left_bot": (20, 160),
            "left_bot2" : (20, 200),
            "right_mid" : (500, 200),
            }


        # Choice of the boxe in function of the nose localisation.
        self.boxe_config = None


    def place_info_data_in_the_frame(self, boxe_body, boxe_face):
        """Create boxe around the head and inside the body in function of the face & body dimensions"""

        xBody, yBody, wBody, hBody = boxe_body
        xFace, yFace, wFace, hFace = boxe_face
        width_face = xFace + wFace

        boxe_display = {
            "side_face": [
                (width_face + 2, yFace - 15),
                (width_face + 2, yFace - 25),
                (width_face + 2, yFace - 35)
            ],
            "side_body": [
                (wBody + self.percent_of(15, wBody) + 2, yBody + 20),
                (wBody + self.percent_of(15, wBody) + 2, yBody + 30)
            ],
        }

        return boxe_display


    def change_color_font(self, mode) -> str():
        """Recuperate font color (black or white) to display in the display main function."""
        self.color_mode = mode


    def choice_emplacement_avaible(self, face_boxe, frame):
        """Choice the emplacement of display boxe of features in function of the face
        localisation."""

        x, y, w, h = face_boxe
        height, width = frame.shape[:2]
        boxe = None

        # Face at left or width of the face > 120 px.
        if x < width / 2 or w > width / 2:
            boxe = self.boxe_right
            self.boxe_config = "right"

        # Face at right.
        elif x > 420:
            boxe = self.boxe_left
            self.boxe_config = "left"

        # Face at center.
        else:
            boxe = self.boxe_center
            self.boxe_config = "left"

        return boxe


    def create_crop_with_length_of_text(self, frame, emplacement, data, name_emplacement):
        """Make a boxe (rectangle) below features in function of the length of the
        title and of the numbers of features.
        Ajust gamma (darkest or lightest) on the emplacements (rectangle) 
        of the display (data in side of frame)."""

        # Deaparture of the rectangle.
        x, y = emplacement

        # left side has a max length of 24
        # right side hax a max length of 19
        the_maximum_length_of_a_chaine = 24 if "left" in name_emplacement else 19
        width = int(6.5 * the_maximum_length_of_a_chaine) # Max length a chain.

        # Number of data * 10 (height of a font at 0.3). 
        height = ( (len(data) + 1) * 10 ) + 8

        #cv2.rectangle(frame, (x, y), (x+width, y+height), (200, 200, 200), 1)

        # Area of informations
        crop = frame[y:y+height, x:x+width]

        # Ajust luminosity in function of the color.
        gamma_choice = 0.6 if self.color_mode == (255, 255, 255) else 1.4
        crop_gamma = self.adjust_gamma(crop, gamma=gamma_choice)

        # Past the area (darkest - lightest) in the frame.
        frame[y:y+height, x:x+width] = crop_gamma


    def display_information_data_side_frame(self, frame, data_info_side_frame, face_boxe, landmarks_face, landmarks_body):
        """Display data in a rectangle around the frame."""

        boxe_data = self.choice_emplacement_avaible(face_boxe, frame)

        # Run emplacements where displaying & data to display.
        for (name_emplacement, emplacement), (name_data, data) in zip(
            boxe_data.items(), data_info_side_frame.items()):

            (x, y) = emplacement

            # Ajust luminosity on the emplacement.
            self.create_crop_with_length_of_text(frame, emplacement, data, name_emplacement)

            # Put title of the features.
            cv2.putText(frame, name_data, (x + 10, y + 10), self.font, self.font_size, self.color_mode)

            # Put features.
            [cv2.putText(frame, str(i), (x + 10, y + 20 + (nb * 10)), self.font, self.font_size, self.color_mode)
             for nb, i in enumerate(data)]

            # Put line before features text.
            cv2.line(frame, (x + 5, y + 2), (x + 5, y + ((len(data) + 1) * 10) + 2), self.color_mode, 1)



    def drawing_line_display_around_body(self, frame, face_display, body_display):
        """Drawing line width data displayed."""

        cv2.line(frame, (face_display[0][0]  - 2, face_display[0][1]), 
        (face_display[-1][0] - 2, face_display[-1][1] - 8), self.color_mode, 1)

        cv2.line(frame, (body_display[0][0]  - 2, body_display[-1][1]),
         (body_display[-1][0] - 2, body_display[-1][1] - 18), self.color_mode, 1)


    def display_information_data_side_body(self, frame, data_info_around_face, face_boxe, body_boxe):
        """Displaying all rectangle along the side of the frame.
        Displaying feature beside the head and the right shoulder."""

        # Recuperate emplacements of features in function of the body & the face localisation.
        boxe_display = self.place_info_data_in_the_frame(body_boxe, face_boxe)

        # Recuperate emplacements.
        face_display, body_display = [boxe_display[i] for i in ["side_face", "side_body"]]
        data_face, data_body = [data_info_around_face[i] for i in ["face", "body"]]

        # Displaying text.
        puting_data = lambda data, slots:\
            [cv2.putText(frame, i, emplacement, self.font, self.font_size, self.color_mode)
            for i, emplacement in zip(data, slots)]

        puting_data(data_face, face_display)
        puting_data(data_body, body_display)

        # Displaying the white line before data.
        self.drawing_line_display_around_body(frame, face_display, body_display)
