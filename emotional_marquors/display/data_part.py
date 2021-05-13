import cv2
import numpy as np


class Data_to_display:
    """Recuperate in the database data for displaying it.
    In the data the most of the time we savegarde timer. So we need to
    treat data."""

    def __init__(self):
        """Constructor"""

        # Dico mood's definate the mood of the person
        # in function of the numbers of the blink. Symbole to words.
        self.dico_mood = {
            "==": "In the mean", "<": "Inf. the mean",
            ">": "Sup. the mean"
        }

    @staticmethod
    def recuperate_gender_data(data_face):
        """From the gender model, we want at less 10 
        features compose by 1 if it's a female else 0"""

        gender = "In course."
        data_gender = data_face["gender"]
 
        if len(data_gender) >= 10:
            gender = "Female" if data_gender.count(1) > data_gender.count(0) else "Male"

        return gender


    @staticmethod
    def recuperate_mood_data(data_analyse):
        """Recuperate mood of the person"""
        mood = data_analyse["closing_eye_definate"]
        mood_perdiod = mood[-1][0] if len(mood) is not 0 else None
        return mood_perdiod if mood_perdiod not in ("", None) else "Calm"


    def recuperate_closing_eyes_freq_data(self, data_eyes):
        """Recuperate closing frequency"""
        frequency = data_eyes["frequency_closing"]["from_mean"]
        return self.dico_mood[frequency]


    @staticmethod
    def recuperate_hand_distance_data(data_hand, label_hand):
        """Recuperate hand distance."""

        data_interet = ["distance", "direction of the hand"]

        to_string = lambda string: str(string).capitalize()

        speed_hand = data_hand["speed"][label_hand]
        speed_hand = to_string(speed_hand[1]) if speed_hand is not None else "None"
        distance_hand, direction_hand = [to_string(data_hand[data][label_hand]) for data in data_interet]

        return speed_hand, distance_hand, direction_hand


    @staticmethod
    def recuperate_head_movement_data(data_analyse, timer):
        """Recuperate head movement"""

        face_list = ["face_direction_y_definate", "face_facing"]

        last_data = lambda liste: liste[-1][0]\
                    if len(liste) > 0 and liste[-1][-1] is timer and\
                    liste[-1][0] is not "None"\
                    else "No process."

        return [last_data(i) for i in [data_analyse[label] for label in face_list]]


    @staticmethod
    def recuperate_leaning(data_analyse, timer):
        """Recuperate head leaning"""

        last_process = "No process."

        data = data_analyse["head_leaning_definate"]

        if len(data) is not 0:
            last_process_in_database, _, last_detection = data[-1]

            if timer - last_detection < 1:
                last_process = "Has detected " + last_process_in_database

        return last_process


    @staticmethod
    def recuperate_other_person_around(data_body) -> dict():
        """Recuperate other person around the person."""

        other = ["Nobody", "Nobody", "Nobody"]

        in_space = data_body["in_social_area"]
        if in_space is not None:
            other = [i[0] if len(i) is not 0 else "Nobody" for nb, i in enumerate(in_space)]
 
        return other


    @staticmethod
    def body_sign_detected(data_analyse, timer):
        """Recuperate body sign detection"""

        body_sign = "No signs"
        body_sign_data = data_analyse["body_sign"]

        if len(body_sign_data) is not 0:
            sign, last_timer = body_sign_data[-1]
            
            if last_timer is timer:
                body_sign = sign

        return body_sign


    @staticmethod
    def recuperate_skin_color(data_face):
        """Recuperate skin color"""
        return data_face["skin_color"].capitalize()\
            if data_face["skin_color"] is not None else ""


    @staticmethod
    def recuperate_color_of_the_body(data_body):
        """Recuperate color of the body"""
        return str(data_body["color"]).capitalize()


    @staticmethod
    def recuperate_etat_eye(data_eyes):
        """Recuperate stat of the eye"""
        return "Open" if data_eyes["open"] else "Close"


    @staticmethod
    def recuperate_closing_frequency(data_eyes):
        """Recuperate closing frequency"""
        return data_eyes["frequency_closing"]["by_min"]


    def recuperate_face_data_interest(self, data_face, face_id, data_displaying_around_body_face):
        """Recuperate skin & gender data"""
        skin = self.recuperate_skin_color(data_face)
        gender = self.recuperate_gender_data(data_face)

        face_data = [f"Id: {face_id}", skin, gender]
        data_displaying_around_body_face["face"] = face_data


    def recuperate_eyes_data_interest(self, data_eyes, data_info_in_side_of_frame):
        """Recuperate eye data."""
        open_eye = self.recuperate_etat_eye(data_eyes)
        from_mean = self.recuperate_closing_eyes_freq_data(data_eyes)
        frequency_closing = self.recuperate_closing_frequency(data_eyes)

        eyes_data = [open_eye, from_mean, f"{frequency_closing}/Min."]
        data_info_in_side_of_frame["Eyes informations"] = eyes_data



    def recuperate_analyse_data_interest(self, data_analyse, data_info_in_side_of_frame, timer):
        """Recuperate head movements."""

        y_movement, face_facing = self.recuperate_head_movement_data(data_analyse, timer)
        leaning = self.recuperate_leaning(data_analyse, timer)
        body_sign = self.body_sign_detected(data_analyse, timer)

        head_data = [y_movement, face_facing, leaning]
        body_data = [body_sign]

        data_info_in_side_of_frame["Analyse of the head"] = head_data
        data_info_in_side_of_frame["Signs of body"] = body_data



    def recuperate_body_data_interest(self, data_body, data_analyse, data_displaying_around_body_face, data_info_in_side_of_frame):
        """Recuperate body data"""
   
        color_of_the_body = self.recuperate_color_of_the_body(data_body)
        space = self.recuperate_other_person_around(data_body)
        mood_from_eyes_closing = self.recuperate_mood_data(data_analyse)

        body_data = [color_of_the_body, mood_from_eyes_closing]
        space_data = [f"50 cm: {str(space[0])}", f"120 cm: {str(space[1])}", f"300 cm: {str(space[2])}"]

        data_info_in_side_of_frame["Social space around body"] = space_data
        data_displaying_around_body_face["body"] = body_data



    def recuperate_hand_data_intrest(self, data_hand, data_info_in_side_of_frame):
        """Recuperate the speed, the distance & the direciton of the hands movements"""

        #Dont push none in the display. Place 0 at the place.
        data_if_not_none_else_0 = lambda data : str(data) if data not in (None, "none", "None") else "0"

        speed1, dist1, direction1 = self.recuperate_hand_distance_data(data_hand, "right")
        speed2, dist2, direction2 = self.recuperate_hand_distance_data(data_hand, "left")

        hands_data = ["Right hand", f'{data_if_not_none_else_0(dist1)} Cm', 
                    f"{data_if_not_none_else_0(speed1)} cm/seconde", direction1,
                    "", "Left hand", f'{data_if_not_none_else_0(dist2)} Cm', 
                    f"{data_if_not_none_else_0(speed2)} cm/seconde", direction2]

        data_info_in_side_of_frame["Hands movement"] = hands_data


    def recuperate_data_interest(self, database, face_id, timer):
        """Recuperate data to displaying"""

        data_face, data_body, data_eye, data_hand, data_analyse = database

        in_course = None
        data_displaying_around_body_face = {
            "face": in_course, 
            "body": in_course
        }


        data_info_in_side_of_frame = { 
            "Eyes informations": in_course,
            "Analyse of the head": in_course,
            "Signs of body": in_course,
            "Social space around body": in_course,
            "Hands movement": in_course,
        }


        # Recuperate data & stock it into the dictionaries.
        self.recuperate_face_data_interest(data_face, face_id, data_displaying_around_body_face)
        self.recuperate_eyes_data_interest(data_eye, data_info_in_side_of_frame)
        self.recuperate_analyse_data_interest(data_analyse, data_info_in_side_of_frame, timer)
        self.recuperate_body_data_interest(data_body, data_analyse, data_displaying_around_body_face, data_info_in_side_of_frame)
        self.recuperate_hand_data_intrest(data_hand, data_info_in_side_of_frame)

        return data_displaying_around_body_face, data_info_in_side_of_frame

