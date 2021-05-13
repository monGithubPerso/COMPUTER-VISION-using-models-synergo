"""Les fonctions dans function_utils sont utilisées de nombreuses fois.
Pour éviter des répétitions on fait hériter toutes nos classes par function_utils."""

from scipy.spatial import distance
from cv2 import resize as cv2_resize
import cv2
import numpy as np


class Utils:

    @staticmethod
    def percent_of(length, percent):
        """Percent formula round to floor."""
        return (length * percent) // 100

    @staticmethod
    def percent_of_not_round(length, percent):
        """Percent formula not round."""
        return (length * percent) / 100

    @staticmethod
    def get_ratio_distance(true_mesure, mesure):
        """Recuperate ratio beetween real distance & video distance"""
        return true_mesure / (mesure * 0.0265)

    @staticmethod
    def scaling_distance_round(coord1, coord2, ratio_head_length):
        """Real distance to video distance"""
        return round(distance.euclidean(coord1, coord2) * ratio_head_length * 0.0265, 2)

    @staticmethod
    def scaling_distance_not_round(coord1, coord2, ratio_head_length):
        """Real distance to video distance"""
        return distance.euclidean(coord1, coord2) * ratio_head_length * 0.0265

    @staticmethod
    def get_crop(frame_to_crop, boxe, width_height_already_add=False):
        """Make a crop from a boxe"""
        x, y, w, h = boxe
        return frame_to_crop[y:h, x:w] if width_height_already_add else frame_to_crop[y : y+h, x : x+w]

    @staticmethod
    def not_empty(liste):
        """Verify list isn't empty"""
        return len(liste) > 0

    @staticmethod
    def make_line_in_a_threshold(thresh):
        """For avoid to recuperate the max contour from the border.
        Draw a white border on a threshold picture.
        """

        height, width = thresh.shape[:2]
        zero = 0

        border = np.array([((zero, zero), (zero, height)), ((zero, zero), (width, zero)),
                        ((width, zero), (width, height)), ((zero, height), (width, height))])

        [cv2.line(thresh, tuple(i), tuple(j), (255, 255, 255), 2) for (i, j) in border]
        return thresh

    @staticmethod
    def is_true_or_false_in_list(liste):
        """Put false if there is false in a list of boolean"""
        return False if False in liste else True

    @staticmethod
    def is_true_in_liste_else_false(liste):
        """Put true if there is false in a list of boolean"""
        return True if True in liste else False

    @staticmethod
    def utils_get_resize_frame_and_return_ratio(frame, xResize, yResize):
        """ Hand tracking & Body contours """
        ratio = ( int(frame.shape[1] / xResize), int(frame.shape[0] / yResize) )
        frame = cv2_resize(frame, (xResize, yResize))
        return frame, ratio


    @staticmethod
    def define_a_direction(lasts_coordinates, coordinates):
        """Function for treating lows numbers.
        Definate side of the movement via two coordinates.
        Recuperate the difference beetween the lasts coordinates & the currents coordinates.
        The maximum & the minimum difference can say if the moves his head in diagonal or
        in only one axis. For example 10 / 5 -> 2 so the movement's in diagonal.
        We make difference beetween lasts & currents coordinates 
        for find the direction of the movement.
        """

        xLast, yLast = lasts_coordinates
        x, y = coordinates

        min_difference, max_difference = sorted([abs(x - xLast), abs(y - yLast)])
        min_difference = min_difference if min_difference > 0 else 0.1

        direction = []

        if max_difference / min_difference > 2.5:
            direction = ["left" if x < xLast else "right"]

        else:
            if x != xLast:
                direction += ["left" if x < yLast else "right"]

            if y != yLast:
                direction += ["top" if y < yLast else "bot"]

        if len(direction) == 0:
            direction += ["Immobile"]

        return " ".join(direction)


    @staticmethod
    def utils_point_is_in_boxe(pts, x, y, w, h):
        """Verify a points is beetween two others"""
        return True if x < pts[0] < w and y < pts[1] < h else False

    @staticmethod
    def utils_points_not_out_of_frame(frame, boxe, margin_if_out_frame):
        """Verify boxe isn't out a frame (0 and extremums sides)"""

        height, width = frame.shape[:2]
        x, y, w, h = boxe

        x1, y1, w1, h1 = [value if value >= 0 else 0 for value in [x, y, w, h]]

        is_supp_zero = lambda value, threshold, marge: value if value <= threshold else threshold - marge

        x2 = is_supp_zero(x1, width, 0)
        y2 = is_supp_zero(y1, height, 0)
        w2 = is_supp_zero(w1, width, margin_if_out_frame)
        h2 = is_supp_zero(h1, height, margin_if_out_frame)

        return x2, y2, w2, h2


    @staticmethod
    def mean_list_round(liste):
        """Mean function to replace with numpy mean."""
        return sum(liste) // len(liste) 

    @staticmethod
    def mean_list_not_round(liste):
        """Mean function to delete."""
        return np.mean(liste)

    @staticmethod
    def utils_groupe_timer_by_range(liste, threshold_time):
        """Regroup timer by range for exemple timmers:
        [1.1, 1.2, 1.3, 10, 10.1, 50] 
        -> regroupe them if the difference with a threshold_time = 0.3 -> [(1.1, 1.3), (10, 10.1), (50)]
        -> regroupe them if the difference with a threshold_time = 20  -> [(1.1, 1.3, 10, 10.1), (50)]
        """

        timer_liste = []

        if len(liste) > 0:
            begin = 0
            last = 0

            for timer in liste:
                if begin == 0:
                    begin = timer
                    last = timer

                else:
                    if timer - last > threshold_time:
                        timer_liste += [(begin, last)]
                        begin = timer
                        last = timer
                    else:
                        last = timer
    
            if begin != 0 and last != 0:
                timer_liste += [(begin, last)]


        return timer_liste


    @staticmethod
    def recuperate_landmarks_from_liste(landmarks, liste):
        """Recuperate landmarks with a list of points"""
        return [landmarks[i] for i in liste]


    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        """Change luminosity of the picture"""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    @staticmethod
    def recuperate_all_phax_in_the_hand(hand) -> list():
        """Hand list's cut in finger list, recuperate all points in on list"""
        return [j for i in hand for j in i]

    @staticmethod
    def utils_recuperate_timer(liste):
        return [timer for (sign, timer) in liste]



    @staticmethod
    def utils_group_timers(liste) -> list():
        """Sometimes we can have blinks who's following; regroup them.
        While the length of the list doesn't change, regroupe timer by a range of 0.3 secondes. 
        """

        last = len(liste)
        oContinuer = True
        while oContinuer:

            for n in range(len(liste) - 1):

                begin1, end1 = liste[n]
                begin2, end2 = liste[n + 1]

                if begin2 - end1 < 0.3:
                    liste[n] = (begin1, end2)
                    liste[n + 1] = (begin1, end2)


            liste = sorted(list(set(liste)))

            if len(liste) == last:
                oContinuer = False

            last = len(liste)

        return liste


