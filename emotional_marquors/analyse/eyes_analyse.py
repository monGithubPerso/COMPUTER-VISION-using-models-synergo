#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""Analyse de la fermeture des yeux.
Récupération des clignements des yeux et de la moyenne de la fermeture entre chaque clignement.
Récupération des clignements inférieurs à 60% de la moyenne. Fusion de clignements qui sont
proche."""

import numpy as np
from utils.function_utils import Utils


class Eyes_analyse(Utils):
    """Here we get the list of moments in the video where the person blinks.
    We recover the difference between 2 flashes and the average between each flash.
    We then recover all blinks that are 60% above the average and regroup them.
    We also recover the blinks that last more than 0.3 ms (because the human average is 0.15 - 0.3)

    closing at 0.5 & 0.7 -> emotion
    closing at 0.5 & 0.7 & 0.9 -> anxiety
    closing for 0.5 to 0.8 -> person in his buble.

    """

    def __init__(self):
        """ """

        self.class_object_data_analyse = {}
        self.class_object_data_eyes = {}
        self.timer = 0

        # In the temporary list we define frequency of the closing in function of the mean
        # by signs.
        self.closing_rpz = {">": "period of anxiety", "<": "period of concentration", "==": "period of calm"}

        # Treshold (in function of the mean of blink) beetween two closings for be considerate as a marquor.
        self.min_threshold_blink = 60



    def getter_data_eyes(self, data_eyes):
        """Copy data eyes from the database."""
        self.class_object_data_eyes = data_eyes

    def getter_data_analyse(self, data_analyse):
        """Copy data analyse from the database."""
        self.class_object_data_analyse = data_analyse

    def getter_timer(self, timer):
        """Recovery timer in the video."""
        self.timer = timer

    def raise_data(self):
        """Raise all data copied."""
        self.class_object_data_analyse = {}
        self.class_object_data_eyes = {}


    def min_closing_from_mean(self, closing_historic) -> list():
        """Recuperate the mean beetween two closings eyes. 
        Recuperate Difference beetween each blink & minimals closings time in function of the mean.

        * self.min_threshold_blink = 60
        """

        # Mean of the difference beetween each closing eyes.
        mean_difference_blink = np.mean([closing_historic[t + 1] - closing_historic[t] 
                                        for t in range(len(closing_historic) - 1)])

        # Treshold closests blinks.
        threshold_quick_close = self.percent_of_not_round(mean_difference_blink, self.min_threshold_blink)

        # Recuperate threshold beetween two closing in function of the mean of the closing.
        min_intervals = [(closing_historic[t], closing_historic[t + 1])
                         for t in range(len(closing_historic) - 1)
                         if closing_historic[t + 1] - closing_historic[t] <= threshold_quick_close]

        return min_intervals


    def closing_analyse(self):
        """Recovery of eye blinks and the average closure between each blink.
        Recovery of blinks below a threshold of the mean."""

        closing_historic = self.class_object_data_eyes["closing_historic"] # Blinking data, temporary list.

        # Blinks are under form: [0.12 0.13 0.17 0.29]. Group: [0.12 0.13 0.17] [0.29]
        closing_historic = self.utils_groupe_timer_by_range(closing_historic, threshold_time=0.4)

        # Recuperate the first element: [0.1 0.13 0.17] & [0.9] -> [0.1, 0.9]
        closing_historic = [begin for (begin, end) in closing_historic]

        if self.not_empty(closing_historic):

            # Recuperate blink duration < 60 % mean.
            min_closing_interval = self.min_closing_from_mean(closing_historic)

            # Regroup blink by interval
            blink_interval = self.utils_group_timers(min_closing_interval)

            # Blink number in a plage who's under one seconds
            blink_less_one = [(begin, end) for (begin, end) in blink_interval if end - begin < 1]
            # Blink number in a period above ine seconds.
            blink_period = [(begin, end) for (begin, end) in blink_interval if end - begin > 1]

            self.class_object_data_analyse["marquors"]["marquors_eyes_anxiety"] = blink_period
            self.class_object_data_analyse["marquors"]["marquors_eyes"] = blink_less_one


    def eyes_closing_significate(self):
        """Recuperate and stock the process period in function of eyes blinking."""

        # Recuperate data from a temporary list.
        closing = self.class_object_data_eyes["frequency_closing"]["from_mean"]
        definate_period = self.closing_rpz[closing] if closing in self.closing_rpz else ""
        self.class_object_data_analyse["closing_eye_definate"] += [(definate_period, self.timer)]


    def eyes_closing_time_significate(self):
        """Put a marquor if there is a duration of a blik beetween 0.3 and 0.45 ms. """

        # Blinking data.
        closing_historic_duration = self.class_object_data_eyes["closing_historic"]
        closing_historic_duration = self.utils_groupe_timer_by_range(closing_historic_duration, 0.4)

        liste_marquors = [(d1, d2) for (d1, d2) in closing_historic_duration if 0.45 >= d2 - d1 > 0.3]
        self.class_object_data_analyse["marquors"]["marquors_eyes_time"] = liste_marquors


