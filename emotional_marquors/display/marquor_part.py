import cv2
import numpy as np
from utils.function_utils import Utils


class Marquors_display(Utils):
	"""Displaying of the marquors"""

	def __init__(self, video_path):
		"""Constructor"""

		self.marquor_color2 = (220, 220, 220)
		self.marquor_color1 = (255, 255, 255)

		# font default
		self.font_color = "white"

		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.color_mode = (255, 255, 255)
		self.replay_marquors = {}
		self.video_path = video_path
		self.marquors_dico = {}

		self.dico_category = {
			"marquors_eyes": "Negative emotion, Secure the person.",
			"marquors_eyes_anxiety": "Negative emotion, Secure the person.",
			"marquors_eyes_time": "Person talk about his domain. We can talk about.",
			"marquors_head_bot": "Negative emotion, Secure the person.",
			"marquors_head_top": "Positive emotion",
            "marquors_face": "Person talk about his domain. We can talk about.",
			"marquors_forehead": "Negative emotion, Secure the person.",
			"mouth_marquors_hide": "Person hide something.",
			"mouth_marquors_honey": "Person wants to be kind",
            "marquor_hand": "Confidence",
            "face_movement": "face_movement",
			"face_facing": "Person has scared.",
            "emotion_fearful": "Person has scared.",
			"emotion_angry": "Person associate angry with the moment.",
            "emotion_disgusted": "Person associate disgusted with the moment.",
			"emotion_sad": "Person associate sad with the moment",
			"hand_sign_marquor": "Hand sign"
		}


	def lunch_a_replay_on_click(self, x, y):
		"""On click lunch a marquor."""
		for _, data in self.marquors_dico.items():

			xBoxe, yBoxe, wBoxe, hBoxe = data["rectangle"]
			signification = data["signification"]

			if xBoxe <= x <= wBoxe and yBoxe <= y <= hBoxe:

				timer_of_the_replay = data["time"]
				begin_of_the_replay, end_of_the_replay = timer_of_the_replay

				cap = cv2.VideoCapture(self.video_path)
				cap.set(cv2.CAP_PROP_POS_MSEC, begin_of_the_replay * 1000)

				while cap.isOpened():

					_, frame = cap.read()

					timer = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)

					frame = cv2.flip(cv2.resize(frame, (640, 360)), 1)
					cv2.putText(frame, "REPLAY", (10, 20), self.font, 0.4, self.color_mode)
					cv2.putText(frame, signification, (10, 50), self.font, 0.35, self.color_mode)

					cv2.imshow("frame", frame)
					if (cv2.waitKey(50) & 0xFF == ord('q')) or timer >= end_of_the_replay:
						break


	def recuperate_marquors_emplacement_for_click_on_it(self, frame, index_marquor, marquor_plage, signification):
		"""Get in a dico, emplacements of the marquors."""

		self.marquors_dico[index_marquor] = {
			"rectangle" : [
				self.start_x_colmun - 5, (30 + self.space_beetween_line) - 5,
				self.start_x_colmun + 100, 30 + self.space_beetween_line + 5],
			"time": marquor_plage,
			"signification": signification
		}

	def change_color_font(self, color_mode):
		"""Recuperate for the script of displaying the color."""
		self.color_mode = color_mode

	@staticmethod
	def recuperate_sorted_marquors(marquors):
		"""Sorted in function of the time marquors."""
		liste = []
		for k, v in marquors.items():
			for i in v:
				liste += [(i, k)]

		liste = sorted(liste)
		return liste


	def define_range_marquor_display(self, begin, end, category):
		"""Reorganize marquors (we can add some second on the marquors)."""

		self.dico_duration = {
			"marquors_eyes": [begin, end],
			"marquors_eyes_anxiety":[begin, end],
			"marquors_eyes_time": [begin, end], 
			"marquors_head_bot": [begin, end],
			"marquors_head_top": [begin, end ],
            "marquors_face": [begin, end], 
			"marquors_forehead": [begin, end],
            "marquor_hand": [begin, end], 	
            "face_movement": [begin, end], 
			"face_facing":[begin, end], 
            "emotion_fearful": [begin, end], 
			"emotion_angry": [begin, end],
            "emotion_disgusted": [begin, end], 
			"emotion_sad": [begin, end],
			"mouth_marquors_hide": [begin, end],
			"mouth_marquors_honey": [begin, end],
			"hand_sign_marquor": [begin, end],
		}

		begin, end = self.dico_duration[category]

		return round(begin, 2), round(end, 2)


	def placing_marquors(self, frame, marquors):
		"""Can place tilte when the marquor'll display."""

		self.marquors_dico = {}

		self.start_x_colmun = 10
		self.column_index = 0
		self.space_beetween_line = 5


		liste = self.recuperate_sorted_marquors(marquors)

		for index_marquor, ((beginO, endO), category) in enumerate(liste):

			if self.space_beetween_line > 300: # new column
				self.start_x_colmun += 120
				self.space_beetween_line = 5
				self.column_index += 1

			(begin, end) = self.define_range_marquor_display(beginO, endO, category)

			self.recuperate_marquors_emplacement_for_click_on_it(frame, index_marquor, (begin, end), self.dico_category[category])

			text = f"Marquor {beginO}:{endO}".capitalize()
			cordinates_text = (self.start_x_colmun + 5, self.space_beetween_line + 35)
			color_text = self.marquor_color1 if self.column_index % 2 == 0 else self.marquor_color2

			cv2.putText(frame, text, cordinates_text, self.font, 0.3, color_text)

			self.space_beetween_line += 15

