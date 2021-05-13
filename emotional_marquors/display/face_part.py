import cv2
import numpy as np
from utils.function_utils import Utils

class Face_display(Utils):
	""" """

	def __init__(self):
		""" """

		self.none_label = [None, "None"]
		self.data_feature_face = ["emotions", "face_direction", "face_showing", "leaning_head", "face_direction_y"]

		# font default
		self.font_color = "white"
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.color_mode = (255, 255, 255)

		self.face_coord = []


	def recuperate_data_of_the_face(self, data_face, timer):
		"""Displaying data of the face in rectangle."""

		data_face_label = np.array([
			"emotions", "face_direction", "face_showing", "face_direction_y"])

		data_feature_face = [
			(str(data_face[label]).capitalize(), label) for label in data_face_label
			if data_face[label] not in self.none_label]

		leaning = data_face["leaning_head"]
		if self.not_empty(leaning):
			last_leaning, last_timer = leaning[-1]
			if timer == last_timer:
				data_feature_face += [(last_leaning, "leaning_head")]

		return data_feature_face


	def rectangle_around_head(self, boxe) -> list:
		""" """
		x, y, w, h = boxe

		width = 50 if w > 200 else 110

		# Slots emplacement around the face.
		x2 = x - self.percent_of(width, w)
		y2 = y - self.percent_of(80, h)
		w2 = x
		h2 = y + h + self.percent_of(80, h)

		x = x2 if x2 > 0 else 20
		y = y2 if y2 > 0 else 20
		w = w2 if w2 < 640 else 620
		h = h2 if h2 < 360 else 340

		return x, y, w, h


	def slot_around_face(self, slot_boxe_face, landmarks_face) -> list:
		""" """
		x, y, w, h = slot_boxe_face

		sloting_face = {

			0: {"coord": (landmarks_face[17], (x, y)),
				"detected": False},
			1: {"coord": (landmarks_face[0], (x, y + self.percent_of(20, h))),
				"detected": False},
			2: {"coord": (landmarks_face[1], (x, y + 10 + self.percent_of(40, h))),
				"detected": False},
			3: {"coord": (landmarks_face[2], (x, y + 10 + self.percent_of(60, h))),
				"detected": False},
			4: {"coord": (landmarks_face[16], (x, y + 10 + self.percent_of(20, h))),
				"detected": False},
			5: {"coord": (landmarks_face[15], (x, y + self.percent_of(40, h))),
				"detected": False},
		}

		return sloting_face


	def add_luminosity_on_text(self, frame, x, y, w, h, gamma1, gamma2):
		""" """

		crop = frame[y:h, x:w]
		gamma_choice = gamma1 if self.color_mode == (255, 255, 255) else gamma2
		crop_gamma = self.adjust_gamma(crop, gamma=gamma_choice)
		frame[y:h, x:w] = crop_gamma


	def drawing_message_on_rectangle(self, frame, coordinates_slots, feature, number_slot):
		"""Message around the head in the rectangle"""

		dico = {"leaning_head": {"left": "Empathy, creativity.", "right": "Analyse, logic."},
			    "face_showing":  {"left": "Analyse, distanciation.", "right": "Communication, empathy."},
				"face_direction_y": {"bot":"Negative emotion", "top": "Positive emotion"}}

		feature, label = feature

		coordinates_begin, coordinates_end = coordinates_slots

		feature = str(feature).upper()
		length_of_message = len(feature)

		# Dimension of the rectangle.
		x1, y1 = coordinates_end
		x2, y2 = (x1 + int(length_of_message * 7.8) + 1, y1 + 9)

		# Ajust luminosity in the rectangle.
		self.add_luminosity_on_text(frame, x1-2, y1-2, x2+2, y2+2, 0.5, 1.4)

		# Rectangle of the rectangle around the text.
		cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), self.color_mode, 1)

		coordinates_end = (x1-2, y2+2) if number_slot in [0, 1] else coordinates_end

		cv2.putText(frame, feature, (x1, y1+9), self.font, 0.4, self.color_mode)

		if label in dico and feature.lower() in dico[label]:
			text = dico[label][feature.lower()]
			cv2.putText(frame, text, (x1, y1+25), self.font, 0.32, self.color_mode)

		cv2.line(frame, coordinates_begin, coordinates_end, self.color_mode, 1)




	def drawing_feature_around_the_head(self, frame, sloting_face, feature_detected):
		""" """
		sloting = []

		for feature in feature_detected:
			for slot in range(6):

				coordinates = sloting_face[slot]["coord"]
				is_occupated = sloting_face[slot]["detected"]

				if is_occupated is False:
					sloting_face[slot]["detected"] = True
					sloting += [coordinates]
					break

		[self.drawing_message_on_rectangle(frame, coord, feature, number_slot)
		 for number_slot, (feature, coord)
		 in enumerate(zip(feature_detected, sloting))]


	def change_color_font(self, color_mode):
		""" """
		self.color_mode = color_mode


	def corner_around_face(self, frame, hand_boxe):


		xHand, yHand, wHand, hHand = hand_boxe

		margin_width = self.percent_of(wHand, 8)
		margin_height = self.percent_of(hHand, 8)

		rectangle = [( (xHand - margin_width), (yHand - margin_height) ),
					 ( (xHand + wHand + margin_width), (yHand + hHand) )]

		x, y, w, h = [points for pairs in rectangle for points in pairs]

		crop = frame[y:h, x:w]
		gamma_choice = 0.7 if self.color_mode == (255, 255, 255) else 1.3
		hand_crop = self.adjust_gamma(crop, gamma=gamma_choice)
		frame[y:h, x:w] = hand_crop


		width_length, height_length = [self.percent_of(length, 20) for length in [wHand, hHand]]

		# Corner left's top #1
		c1x = ( (x, y), (x + width_length, y) )
		c1y = ( (x, y), (x, y + height_length) )

		# Corner right's top #2
		c2x = ( (w, y), (w - width_length, y) )
		c2y = ( (w, y), (w, y + height_length) )

		# Corner left's bot #3
		c3x = ( (x, h), (x + width_length, h) )
		c3y = ( (x, h), (x, h - height_length) )

		# Corner right's bot #4
		c4x = ( (w, h), (w - width_length, h) )
		c4y = ( (w, h), (w, h - height_length) )

		corners = [c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y]

		[cv2.line(frame, coord1, coord2, self.color_mode, 2) for (coord1, coord2) in corners]




	def face_mode_display(self, frame, face_boxe, landmarks_face, data_face, mode, timer):
		""" """

		if face_boxe is not None:

			if mode is "face":
				self.corner_around_face(frame, face_boxe)

			data_feature_face = self.recuperate_data_of_the_face(data_face, timer)

			slot_face_boxe = self.rectangle_around_head(face_boxe)

			sloting_face = self.slot_around_face(slot_face_boxe, landmarks_face)

			self.drawing_feature_around_the_head(frame, sloting_face, data_feature_face)
