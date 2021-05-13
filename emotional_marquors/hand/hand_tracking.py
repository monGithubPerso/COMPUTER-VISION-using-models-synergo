
import numpy as np
from cv2 import boundingRect as cv2_boundingRect
from cv2 import rectangle as cv2_rectangle
from scipy.spatial import distance
import mediapipe as mp
from utils.function_utils import Utils
import cv2


class Hand_tracking(Utils):
	"""Hand detection, hand tracking. At each frame
	define the hand (right or left, hand of who ?)."""

	def __init__(self):

		self.mp_hands = mp.solutions.hands

		self.min_detection_hand = 0.5
		self.max_num_hands_to_search = 2
		self.min_tracking_hands = 0.5
		self.static_image_mode = False

		self.hands_searching = self.mp_hands.Hands(
			self.min_detection_hand,
			self.max_num_hands_to_search,
			self.min_tracking_hands,
			self.static_image_mode,
		)
		
		self.mp_drawing = mp.solutions.drawing_utils

		self.RAYON_THREHSOLD = None

		self.hand_data = ["hands", "has_been_detected", "boxe", 
						"direction of the hand", "faceHand", "speed", "distance"]

		self.hands_labels = np.array(["right", "left"])
		self.maximum_distance_wrinkle_hand = 50

		self.class_object_data_body = {}
		self.class_object_data_face = {}
		self.class_object_data_hand = {}


	def getter_data_hand(self, data_hand):
		"""Get hand data"""
		self.class_object_data_hand = data_hand

	def getter_data_face(self, data_face):
		"""Get face data"""
		self.class_object_data_face = data_face

	def getter_data_body(self, data_body):
		"""Get body data"""
		self.class_object_data_body = data_body

	def raise_data(self):
		"""Raise data"""
		self.class_object_data_face = {}
		self.class_object_data_hand = {}
		self.class_object_data_body = {}


	def remove_hand_out_frame(self, frame, hand_detected):
		"""Hand can be predict out of the frame. Remove them."""

		hand_filter = []

		height, width = frame.shape[:2]

		for hand in hand_detected:
			hand_to_finger = [phax for finger in hand for phax in finger]
			counter_out = sum([1 for (x, y) in hand_to_finger if x > width or y > height or x < 0 or y < 0])
			if counter_out < 10:
				hand_filter += [hand]

		return hand_filter


	def hands_detection(self, frame):
		"""Hands detection."""

		# Search hands in a resize frame; 
		#recuperate ratio to put landmarks at the right place in the none resize frame.

		frame_resize, (a, b) = self.utils_get_resize_frame_and_return_ratio(
			frame, xResize=400, yResize=225)

		frame.flags.writeable = False

		hands_found = self.hands_searching.process(frame_resize)

		hand_scaled = []

		if hands_found.multi_hand_landmarks:

			# Hand landmarks
			hand_mediapipe = [(round(i.x * frame.shape[1] * a), round(i.y * frame.shape[0] * b))
							  for points in hands_found.multi_hand_landmarks
							  for i in np.array(points.landmark)]

			# Cut all landmarks to 21 points.
			hand_mediapipe = [hand_mediapipe[i: i+21] for i in range(0, len(hand_mediapipe), 21)]

			# Group them by fingers.
			hand_scaled = [[hand[:5]] + [hand[i: i+4] for i in range(5, len(hand) - 1, 4)]
						   for hand in hand_mediapipe]

		hand_scaled = self.remove_hand_out_frame(frame, hand_scaled)
		return hand_scaled



	@staticmethod
	def updata_data_hand_tracking(hand_person, hand, paume, label_hand, id_person):
		"""Updating data hand"""

		boxe = cv2_boundingRect(np.array([j for i in hand for j in i]))
		hand_person[id_person]["hands"][label_hand] = paume
		hand_person[id_person]["has_been_detected"][label_hand] = True
		hand_person[id_person]["boxe"][label_hand] = boxe
		hand_person[id_person]["landmarks"][label_hand] = hand


	def localise_hand_from_last_coordinates(self, paume, hand, face_person, hand_person):
		"""From last hand coodinates, associate last points to the new points
		from the minimal distance and a maximal distance (RAYON_THREHSOLD). Case multipl person."""

		hand_detected = False
		hand_label = False
		person_id = None

		for id_face, data_hand in hand_person.items():

			# Run each last hand and each new hands.
			min_distance = [(distance.euclidean(paume, data_hand["hands"][label]), label)
							for label in self.hands_labels
							if data_hand["hands"][label] is not None and
							data_hand["has_been_detected"][label] is False]

			# Recuperate the minimal distance.
			if self.not_empty(min_distance):
				min_distance, hand_label = min(min_distance)

				# Verify minimal hand distance isn't to far.
				if min_distance < self.RAYON_THREHSOLD:
					hand_detected = True
					person_id = id_face

		return hand_detected, hand_label, person_id



	def tracking_with_body(self, hands_detected, hand_person, body_pers):
		""" """

		# recuperate id person & wrinkles
		id_face_body_detected = None
		wrinkle1 = None
		wrinkle2 = None

		for face_id, data in body_pers.items():
			landmarks_body = data["landmarks"]
			if self.not_empty(landmarks_body):
				id_face_body_detected = face_id
				wrinkle1 = landmarks_body[16][:2] if landmarks_body[16][-1] >= 0.5 else []
				wrinkle2 = landmarks_body[15][:2] if landmarks_body[15][-1] >= 0.5 else []
				break

		# Recuperate only wrinkles detected.
		wrinkle_list = [i for i in [wrinkle1, wrinkle2] if i not in ([], None)]

		hand_detected_to_remove = []

		# Recuperate hand of the skeletton
		if id_face_body_detected is not None: # no face id detected.

			hand_find = []

			for wrinkle in wrinkle_list:
				palm_winkle = []
 
				for index, hand in enumerate(hands_detected):
					palm = hand[0][0]
					palm_winkle += [(distance.euclidean(wrinkle, palm), index, wrinkle)]

				palm_winkle = sorted(palm_winkle)
				if self.not_empty(palm_winkle): # no palm - wrinkle detected.

					# Recuperate the minimum distance beetween: all wrinkles - all hands.
					distance_wrinkle_hand, index_hand, wrinkle_min = palm_winkle[0]

					# Filter distance. For example, a person with his hand in his pocket can be detected.
					if distance_wrinkle_hand < self.maximum_distance_wrinkle_hand:
						# Index hand matching with wrinkle for remove them.
						hand_detected_to_remove += [hands_detected[index_hand]]
						hand_find += [(hands_detected[index_hand], wrinkle_min)]

			# Remove hand on hand detected list for the next hand detection (without help of the body skeletton).
			[hands_detected.remove(i) for i in hand_detected_to_remove if i in hands_detected]


			# Update data with labeling hand.
			for (hand, wrinkle_detected) in hand_find:

				if wrinkle_detected == landmarks_body[16][:2]:
					self.updata_data_hand_tracking(hand_person, hand, hand[0][0], "left", id_face_body_detected)
				elif wrinkle_detected == landmarks_body[15][:2]:
					self.updata_data_hand_tracking(hand_person, hand, hand[0][0], "right", id_face_body_detected)

		return hands_detected


	def tracking_with_distance(self, hands_detected, face_person, hand_person, body_pers):
		""" """

		for hand in hands_detected:

			thumb, _, _, _, auricular = hand
			paume = thumb[0]

			# Maximum value hand can move in one frame.
			if self.RAYON_THREHSOLD is None:
				self.RAYON_THREHSOLD = distance.euclidean(paume, auricular[0]) * 3.5

			# Compare last & current coordinates.
			hand_detected, label_hand, id_person = self.localise_hand_from_last_coordinates(
				paume, hand, face_person, hand_person)

			if hand_detected is False:
				label_hand, id_person = self.define_first_appear_hand(hand, face_person, hand_person)

			if label_hand is not None:

				# If hand are in the same side. Case Multipl persons in frame.
				redefine_label = self.controle_hand_are_in_the_same_side(paume, hand_person, id_person, label_hand)
				label_hand = redefine_label if redefine_label is not None else label_hand

				self.updata_data_hand_tracking(hand_person, hand, paume, label_hand, id_person)


	def hand_tracking(self, hands_detected, face_person, hand_person, body_pers):
		"""Associate hand to a face & definate hand label. 
		With help of body skeletton & distance in the case where body skeletton isn't detected."""

		hand_detected = self.tracking_with_body(hands_detected, hand_person, body_pers)
		self.tracking_with_distance(hands_detected, face_person, hand_person, body_pers)
		




	def define_first_appear_hand(self, hand_detected, face_person, hand_person_data):
		"""Define hand label (right of left hand) in function of the nose. Case multipls persons."""

		palm = hand_detected[0][0]

		# Closest hand to closest nose.
		faces_nose = {
			face_id: data_face["face_nose_repear"]
			for face_id, data_face in face_person.items()}

		distance_hand_face = [
			(distance.euclidean(nose, palm), face_id, nose)
			for face_id, nose in faces_nose.items()]

		label_hand = None
		id_person = None

		if self.not_empty(distance_hand_face):
			_, id_person, nose_coordinate = min(distance_hand_face)
			label_hand = "right" if palm[0] >= nose_coordinate[0] else "left"

		return label_hand, id_person



	@staticmethod
	def redefine_hand_label(hand_person, id_person, label_redefine, last_label):
		"""Redefine hand label in case detection's wrong.
		- Two hands in a same side + no body skeletton detected on the person. -"""

		hand_person[id_person]["hands"][label_redefine] = hand_person[id_person]["hands"][last_label]
		hand_person[id_person]["has_been_detected"][label_redefine] = hand_person[id_person]["has_been_detected"][last_label]
		hand_person[id_person]["boxe"][label_redefine] = hand_person[id_person]["boxe"][last_label]
		hand_person[id_person]["landmarks"][label_redefine] = hand_person[id_person]["landmarks"][last_label]


	def controle_hand_are_in_the_same_side(self, paume, hand_person, id_person, label_hand):
		"""More one person in frame. Search to verify hand label."""

		# Hand detections.
		hand_detected = [hand_person[id_person]["has_been_detected"][label] for label in self.hands_labels]
		has_been_detected = {"right": hand_detected[0], "left": hand_detected[1]}

		redefine_label = None

		# Hand label already define & are in the same side
		if has_been_detected[label_hand]:

			# Choice the hand at the extremum.
			# Side : right first hand (500, y) second (300, y): first hand is the right hand.

			x1, y1 = paume
			x2, y2 = hand_person[id_person]["hands"][label_hand]

			if label_hand in "right" and x1 > x2:
				redefine_label = "right"
				self.redefine_hand_label(hand_person, id_person, "left", "right")

			elif label_hand in "left" and x1 < x2:
				redefine_label = "left"
				self.redefine_hand_label(hand_person, id_person, "right", "left")

		return redefine_label 



	def get_area_around_poignets(self):
		"""Get a boxe of 50 % of the face around wrinkles
		if a hand's detected of of the boxe remove it - One person case"""

		_, _, wFace, hFace = self.class_object_data_face["face_box"]
		landmarks_body = self.class_object_data_body["landmarks"]

		percent_width, percent_height  = [self.percent_of(i, 10) for i in [wFace, hFace]]

		pts = ([20, 22, 18, 16],  [15, 17, 21, 19])

		points = [self.recuperate_landmarks_from_liste(landmarks_body, liste) for liste in pts]

		poignets = [] 
		for landmarks in points:
			x, y, w, h = cv2_boundingRect(np.array(landmarks))
			poignets += [(x - percent_width, y - percent_height, x + w + percent_width, y + h + percent_height)]

		return poignets


	def remove_false_detection_if_there_is_one_person_from_body_landmarks(self, draw_frame):
		"""One person detected, we can detecte body. Recuperate hands detected and
		verify hands are in a perimeter of 120 % of the wrists. If there arn't it's a false detection."""

		landmarks_body = self.class_object_data_body["landmarks"]
		boxe_face = self.class_object_data_face["face_box"]

		if self.not_empty(landmarks_body) and boxe_face is not None:

			wrists = self.get_area_around_poignets()

			[cv2_rectangle(draw_frame, (x, y), (w, h), (255, 0, 0), 1) for (x, y, w, h) in wrists]
			cv2.circle(draw_frame, landmarks_body[16][:2], 5, (255, 255, 0), 5)


			for label in self.hands_labels:

				landmarks = self.class_object_data_hand["landmarks"][label]

				if self.not_empty(landmarks):

					is_in = [self.utils_point_is_in_boxe(landmarks[0][0], x, y, w, h) for (x, y, w, h) in wrists]

					# False detection put data to None
					if True not in is_in:

						for data_name in self.hand_data:
							self.class_object_data_hand[data_name][label] = None

						self.class_object_data_hand["landmarks"][label] = []




	def redefinate_number_hand_detection(self, faces_detected):
		"""Redefinate number of hands to detecte in function of the faces detected."""

		number_hands_to_search_from_face = len(faces_detected) * 2

		if number_hands_to_search_from_face not in (self.max_num_hands_to_search, 0):
			self.max_num_hands_to_search = number_hands_to_search_from_face

			self.hands_searching = self.mp_hands.Hands(
				self.min_detection_hand,
				self.max_num_hands_to_search,
				self.min_tracking_hands,
				self.static_image_mode,
			)

