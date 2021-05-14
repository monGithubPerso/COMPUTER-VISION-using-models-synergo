#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


from scipy.spatial import distance

from cv2 import CascadeClassifier as cv2_CascadeClassifier
from cv2 import resize as cv2_resize
from cv2 import cvtColor as cv2_cvtColor
from cv2 import split as cv2_split
from cv2 import boundingRect as cv2_boundingRect
from cv2 import COLOR_BGR2LAB as cv2_COLOR_BGR2LAB
from cv2 import putText as cv2_putText
from cv2 import FONT_HERSHEY_SIMPLEX

import numpy as np

from keras.models import load_model
from face_recognition import face_locations
from keras.applications.mobilenet import preprocess_input as keras_preprocess_input

import dlib
from imutils import face_utils
from utils.function_utils import Utils


class Eye_sign(Utils):
	"""Recuperate of the closing eyes"""

	def __init__(self, path_eyes, path_haar, path_predictor):

		"""
		blink_ratio_threhsold : Define if eye's blink.
		right_eye, left_eye : landmarks.
		mean_time_closing_duration: blink -> we stop detection during 0.25 seconds
		"""
	
		# Landmarks - points of the contours of the eyes.
		self.right_eye = [36, 37, 38, 39, 40, 41]
		self.left_eye = [42, 43, 44, 45, 46, 47]

		# Threshold for say if the eyes if close.
		self.blink_ratio_threhsold = 4.8
		self.mean_time_closing_duration = 0.25

		# Width of the face's in mean 14.9 cm
		# We can scale distance (real width face / width face in the video).
		# all distance are (reduce or increase) of that.
		self.width_head = 14.9

		self.scaleFactor = 1.3
		self.minNeighbors = 1
		self.minSize = (80, 80)

		self.path_haar = path_haar
		self.path_eyes = path_eyes

		self.face_cascade = cv2_CascadeClassifier(path_haar)

		self.detector_dlib = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(path_predictor)

		self.ratio_pairs_one = [(1, 5), (2, 4), (0, 3)]
		self.scale = 0.5

		left_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		right_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		self.eyes_indexs = [(right_index), (left_index)]

		self.model = load_model(path_eyes)

		self.class_object_data_face = {}
		self.class_object_data_eyes = {}

		self.timer = 0 # Video time.

		self.blink_threshold_time = 0 # Time detection beetween 2 closings.

		# A face can be detected 2 times.
		# Threshold distance around a face already detected for avoid a second detection.
		self.is_the_same_face_threshold = 10

		# Eyes are close.
		self.ratio_eyes_are_close = (0.18, 5)

		# Eyes can be close need eye model detection
		self.ratio_eyes_are_close_verification = (0.2, 4.8)



	def getter_data_face(self, data_face):
		"""Get face data"""
		self.class_object_data_face = data_face

	def getter_data_eyes(self, data_eyes):
		"""Get eyes data"""
		self.class_object_data_eyes = data_eyes

	def getter_timer(self, timer):
		"""Get timer in the video"""
		self.timer = timer

	def raise_data(self):
		"""Raise data"""
		self.class_object_data_face = {}
		self.class_object_data_eyes = {}


	def recuperate_landmarks_interest(self, landmarks_face) -> list():
		eyes_points = np.array([self.left_eye, self.right_eye])
		return [[landmarks_face[n] for n in eye] for eye in eyes_points]


	def get_blink_ratio_1(self, eyes_landmarks) -> list():

		get_distance = np.array([
			[distance.euclidean(eye[coord1], eye[coord2]) for (coord1, coord2) in self.ratio_pairs_one]
			for eye in eyes_landmarks])

		return np.sum([(A + B) / (2.0 * C) for (A, B, C) in get_distance]) / 2


	@staticmethod
	def get_blink_ratio_2(eye_landmarks):
		"""Récupération des distances: longueurs et largeurs des yeux et du ratio entre des deux distances."""

		midPoint = lambda point1, point2: ( (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2 )

		# loading all the required points
		corner_left  = (eye_landmarks[0][0], eye_landmarks[0][1])
		corner_right = (eye_landmarks[3][0], eye_landmarks[3][1])

		center_top    = midPoint(eye_landmarks[1], eye_landmarks[2])
		center_bottom = midPoint(eye_landmarks[5], eye_landmarks[4])

		# calculating distance
		horizontal_length = distance.euclidean(corner_left, corner_right)
		vertical_length = distance.euclidean(center_top, center_bottom)

		return horizontal_length / vertical_length



	def crop_eyes(self, gray, landmarks):
		"""Crop of the eyes (select eye region of the frame)"""
		crop_lambda = lambda rect: gray[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
		landmarks = [landmarks[begin : end] for (begin, end) in self.eyes_indexs]
		crops = [crop_lambda(cv2_boundingRect(np.array([eye_points]))) for eye_points in landmarks]
		return crops


	def face_detection_with_face_recognition(self, rgb_frame):
		"""I dont understand you can see the github in ciration in (blink models)"""

		original_height, original_width = rgb_frame.shape[:2]

		resized_image = cv2_resize(rgb_frame,  (0, 0), fx=self.scale, fy=self.scale)

		lab = cv2_cvtColor(resized_image, cv2_COLOR_BGR2LAB)
		l, _, _ = cv2_split(lab)

		resized_height, resized_width = l.shape[:2]
		height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

		face_loc = face_locations(l, model='hog')

		return face_loc, height_ratio, width_ratio



	def predict_eye_state(self, image):
		"""Treatment of crop for model prediction"""
		image = cv2_resize(image, (20, 10))
		image = image.astype(dtype=np.float32)

		image_batch = np.reshape(image, (1, 10, 20, 1))
		image_batch = keras_preprocess_input(image_batch)

		pred = np.argmax( self.model.predict(image_batch)[0] )

		return pred



	def is_same_face_detected(self, current_face_boxe, face_boxe_detected):
		"""Haarcascade can detect the same face two times. Try to avoid that."""

		x1, y1 , x2, y2 = face_boxe_detected # Face already detected.
		xFace, yFace, wFace, hFace = current_face_boxe # Haarcascade detections.

		# Threshold distance around face if distance < at the threshold, it's the same face.
		threshold = self.percent_of(x2, self.is_the_same_face_threshold)

		distance_in_threshold = lambda pts1, pts2: abs(pts1 - pts2) < threshold

		same_face = distance_in_threshold(xFace, x1) and distance_in_threshold(yFace, y1) and\
					distance_in_threshold(wFace+xFace, x2) and distance_in_threshold(yFace+hFace, y2)

		return same_face



	def update_closing_eyes_in_database(self):
		"""Update closing eyes in database (open eye false, during of blinks & begening of the blink).
		In closing_historic 1) we recuperate only the begening of the blink.
			-> With it we can detect 2 blinks successives.
		In closing_historic 2) we recuperate the difference beetween end & begin.
			-> With it we can detect a long blink or a short blink."""

		historic_blinking = self.class_object_data_eyes["closing_historic"] # List where we keep data.

		self.class_object_data_eyes["open"] = False # Indicate if eyes are open.
		self.class_object_data_eyes["is_closing"] += [self.timer] # Temporary list.

		# First blink historic_blinking's empty.
		if not self.not_empty(historic_blinking):
			self.class_object_data_eyes["closing_historic"] += [self.timer]

		else:
			# Other blinks. We can put a timer beetween two blinks for avoid 2 blinks if it's the same blink.
			# False detection of an open eye in a close eye.
			last_blink_timer = historic_blinking[-1]
			if self.timer - last_blink_timer > self.blink_threshold_time:
				self.class_object_data_eyes["closing_historic"] += [self.timer]


	def recuperate_data_closing_eyes(self):

		landmarks = self.class_object_data_face["face_landmarks"]
		face_box = self.class_object_data_face["face_box"]
		eyes_landmarks = self.recuperate_landmarks_interest(landmarks)

		return landmarks, face_box, eyes_landmarks


	def ear_detection(self, ear_ratio, blink_ratio):
		"""Verify if ear are (less or above) thresholds."""

		eyes_close = (ear_ratio < self.ratio_eyes_are_close[0])\
					 or (blink_ratio > self.ratio_eyes_are_close[1])

		need_verification = (ear_ratio <= self.ratio_eyes_are_close_verification[0])\
							or (blink_ratio > self.ratio_eyes_are_close_verification[1])

		return eyes_close, need_verification


	def model_prediction(self, rgb_frame, gray):
		"""Using face recognition for detect the face dlib for 
		recuperate the point and a blink eye model."""

		eyes_crops = []

		face_boxe = self.class_object_data_face["face_box"]

		# Searching face with face_recognition.
		face_locations, height_ratio, width_ratio = self.face_detection_with_face_recognition(rgb_frame)

		# Found face.
		for (y1, x2, y2, x1) in face_locations:

			# Resize face.
			x1, y1, x2, y2 = [int(i * width_ratio) if nb % 2 == 0 else int(i * height_ratio) 
							for nb, i in enumerate([x1, y1, x2, y2])]

			# With the haarcascade we can have 2 detections in the same face
			# verify face hasn't been already detected.
			if self.is_same_face_detected(face_boxe, (x1, y1, x2, y2)):

				# Searching Dlib points.
				shape = self.predictor(gray, dlib.rectangle(x1, y1, x2, y2))
				# Reshaping data for the model.
				face_landmarks = face_utils.shape_to_np(shape)
				eyes_crops = self.crop_eyes(gray, face_landmarks)

		return eyes_crops


	def closing_eyes(self, frame, gray, rgb_frame, draw_frame):
		"""
		Haarcascade -> DLIB -> closing (ear > 5 or ear < 0.18).
		Haarcascade -> DLIB -> closing (ear > 4.8 or ear < 0.20) -> face recognition -> DLIB -> model.

		We recover the coordinates of the periphery of the eye given by DLIB.
		Then we will divide: the horizontal distance of the eye
		by the vertical distance of the eye. This gives us a ratio.
		If the ratio is greater than 5 for the first ratio or less than 0.18 for the second ratio then the eye is closed.
		If the eye is closed we store the time index of the video.
		To avoid bad detections, we only recover the first closure.
		If the ratio is greater than 4.8 or less than 0.2 we use a model that uses face recognition
		and dlib."""

		# Recuperate data interest.
		landmarks, face_box, eyes_landmarks = self.recuperate_data_closing_eyes()

		# Thresholds eyes closes.
		ear_ratio = self.get_blink_ratio_1(eyes_landmarks)
		blink_ratio = np.mean([self.get_blink_ratio_2(eyes) for eyes in eyes_landmarks])

		# Ratio and thresholds limits.
		eyes_close, need_verification = self.ear_detection(ear_ratio, blink_ratio)

		# Eyes ratio under the threshold limits (eyes are closes)
		if eyes_close:
			# Save data.
			self.update_closing_eyes_in_database()

		# Eyes are a litlle bit above threshold limits verify with model.
		elif need_verification:

			# Model using face recognition for detected open - close eyes.
			eyes_crops = self.model_prediction(rgb_frame, gray)

			# Model prediction
			open_eye = [True if self.predict_eye_state(i) else False for i in eyes_crops]

			if False in open_eye:
				self.update_closing_eyes_in_database()



	def closing_eyes_frequency(self):
		"""Récupération du temps de la détection du visage.
		Récupération des fermtures de yeux.
		Comparaison de la fermture avec la moyenne Humaine

		25 / min: calm
		10 / min: concentré
		30 / min: anxieux.
		"""

		if not self.class_object_data_eyes["open"]:

			# Recuperate moment in video were eyes closed.
			historic_eyes_clink = self.class_object_data_eyes["closing_historic"]
			# We recuperate each moment were they are close. Reorganize them by group of range of 0.4.
			# For example: Blinks are under form: [0.12 0.13 0.17 0.29]. Group: [0.12 0.13 0.17] & [0.29]
			historic_eyes_clink = self.utils_groupe_timer_by_range(historic_eyes_clink, threshold_time=0.4)

			# Recuperate the begening of the close.
			historic_eyes_clink = [begin for (begin, end) in historic_eyes_clink]

			if self.not_empty(historic_eyes_clink):

				# Recuperate the begening of the face detection and the last blink of the eye.
				timer_is_in_frame_from = self.class_object_data_face["timer_detection"]
				timer_detection = self.timer - timer_is_in_frame_from[-1]

				# Recuperate blink eye in range face detection - last blink.
				are_close_in_range = np.array([i for i in historic_eyes_clink if i >= timer_is_in_frame_from[-1]])

				same_time = historic_eyes_clink[-1] == self.timer

				# Compare ratio.
				if timer_detection > 0 and same_time:
					frequence_closing_eyes = len(are_close_in_range) / timer_detection

					from_mean = ">" if frequence_closing_eyes > 25 / 60 else "=="
					from_mean = "<" if frequence_closing_eyes < 15 / 60 else from_mean

					# Save data.
					self.class_object_data_eyes["frequency_closing"]["from_mean"] = from_mean
					self.class_object_data_eyes["frequency_closing"]["by_min"] = round(frequence_closing_eyes * 60)



	def cant_detect_eyes(self):
		"""Supression des mauvaises détections de fermeture:
		ouverture & fermture successives trop proches"""

		# Recuperate data.
		eyes_blink_historic = self.class_object_data_eyes["closing_historic"]

		# Duration beetween closing eyes.
		time_beetween_close = np.array([eyes_blink_historic[t + 1] - eyes_blink_historic[t]
							   for t in range(len(eyes_blink_historic) - 1)])

		# Chain close - open < 0.6 seconds.
		thresold = 0.6
		false_detection = np.array([True if time_close <= thresold else False
									for time_close in time_beetween_close])

		# Index data to remove.
		counter_close = 0
		to_remove = []

		for index, space_inf_to_thresold in enumerate(false_detection):
			if space_inf_to_thresold is True:
				counter_close += 1

			else:
				# More six closes - opens < 0.6 seconds are following.
				if counter_close >= 6:

					begin_delete = index - counter_close
					end_delete = index + 2
					to_remove += np.array([i for i in eyes_blink_historic[begin_delete: end_delete]])

				counter_close = 0

		# Removing.
		[eyes_blink_historic.remove(close_detection) for close_detection in to_remove]

		#Update data.
		self.class_object_data_eyes["closing_historic"] = eyes_blink_historic
