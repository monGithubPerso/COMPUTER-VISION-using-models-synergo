#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Commands:
# 	python main.py --video {path of the video}



"""
The program does not seem to give some results
that on only one type of video with many conditions.
The face must be clearly visible (there must be no wick or lute),
the person must be in front of the video and preferably
visible at trunk level. Video can work
completely false if one of the detections is over exploited
(for example detection of confidence by the height of
hands and a video on sign language.
detection of blinking eyes (stress) and world record for blinking eyes ect ...)
All meanings given by this program
are likely to be false.
The program's not complete.




# detection feature -> creation or recuperation on database -> treatment on feature -> analyse -> display -> event mouse.


"""


import cv2
import time
from imutils import resize as imutils_resize
import argparse
import numpy as np
from utils.main_utils import Main_utils



class Main(Main_utils):
	"""Main call main utils."""

	def __init__(self, paths):
		""" """

		print("[init] Main constructor.")

		Main_utils.__init__(self, paths)

		# Remove pictures in folder of recognition picture
		self.removing_pictures_facial_recognition(paths[8])

		self.frame     = None # for display,
		self.copy1     = None # for work,
		self.virgin    = None # for treatment,
		self.rgb_frame = None # for hand & body detection.

		# DATA: Video Paths
		self.path_video = paths[0]

		self.timer = 0 # time in the video
		self.frame_deplacement = 0 # FPS calcul

		# FPS mean calcul
		self.cap_pos_msec_mean = []

		# Instance of all class.
		self.load_concstructors()

		print("[init] Constructor done.")


	def frame_treatments(self, frame):
		"""Recuperate & treat the differents frames needs"""

		# Resize frame with imutils (resize height in function of the width).
		# Frame of the display.
		self.frame = cv2.flip(imutils_resize(frame, width=640), 1)
		# Dlib frame detection (gray).
		self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		# Mediapipe frame detection (rgb).
		self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
		# Copy for work & virgin for detection.
		self.copy1, self.virgin = [self.frame.copy() for _ in range(2)]



	def features_detection_database_features_association(self):
		"""Tracking feature of the last & the current frame."""

		# Features (faces, hands, eyes & body) detections.
		features = self.detecting_features()
		# Database.
		database = self.recuperate_person_data(self.timer, features[0])
		# Associations lasts features with new features.
		self.tracking_features(features, database, self.timer)

		return features, database


	def treatment(self, data_hand, data_face, data_body, data_eye, data_analyse, face_pers, features):
		"""Relevant data on features."""

		# Treatment - relevant data of the sign of the hand
		self.hand_sign_detection(data_hand, data_face)
		# Treatment - relevant data of the face (model)
		self.face_information_data( data_face)
		# Treatment - relevant data of the face
		self.faces_data(data_hand, data_face, data_body, self.frame_deplacement)
		# Treatment - relevant data of the eyes
		self.eyes_data(data_face, data_eye)
		# Treatment - relevant data of the body
		self.body_data(data_face, data_body, data_hand, face_pers)
		# Treatment - relevant data of the hand
		self.hand_data(data_face, data_hand, data_body, features[0])
		# Treatment - relevant data of the eyes
		self.analye_eyes(data_analyse, data_eye)


	def analyse(self, data_face, data_analyse, data_body, data_hand, face_id):
		"""Analyse of data."""

		# Analyse of the face
		self.analyse_face(data_face, data_analyse, data_body)
		# Analyse of the head
		self.analyse_head(data_face, data_analyse, data_body)
		# Analyse of the hand
		self.analyse_hand(data_hand, data_analyse, face_id)
		# Analyse of the body
		self.analyse_body(data_analyse, data_body)


	def operation(self, database, features):
		"""Lunch all function needed"""
		
		# Recuperate data from database
		face_pers, hand_pers, eye_pers, body_pers, analyse_pers = database

		for data_face, data_hand, data_eye, data_body, data_analyse in\
		 zip(face_pers.items(), hand_pers.items(), eye_pers.items(), body_pers.items(), analyse_pers.items()):

				face_id, data_face = data_face
				_, data_hand = data_hand
				_, data_eye = data_eye
				_, data_body = data_body
				_, data_analyse = data_analyse

				# Faces detected in the last frame.
				if data_face["is_detected"] and data_face["face_box"] is not None:

					self.treatment(data_hand, data_face, data_body, data_eye, data_analyse, face_pers, features)
					self.analyse(data_face, data_analyse, data_body, data_hand, face_id)

				# Displaying.
				self.frame = self.displaying2.placing_data(
					self.frame, self.virgin, face_id, data_face, data_body, data_eye,
					data_hand, data_analyse, face_pers, [], self.timer)

				marquor = self.reorganise_marquor(data_analyse["marquors"])
				print("FACE ID  ", face_id, data_analyse["marquors"])


	def video_capture(self):
		"""Lunch the video"""

		cap = cv2.VideoCapture(self.path_video)
		cap.set(cv2.CAP_PROP_POS_MSEC, 0 * 1000)

		pause = 1
		counter_frame = 0

		while True:

			self.video, frame = cap.read()

			# Timer in the video in ms to s.
			self.timer = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
			self.frame_deplacement= round(np.mean(self.cap_pos_msec_mean), 1)

			# Frame resize & frames filters. Frame are flips ! (right in video is right in real)
			self.frame_treatments(frame)

			# Detection & association of features, getter database.
			features, database = self.features_detection_database_features_association()

			# If last face detected is highter of 0.5 secs (as a cinematic) pass.
			there_are_no_detection_from_a_while = self.there_are_no_detection_timer(database[0])

			if not there_are_no_detection_from_a_while:

				# Make operation on the features.
				self.operation(database, features)

				# Raise or update some detection in the database 
				# (for example: put hand detection to False, current nose becomes last nose...).
				self.person.put_false_detections(self.timer)

			# Recovery timer for calcul the average of frame / seconds,
			self.cap_pos_msec_mean += [self.timer]

			# Timer displaying.
			cv2.putText(self.frame, str(self.timer), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

			# Showing frames.
			#frames = [("frame", self.frame), ("copy", self.copy1)]
			#[cv2.imshow(frame_name, frame_display) for (frame_name, frame_display) in frames]
			cv2.imshow("frame", self.frame)

			# Click events.
			cv2.setMouseCallback("frame", self.displaying2.click_on_arrow_page)

			# Waiter.
			k = cv2.waitKey(pause)
			if k == ord('q'):
				pause = self.display_data(pause)
			elif k == ord("a"):
				self.video = False
				break
	

if __name__ == "__main__":

	a = argparse.ArgumentParser()
	a.add_argument("-v", "--video", help="path to video")
	args = a.parse_args()

	path_dlib = "data\models\dlib_model\shape_predictor_68_face_landmarks.dat"
	path_genre_model = "data\models\genre\gender_net.caffemodel"
	path_genre_txt = "data\models\genre\gender_deploy.prototxt"
	path_skin_color = "data\models\skin_color\skin_color.pb"
	path_skin_color_txt = "data\models\skin_color\skin_label.txt"
	path_emotion1 = "data\models\emotions\model.h5"
	path_emotion2 = "data\models\emotions\_mini_XCEPTION.102-0.66.hdf5"
	path_picture_face_recognition = "data\picture_face_recognition"
	# Absolute path.
	path_recognition_dlib = r"C:\Users\jeanbaptiste\Desktop\hardcoding\data\models\dlib_model\reconigtion\dlib_face_recognition_resnet_model_v1.dat"
	path_eyes = "data\models\eyes\weights.149-0.01.hdf5"
	haar = "data\models\haarcascade_frontalface_alt.xml"

	paths = [args.video, path_dlib, path_genre_model, path_genre_txt, path_skin_color,
			path_skin_color_txt, path_emotion1, path_emotion2,
			path_picture_face_recognition, path_recognition_dlib, path_eyes, haar]

	main_utilitary = Main_utils(paths)
	main_ = Main(paths)

	main_.video_capture()


