
import os
import time
import numpy as np
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=1)

#
from person.person import Person

#
from hand.hand_tracking import Hand_tracking
from hand.hand_sign import Hand_position
from hand.hand_movement import Hand_movements
from hand.hand_sign import Hand_sign

#
from face.face_tracking import Face_tracking
from face.face_informations import Face_information
from face.face_movement import Face_movements

#
from body.body import Body_distance

#
from eyes.pupille_tracking import Eyes_tracking
from eyes.eyes_sign import Eye_sign

#
from analyse.hand_analyse import Hand_analyse
from analyse.head_analyse import Head_analyse
from analyse.eyes_analyse import Eyes_analyse
from analyse.body_analyse import Body_analyse

#
from display.display1 import *



class Main_utils():
	"""To avoid overloading in Main.py, Main_utils takes care of calling the classes you need.
	Basically, main_utils:

	- class instance call,

	- call the feature detection classes (faces, hands ...),

	- creates and retrieves the database (in the form of a dictionary),

	- makes the association and the tracking of the features between them
	for example: these hands, these eyes and this body sound to this face ...

	- collects and processes these different features
	such as feature removal, measuring angles and distances,

	- then do the analysis or try to match measurements to representations
	for example: the angle is 10 degrees, the distance is 5 cm so it's a cat.
	"""


	def __init__(self, paths):
		"""Constructor."""

		# Paths needed.
		self.path_video = paths[0]

		self.path_dlib = paths[1] # Face detection

		self.path_genre_model = paths[2] # Genre of a person detection
		self.path_genre_txt = paths[3]

		self.path_skin_color = paths[4] # Skin color
		self.path_skin_color_txt = paths[5]

		self.path_emotion1 = paths[6] # Emotion detection
		self.path_emotion2 = paths[7]

		self.path_picture_face_recognition = paths[8] # Recognition
		self.path_recognition_dlib = paths[9]

		self.path_eyes = paths[10] # Eyes detection
		self.haar = paths[11] # Face detection


		# CONSTRUCTOR: Hand
		self.hand_tracking = None
		self.hand_movements = None
		self.hand_position = None
		self.hand_sign = None

		# CONSTRUCTOR: Eyes
		self.eyes_tracking = None
		self.eye_sign = None

		# CONSTRUCTOR: Face
		self.face_tracking = None
		self.face_movements = None
		self.face_information = None

		# CONSTRUCTOR: Body
		self.body_distance = None

		# CONSTRUCTOR: Analyses
		self.eyes_analyse = None
		self.hand_analyse = None
		self.head_analyse = None
		self.body_analyse = None

		# CONSTRUCTOR: database
		self.person = Person(self.path_picture_face_recognition)

		# CONSTRUCTOR: Display
		self.displaying2 = Displaying2(self.path_video, self.path_picture_face_recognition)


		# Tresholds

		# Stop all operations (except detection of features (faces, hands ...)) 
		# if the last face was detected above 0.5 secs.
		self.treshold_face_detection_timer = 0.5



	def load_constructor_hand_trackings(self):
		"""Load Hands class"""

		self.hand_tracking = Hand_tracking()
		self.hand_movements = Hand_movements()
		self.hand_position = Hand_position()
		self.hand_sign = Hand_sign()

		print("load_constructor_hand_trackings done")


	def load_constructor_eyes(self):
		"""Load eyes class"""

		self.eyes_tracking = Eyes_tracking()
		self.eye_sign = Eye_sign(self.path_eyes, self.haar, self.path_dlib)

		print("load_constructor_eyes done")


	def load_constructor_face_trackings(self):
		"""Load face tracking class"""

		self.face_tracking = Face_tracking(
			self.path_dlib, self.path_recognition_dlib,
			self.path_picture_face_recognition, self.haar
		)

		print("load_constructor_face_trackings done")


	def face_constructor_face_info(self):
		"""Load face informations class (emotion, skin color)"""
		self.face_movements = Face_movements()
		self.face_information = Face_information(
			self.path_emotion1, self.path_emotion2, self.path_skin_color, self.path_skin_color_txt,
			self.path_genre_txt, self.path_genre_model
		)

		print("face_constructor_face_info done")


	def constructor_body_distance(self):
		"""Load Body tracking & body movement class"""

		self.body_distance = Body_distance()

		print("constructor_body_distance done")


	def constructor_analyse(self):
		"""Load analyse class"""

		self.eyes_analyse = Eyes_analyse()
		self.hand_analyse = Hand_analyse()
		self.head_analyse = Head_analyse()
		self.body_analyse = Body_analyse()

		print("constructor_analyse done")


	def load_concstructors(self):
		"""All class instances."""

		hand = pool.apply_async(self.load_constructor_hand_trackings)
		eyes = pool.apply_async(self.load_constructor_eyes)
		face = pool.apply_async(self.load_constructor_face_trackings)

		face_info = pool.apply_async(self.face_constructor_face_info)
		body = pool.apply_async(self.constructor_body_distance)
		analyse = pool.apply_async(self.constructor_analyse)

		[detection.get() for detection in np.array([hand, eyes, face, face_info, body, analyse])]

		print("load_concstructors done")


	@staticmethod
	def display_data(pause):
		"""In click on q we can switch modes of display.
		We have two modes. Let's display the video alone or press a button for
		display picture by picture. 1 video alone - 0 picture by picture."""
		return 1 if pause == 0 else 0


	@staticmethod
	def removing_pictures_facial_recognition(path_facial_recognition):
		"""Create folder for picture faces recognition.
		At each execution, remove all pictures saved"""

		# If folder recognition doesn't exists create it.
		if not os.path.exists(path_facial_recognition):
			os.makedirs(path_facial_recognition)

		# Remove picture save from the last execution.
		picture_face_recognition = os.listdir(path_facial_recognition)
		[os.remove(f"{path_facial_recognition}/{face_id}") for face_id in picture_face_recognition]



	def detecting_features(self):
		"""Try to detecting in the frame faces, bodies, hands & eyes (theirs coordinates)
		if they're exist."""

		#time_detecting_features = time.time()

		face_detecting = pool.apply_async(self.face_tracking.face_landmarks_detection, (self.gray, self.copy1))
		hand_detecting = pool.apply_async(self.hand_tracking.hands_detection, (self.rgb_frame, ))
		body_detection = pool.apply_async(self.body_distance.body_detection, (self.rgb_frame, ))
		#eyes = self.eyes_tracking.eyes_landmarks(faces, self.virgin, self.gray)

		faces, hands, body = [detection.get() for detection in np.array([face_detecting, hand_detecting, body_detection])]

		#print("detection of the features", time.time() - time_detecting_features)

		return faces, hands, [], body



	def recuperate_person_data(self, timer, faces):
		"""Recuperate database or create a person in the database."""

		#time_recuperate_database = time.time()

		have_found_face = len(faces) > 0
		if have_found_face:
			self.person.create_person(faces, timer)

		database = self.person.getter_database_person()

		#print("recuperation database", time.time() - time_recuperate_database)

		return database



	def tracking_features(self, features_detected, database, timer):
		"""Detection of the part of body in the frame.""" 

		#time_tracking_feature = time.time()

		faces_detected, hands_detected, eyes_detected, body_detected = features_detected
		face_pers, hand_pers, eye_pers, body_pers, analyse_pers = database

		self.face_tracking.tracking_face(self.frame, faces_detected, database)
		self.body_distance.body_association(face_pers, body_pers, body_detected)

		track_hands = pool.apply_async(self.hand_tracking.hand_tracking, (hands_detected, face_pers, hand_pers, body_pers))
		#track_eyes = pool.apply_async(self.eyes_tracking.eyes_association, (face_pers, eye_pers, eyes_detected))

		[tracker.get() for tracker in np.array([track_hands])]

		#print("detection of features", time.time() - time_tracking_feature)


	def hand_sign_detection(self, data_hand, data_face):
		"""Hands can avoid the face detection. 
		If in the last frames faces were detected and hands were in the face area lunch it."""

		for label in ["right", "left"]:

			# Copy data from database in a class object for treatment.
			self.hand_sign.getter_data_hand(data_hand)
			self.hand_sign.getter_data_face(data_face)
			self.hand_sign.getter_timer(self.timer)

			# Hand touching face (localise the face emplacement).
			self.hand_sign.fingers_in_area_face(self.copy1, self.virgin, label)
			# Hand gesture.
			self.hand_sign.signs_of_the_hand(self.copy1, self.virgin, label)

		# Raise data interests (only in the class).
		self.hand_sign.raise_data()



	def face_information_data(self, data_face):
		"""Lunch all face function detection gender, skin color."""

		# Copy data from database in a class object for treatment.
		self.face_information.getter_data_face(data_face)
		self.face_information.getter_timer(self.timer)

		# Biologic genre of the person.
		gender = pool.apply_async(self.face_information.gender_detection, (self.virgin, ))
		# Skin color of the person.
		skin_color = pool.apply_async(self.face_information.skin_color, (self.virgin, ))
		# Emotion on the face of the person.
		emotions = pool.apply_async(self.face_information.emotion_detection, (self.gray, ))

		[movement.get() for movement in np.array([gender, skin_color, emotions])]

		# Raise data interests (only in the class).
		self.face_information.raise_data()





	def faces_data(self, data_hand, data_face, data_body, frame_deplacement):
		"""Lunch all face function"""
		
		# Copy data from database in a class object for treatment.
		self.face_movements.getter_data_hand(data_hand)
		self.face_movements.getter_data_face(data_face)
		self.face_movements.getter_data_body(data_body)
		self.face_movements.getter_timer(self.timer)
		self.face_movements.getter_frame_deplacement(frame_deplacement)

		self.face_tracking.getter_timer(self.timer)

		#time_treatment_faces = time.time()

		# Face of the face (right or left visible).
		face_facing = pool.apply_async(self.face_movements.face_facing, (self.copy1, ))
		# Face leaning.
		face_leaning = pool.apply_async(self.face_movements.face_leaning, (self.copy1, ))
		# Face facing.
		face_facing = pool.apply_async(self.face_movements.face_facing, (self.copy1, ))
		# Face movement (speed).
		face_move = pool.apply_async(self.face_movements.face_movement)

		face_vertical_movement = pool.apply_async(self.face_movements.face_vertical_movement)

		# Beetween wrinkle.
		beetween_eyes = pool.apply_async(self.face_movements.beetween_eye, (self.copy1, self.virgin, self.gray))
		# Forehead wrinkle.
		forehead = pool.apply_async(self.face_movements.foreheahd, (self.copy1, self.virgin, self.gray))
		# 
		face_sign = pool.apply_async(self.face_movements.face_sign)
		# Lips movements.
		lips_movement = pool.apply_async(self.face_movements.lips_movements, (self.copy1,))

		waiters = np.array([face_facing, face_leaning, face_move, beetween_eyes, 
							forehead, face_sign, lips_movement, face_vertical_movement])
		[movement.get() for movement in waiters]

		self.face_movements.face_in_movement_in_left_or_right()

		# Raise data interests (only in the class).
		self.face_movements.raise_data()

		#print("Face treatment", time.time() - time_treatment_faces)


	def eyes_data(self, data_face, data_eye):
		"""Eyes blinking and eye movements (right - left)"""

		#time_eye_detection = time.time()

		# Copy data from database in a class object for treatment.
		self.eye_sign.getter_data_face(data_face)
		self.eye_sign.getter_data_eyes(data_eye)
		self.eye_sign.getter_timer(self.timer)

		# Detection a closing eyes.
		closing_eyes = pool.apply_async(self.eye_sign.closing_eyes, (self.frame, self.gray, self.rgb_frame, self.copy1))
		# Frequency of the closing eyes.
		closing_eyes_frequency = pool.apply_async(self.eye_sign.closing_eyes_frequency)
		# Remove false closing eyes detections.
		false_detection = pool.apply_async(self.eye_sign.cant_detect_eyes)

		waiters = np.array([closing_eyes, closing_eyes_frequency, false_detection])
		[move.get() for move in waiters]

		# Raise data interests (only in the class).
		self.eye_sign.raise_data()
		
		#print("eyes_data", time.time() - time_eye_detection)



	def body_data(self, data_face, data_body, data_hand, face_dico):
		"""Color of the body, detection of another face in the differents body space"""

		#body_timer = time.time()

		# Copy data from database in a class object for treatment.
		self.body_distance.getter_data_hand(data_hand)
		self.body_distance.getter_data_face(data_face)
		self.body_distance.getter_data_body(data_body)
		self.body_distance.getter_timer(self.timer)

		# Recuperate 50cm, 1m50, 3m around the body.
		body_space = pool.apply_async(self.body_distance.body_spaces, (self.frame, ))
		# Verify if there are others face in the space around the body.
		other_body_in_space = pool.apply_async(self.body_distance.other_body_in_space, (face_dico, ))
		# Recuperate contours of the body.
		body_contours = pool.apply_async(self.body_distance.body_contours, (self.rgb_frame, self.copy1))

		# Recuperate the color of the body (t-shirt).
		body_color = pool.apply_async(self.body_distance.body_color, (self.virgin, self.copy1))
		# Detection of the movements of the arms.
		arm_movements = pool.apply_async(self.body_distance.arm_movements)
		# Try to detect a shcema of the arms movements.
		arms_signs = pool.apply_async(self.body_distance.arm_signs, (self.copy1, ))

		waiters = np.array([body_space, other_body_in_space, body_contours, body_color, arm_movements, arms_signs])
		[i.get() for i in waiters]

		# Raise data interests (only in the class).
		self.body_distance.raise_data()

		#print("body_data final", time.time() - body_timer)



	def hand_data(self, data_face, data_hand, data_body, face_detected):
		"""Hand movements - hand side, hand direction & hand position."""

		#hand_treatment = time.time()

		# Get data need & copy them.
		self.hand_movements.getter_data_hand(data_hand)
		self.hand_movements.getter_data_face(data_face)
		self.hand_movements.getter_timer(self.timer)

		self.hand_position.getter_data_hand(data_hand)
		self.hand_position.getter_timer(self.timer)

		self.hand_tracking.getter_data_body(data_body)
		self.hand_tracking.getter_data_face(data_face)
		self.hand_tracking.getter_data_hand(data_hand)

		self.hand_tracking.remove_false_detection_if_there_is_one_person_from_body_landmarks(self.copy1)

		hand_label = ["right", "left"]

		for label in hand_label:
			
			# Hand vertical movement.
			hand_localisation = pool.apply_async(self.hand_movements.define_hand_localisation_mean_y, (label, self.frame))
			# Speed of the hands.
			speed_movement = pool.apply_async(self.hand_movements.hand_speed_dist, (label, self.copy1))
			# Direction of the hands.
			direction_movement = pool.apply_async(self.hand_position.hand_direction, (label, ))
			# Face of the hands (back or palm).
			facing_hand = pool.apply_async(self.hand_position.facing_hand, (label, self.copy1))


		waiters = np.array([speed_movement, direction_movement, facing_hand, hand_localisation])
		[i.get() for i in waiters]

		# Raise data interests (only in the class).
		self.hand_movements.raise_data()
		self.hand_position.raise_data()
		self.hand_tracking.raise_data()

		#print("hand_data", time.time() - hand_treatment)


	def analye_eyes(self, data_analyse, data_eye):
		"""Lunch all eyes analysis function"""

		# Copy somes data of the database in the class.
		self.eyes_analyse.getter_data_analyse(data_analyse)
		self.eyes_analyse.getter_data_eyes(data_eye)
		self.eyes_analyse.getter_timer(self.timer)

		# Analyse of the closing.
		closing_frequency_analyse = pool.apply_async(self.eyes_analyse.eyes_closing_significate)
		#
		closing_analyse = pool.apply_async(self.eyes_analyse.closing_analyse)
		# Recuperate long closing (> 300 ms).
		closing_analyse_duration = pool.apply_async(self.eyes_analyse.eyes_closing_time_significate)

		[element.get() for element in [closing_frequency_analyse, closing_analyse, 
		closing_analyse_duration]]

		# Raise data interests (only in the class).
		self.eyes_analyse.raise_data()


	def analyse_face(self, data_face, data_analyse, data_body):
		"""Lunch all face analysis function"""

		# Copy somes data of the database in the class.
		self.head_analyse.getter_data_face(data_face)
		self.head_analyse.getter_data_analyse(data_analyse)
		self.head_analyse.getter_data_body(data_body)
		self.head_analyse.getter_timer(self.timer)

		# Definate face facing.
		face_facing_analyse = pool.apply_async(self.head_analyse.analyse_face_face)
		# Recuperate (or delete) beetween wrinkles detection.
		marquors_analyse_beetween = pool.apply_async(self.head_analyse.analyse_marquors_face_wrinkles, ("marquors_face", "beetween_wrinkle"))
		# Recuperate (or delete) forehead wrinkles detection.
		marquors_analyse_forehead = pool.apply_async(self.head_analyse.analyse_marquors_face_wrinkles, ("marquors_forehead", "foreheahd"))
		# Recuperate lips movements interest.
		lips_move = pool.apply_async(self.head_analyse.lips_analyse)

		waiters = np.array([face_facing_analyse, marquors_analyse_beetween, marquors_analyse_forehead, lips_move])
		[element.get() for element in waiters]

		# Raise data interests (only in the class).
		self.head_analyse.raise_data()


	def analyse_head(self, data_face, data_analyse, data_body):
		"""Lunch all analysis head function"""

		# Copy somes data of the database in the class.
		self.head_analyse.getter_data_face(data_face)
		self.head_analyse.getter_data_analyse(data_analyse)
		self.head_analyse.getter_data_body(data_body)
		self.head_analyse.getter_timer(self.timer)

		# Recuperate (or delete) bot head movement.
		head_vertical_analyse_bot = pool.apply_async(self.head_analyse.head_analyse_vertical, ("bot", ))
		# Recuperate (or delete) top head movement.
		head_vertical_analyse_top = pool.apply_async(self.head_analyse.head_analyse_vertical, ("top", ))
	
		# Definate if it's a leaning & if the head's leaning (right or left)
		head_leaning_analyse = pool.apply_async(self.head_analyse.head_analyse_leaning)
		# Definate face of the face.
		face_facing_analyse = pool.apply_async(self.head_analyse.analyse_face_face)

		face_movement = pool.apply_async(self.head_analyse.face_movement_analyse)

		# Savegarde emotions if it's not neutral, happy and surprised.
		emotion = pool.apply_async(self.head_analyse.analyse_emotion)

		waiter = np.array([head_vertical_analyse_bot, head_vertical_analyse_top, 
						head_leaning_analyse, face_facing_analyse, emotion, face_movement])

		[element.get() for element in waiter]


		# Raise data interests (only in the class).
		self.head_analyse.raise_data()


	def analyse_hand(self, data_hand, data_analyse, face_id):
		"""Lunch all analysis hand function"""

		# Copy somes data of the database in the class.
		self.hand_analyse.getter_data_hand(data_hand)
		self.hand_analyse.getter_data_analyse(data_analyse)
		self.hand_analyse.getter_timer(self.timer)

		[self.hand_analyse.hand_speed_analyse(label) for label in ["right", "left"]]
		
		#
		process_hand = pool.apply_async(self.hand_analyse.analyse_process_use_hand)
		# Recuperate or no hand movement on the vertical axis.
		hand_analyse = pool.apply_async(self.hand_analyse.hand_localisation_analyse, (self.copy1, ))
		# Definate sign of the hands.
		hand_sign = pool.apply_async(self.hand_analyse.analyse_hand_sign)
		#
		#a = [pool.apply_async(self.hand_analyse.touch_face_analyse, (face_id, label)) for label in ["right", "left"]]

		waiters = [hand_sign, hand_analyse, process_hand]
		[element.get() for element in waiters]

		# Raise data interests (only in the class).
		self.hand_analyse.raise_data()


	def analyse_body(self, data_analyse, data_body):
		"""Try to find a position thank's to arms of the body localisation."""

		# Copy somes data of the database in the class.
		self.body_analyse.getter_data_analyse(data_analyse)
		self.body_analyse.getter_data_body(data_body)
		self.body_analyse.getter_timer(self.timer)

		# Definate arms positions.
		self.body_analyse.arms_sign()

		# Raise data interests (only in the class).
		self.body_analyse.raise_data()


	def there_are_no_detection_timer(self, face_pers):
		"""Sometimes, we havn't got faces detections because the hands can be on the face.
		Here we stop all treatments if the last face detected is above of threshold_face_detected (0.5 sec)."""

		recent = [self.timer - data_face["last_timer"] < self.treshold_face_detection_timer 
				 for face_id, data_face in face_pers.items()]

		recent = False if True in recent else True

		return recent


	@staticmethod
	def recuperate_timer(marquors_dict, exception):
		"""Recuperate only timer in dictionnary of marquor."""
	
		liste = [timer for marquors_name, data in marquors_dict.items()
				for timer in data if marquors_name not in exception]

		return sorted(liste)


	@staticmethod
	def removing_marquor(marquors, remove):
		"""In the dictionary of marquors, remove marquors."""

		[data.remove(timer) for data in marquors.values() 
		for timer in remove if timer in data]

		return marquors


	def reorganise_marquor(self, marquors):
		"""Sometimes some marquors are in range of another marquor.
		For example first marquor's (0.2, 0.4) & the second's (0.1, 0.5).
		Raise the first marquor."""

		exception_marquors = []

		# Recuperate all marquors time from all category.
		liste = self.recuperate_timer(marquors, exception=exception_marquors)

		remove = []

		last = len(liste)
		oContinuer = True
		while oContinuer:
			
			# If a marquors's in range of another remove it.
			for (begin1, end1) in liste:
				for (begin2, end2) in liste:
					if begin1 != begin2 and end1 != end2 and begin2 < end1 and begin2 > begin1 and end2 < end1:
						liste.remove((begin2, end2))
						remove += [(begin2, end2)]

			if last == len(liste):
				oContinuer = False

			last = len(liste)

		# Remove marquors in range of another.
		marquors = self.removing_marquor(marquors, remove)

		return marquors

















