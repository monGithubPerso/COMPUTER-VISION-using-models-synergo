import cv2
import numpy as np

from display.feature_part import Feature_display
from display.data_part import Data_to_display
from display.face_part import Face_display
from display.hand_part import Hand_display
from display.info_part import Info_display
from display.marquor_part import Marquors_display
from utils.function_utils import Utils



class Displaying2(Utils):
	"""Gestion of the click on the frame, displaying of data."""

	def __init__(self, video_path, path_recognition):
		"""Constructor"""

		# font default
		self.font_color = "white"
		self.color_mode = (255, 255, 255)
		self.font = cv2.FONT_HERSHEY_SIMPLEX

		self.none_label = [None, "None"]
		self.data_feature_face = [
			"emotions", "face_direction", "face_showing",
			"leaning_head", "face_direction_y"]


		# Display all features (info, face ect...)
		self.mode = "all"

		# Page localisation (Page even are marquor, pair are display).
		# Number of page = (numbers face * 2) + marquors extra pages.
		self.window = 0
		self.face_detected = 0

		# Display or not the tool barre (1 displaying 0 not displaying).
		self.arrow_barre = 0

		# Path of the current video for the replay.
		self.video_path = video_path

		# (number face * 2)
		self.dico_page = {}

		# Initialise all instance needed for the display.
		self.data_to_display = Data_to_display() # Recuperate in the database all data need.
		self.marquors_display = Marquors_display(video_path) # Marquors (emplacements, replay).
		self.info_display = Info_display() # Features in side of the frame.
		self.hand_display = Hand_display() # Hand display.
		self.feature_display = Feature_display() # Skelleton & face landmarks display.
		self.face_display = Face_display() # Face display
 
		# Emplacements in the window clickable (in click in these emplacements produce somethings).
		self.arrow_barre_open = (5, 35, 320, 360)
		self.arrow_barre_close = (375, 385, 320, 360)
		self.face_mode = (20, 80, 320, 350)
		self.hand_mode = (90, 150, 320, 350)
		self.info_mode = (160, 220, 320, 350)
		self.all_mode = (300, 360, 320, 350)
		self.feature_mode = (230, 290, 320, 350)
		self.modificate_font_color = (550, 570, 330, 350)

		# Toolbar text in rectangles coordinates.
		self.coordinates_rectangle_toolbar = [
			(20, 80), (90, 150), (160, 220), (230, 290), (300, 360)]

		# Toolbar texte emplacement.
		self.label_and_position_toolbar = [
			("FACE", (34, 340)), ("HAND", (103, 340)),
			("INFO", (175, 340)), ("FEATURE", (233, 340)),
			("ALL", (320, 340))]

		# Path picture for recognition treatment.
		self.path_recognition = path_recognition


	def change_color_font(self):
		"""Modification of the color font"""

		if self.font_color is "white":
			# Color's white pass to black.
			self.color_mode = (0, 0, 0)
			self.font_color = "black"

		else:
			# Color's black pass to white.
			self.color_mode = (255, 255, 255)
			self.font_color = "white"


	def arrow_display_to_the_next_page(self, frame):
		"""Displaying position of the page & font button."""

		# Number's of the page.
		pages = str(len(self.dico_page) * 2)
		current = str(self.window)
		index_page = current + "/" + pages

		# Changing page button
		cv2.putText(frame, index_page, (603, 330), self.font, 0.3, self.color_mode)
		cv2.arrowedLine(frame, (600, 340), (630, 340), self.color_mode, 3)

		# Font button
		cv2.putText(frame, "Font", (550, 330), self.font, 0.3, self.color_mode)
		cv2.circle(frame, (560, 340), 6, self.color_mode, 2)
		cv2.circle(frame, (560, 340), 2, self.color_mode, -1)


	def switch_window_on_click_arrow(self, x, y):
		"""User has click on the arrow for changing of window."""

		# Current page
		self.window += 1

		# If current + 1 page's highter of the number
		# of page, return to the first page.
		return_first_page = self.face_detected if self.face_detected == 1 else self.face_detected + 1

		if self.window > return_first_page:
			self.window = 0


	def click_on_arrow_page(self, event, x, y, flags, params):
		"""Click gestion. Toolbar, switch page & lunch a marquor on click."""

		# Event of the mouse.
		if event == cv2.EVENT_LBUTTONDOWN:

			# Verify mouse coordinates (x, y) is in a boxe of clickable.
			click_on_emplacement = lambda x, y, boxe: True if boxe[0] <= x <= boxe[1] and\
								   boxe[2] <= y <= boxe[3] else False

			# Switching page.
			if click_on_emplacement(x, y, (580, 640, 320, 360)):
				self.switch_window_on_click_arrow(x, y)

			# Toolbar.
			if self.window % 2 is 0:
				self.toolbar_click(x, y, click_on_emplacement)

			# On click marquor.
			elif self.window % 2 is not 0:
				self.marquors_display.lunch_a_replay_on_click(x, y)


	def toolbar_click(self, x, y, click_on_emplacement_function):
		"""On click on the toolbar we can have differents display. Onclick on a rectangle
		modify the mode."""

		if click_on_emplacement_function(x, y, self.arrow_barre_open):
			self.arrow_barre = 1

		elif click_on_emplacement_function(x, y, self.arrow_barre_close):
			self.arrow_barre = 0

		elif click_on_emplacement_function(x, y, self.face_mode) and self.arrow_barre is 1:
			self.mode = "face"

		elif click_on_emplacement_function(x, y, self.hand_mode) and self.arrow_barre is 1:
			self.mode = "hand"

		elif click_on_emplacement_function(x, y, self.info_mode) and self.arrow_barre is 1:
			self.mode = "info"

		elif click_on_emplacement_function(x, y, self.all_mode) and self.arrow_barre is 1:
			self.mode = "all"

		elif click_on_emplacement_function(x, y, self.feature_mode) and self.arrow_barre is 1:
			self.mode = "feature"

		elif click_on_emplacement_function(x, y, self.modificate_font_color):
			self.change_color_font()



	def build_dico_window(self, face_person_detected):
		"""Build page of the dico."""
		number_of_faces_detected = len(face_person_detected)

		if number_of_faces_detected > self.face_detected:
			self.face_detected = number_of_faces_detected

		for index_page in range(number_of_faces_detected):
			if index_page not in self.dico_page:
				self.dico_page[index_page] = [index_page * 2, (index_page * 2) + 1]


	def luminosity_tool_barre(self, frame, virgin):
		"""Recuperate original background and put modify luminosity on."""

		# Recuperate the first emplacement & put all others emplacements
		# of the tool barre with the same color.
		fisrt_slots = self.coordinates_rectangle_toolbar[0]

		crop = virgin[320:350, fisrt_slots[0]:fisrt_slots[1]]
		# Ajust luminosity.
		gamma_choice = 0.7 if self.color_mode == (255, 255, 255) else 1.4
		crop_gamma = self.adjust_gamma(crop, gamma=gamma_choice)
  
		for (x, w) in self.coordinates_rectangle_toolbar:
			# Glue the localisation by the gamma crop.
			frame[320:350, x:w] = crop_gamma


	def tool_barre(self, frame, virgin):
		"""Toolbar's differents displaying. Need to open the tool bar, to close it and to
		display it."""

		close = 0

		# Arrow down if arrow barre's close.
		if self.arrow_barre is close:
			cv2.arrowedLine(frame, (20, 350), (20, 335), self.color_mode, 2)

		else:
			# Modify luminosity of the tool barre.
			self.luminosity_tool_barre(frame, virgin)

			# Make rectangles of the toolbar.
			[cv2.rectangle(frame, (x, 320), (w, 350), self.color_mode, 1)
			 for (x, w) in self.coordinates_rectangle_toolbar]

			# Place label in.
			[cv2.putText(frame, text, coordinates, self.font, 0.4, self.color_mode)
			 for (text, coordinates) in self.label_and_position_toolbar]

			# Arrow up if arrow barre's open.
			cv2.arrowedLine(frame, (380, 320), (380, 350), (255, 255, 255), 2)


	def marquor_title(self, frame, id_face):
		"""Put tilte of marquors."""
		cv2.putText(frame, f"Marquors Page of {id_face}", (50, 20), self.font, 0.4, self.color_mode)


	def switch_color(self):
		"""In all class, switch color of the display if user press 'font' button"""

		self.marquors_display.change_color_font(self.color_mode)
		self.info_display.change_color_font(self.color_mode)
		self.hand_display.change_color_font(self.color_mode)
		self.feature_display.change_color_font(self.color_mode)
		self.face_display.change_color_font(self.color_mode)



	def face_id_not_in_detection(self, frame, face_id, face_pers):
		"""If there is no detection of the person, put the picture of face recognition."""

		id_pers_from_window = (self.window // 2) if self.window % 2 is 0 else (self.window // 2) - 1
		id_pers_from_window = id_pers_from_window if id_pers_from_window > 0 else 0

		detection = face_pers[id_pers_from_window]["is_detected"]

		if not detection and self.window % 2 == 0:

			frame_height, frame_width = frame.shape[:2]
			# Frame to black.
			frame[0:frame_height, 0:frame_width] = 0, 0, 0

			image = cv2.imread(self.path_recognition + f"/{id_pers_from_window}.png")
			image_height, image_width = image.shape[:2]

			x = int((frame_width / 2) - (image_width / 2))
			y = int((frame_height / 2) - (image_height / 2))

			frame[y:y+image.shape[0], x:x+image.shape[1]] = image



	@staticmethod
	def recuperate_data_need(data_face, data_body):
		"""Recuperate data needed."""

		landmarks_face = data_face["face_landmarks"]
		landmarks_body = data_body["landmarks"]
		face_boxe = data_face["face_box"]
		body_boxe = data_body["contour_body"]

		data = [landmarks_face, landmarks_body, face_boxe, body_boxe]

		return data


	def placing_data(self, frame, virgin, face_id, data_face, data_body, data_eye,
					 data_hand, data_analyse, face_pers, marquors, timer):
		""" """

		# Recuperate data
		data = self.recuperate_data_need(data_face, data_body)
		landmarks_face, landmarks_body, face_boxe, body_boxe = data

		# Number of page are - number of detection * 2 -
		# One page for the display and another for the marquors.
		self.build_dico_window(face_pers)

		database = data_face, data_body, data_eye, data_hand, data_analyse

		# A face detection's false is a face non detected in the last frame.
		self.face_id_not_in_detection(frame, face_id, face_pers)

		# Page is even. Display focus on face.
		if self.window % 2 is 0 and self.window in self.dico_page[face_id]:
			try:
				data_to_display = self.data_to_display.recuperate_data_interest(database, face_id, timer)
				data_info_around_face, data_info_side_frame = data_to_display

				if self.mode in ("face", "all"):
					self.face_display.face_mode_display(frame, face_boxe, landmarks_face, data_face, self.mode, timer)

				if self.mode in ("hand", "all"):
					self.hand_display.hand_mode_display(frame, virgin, data_hand, data_analyse, face_id, timer, self.mode)

				if self.mode is "feature":
					self.feature_display.features_mode(frame, data_body, data_hand, data_face, face_boxe)

				if self.mode in ("info", "all"):

					self.info_display.display_information_data_side_frame(frame, data_info_side_frame,
						face_boxe, landmarks_face, landmarks_body)

					self.info_display.display_information_data_side_body(
						frame, data_info_around_face, face_boxe, body_boxe)
			except:
				pass
			self.tool_barre(frame, virgin)

		# Page's odd. Marquor display.
		elif self.window % 2 != 0 and self.window in self.dico_page[face_id]:
			
			# Title of the page.
			self.marquor_title(frame, face_id)

			# Placing marquors in column with a different color in each colum.
			self.marquors_display.placing_marquors(frame, data_analyse["marquors"])


		# On click on the arrow page, add 1 to the dico page.
		# If we are highter of (number of detection * 2) return to the first page.
		self.arrow_display_to_the_next_page(frame)

		# On click on font button, pass color of the text to white or black.
		self.switch_color()

		return frame

