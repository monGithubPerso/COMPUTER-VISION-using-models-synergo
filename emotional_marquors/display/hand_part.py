import cv2
import numpy as np
from utils.function_utils import Utils



class Hand_display(Utils):
	"""Displayin of the hand feature."""

	def __init__(self):
		"""Constructor"""

		# font default
		self.font_color = "white"
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.color_mode = (255, 255, 255)

		# For the hand sign detection we raise the frame 
		# on the localisation of the display (copy past virgin frame on the displaying frame).
		# Raise it one time for the esthetic.
		# Can raise the frame if it false.
		self.have_raise_frame = False

		# Area of the hand. If area's less of that
		# display only the label.
		self.threshold_area_display_all_data_in_hand = (45, 30)

		self.dico_sign = {
			"peace": "Sign of peace or victory.",
			"fist": "There is a negative emotion.", 
			"approbation": "In approbation with himself or other.",
			"confident": "Positive emotion or showing.",
			"stop": "Distanciation with something."
		}



	def change_color_font(self, color_mode):
		"""Getter the color to display."""
		self.color_mode = color_mode



	def recuperate_hand_data_interest(self, data_hand, data_hand_analyse, label, face_id, timer):
		"""Recuperate data to display"""

		sign_detected = ""
		signs = data_hand_analyse["sign"][label] # sign detected.
		if len(signs) > 0:
			last_sign, last_timer = signs[-1]
			if timer == last_timer:
				sign_detected = last_sign

		hand_face_side = data_hand["faceHand"][label] # hand face (palm or back of the hand).
		hand = data_hand["landmarks"][label] # Points of the hand.
		hand_boxe = data_hand["boxe"][label] # rectangle around the hand.

		return [hand, hand_boxe, f"Id:{face_id}", f"{label}", hand_face_side, sign_detected]


	def rectangle_around_hand(self, hand_boxe):
		"""Draw the rectangle around the hands. Recuperate the hand boxe and put a margin
		in function of width & height of the hands."""

		xHand, yHand, wHand, hHand = hand_boxe

		margin_width = self.percent_of(wHand, 8)
		margin_height = self.percent_of(hHand, 8)

		rect = [( (xHand - margin_width), (yHand - margin_height) ),
				( (xHand + wHand + margin_width), (yHand + hHand + margin_height) )]

		return rect


	def corner_around_the_hand(self, rectangle, hand_boxe):
		"""Put corners on the boxe of the hand (rectangle around the hand).
	   *1     *4
		_   _
	   |     |         *1  ___  <-- (c1x)
						  |    
	   |_   _|            | <------ (c1y)
	  *2      *3                     
		"""

		xHand, yHand, wHand, hHand = hand_boxe
		x, y, w, h = [points for pairs in rectangle for points in pairs]

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

		return [c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y]


	def rectangle_around_the_hand(self, frame, hand_boxe, sign_detected):
		"""Display around the hand. Put a rectangle with corner around the hand"""

		# Add margins to the hand boxe for a bigger boxe. 
		rect_around_hand = self.rectangle_around_hand(hand_boxe)

		# Recuperate rectangle around the hand.
		#if sign_detected is not "":
		#	cv2.rectangle(frame, rect_around_hand[0], rect_around_hand[1], self.color_mode, 1)

		# Recuperate corners on the rectangle.
		corners = self.corner_around_the_hand(rect_around_hand, hand_boxe)

		# Draw corners.
		[cv2.line(frame, coord1, coord2, self.color_mode, 2) for (coord1, coord2) in corners]

		return rect_around_hand

	
	def gamma_the_text_on_the_hand(self, frame, slots, features):
		"""darken or brighten a, b, or c text."""

		width = [18, 20, 25]

		c = 0
		for (emplacement, text) in zip(slots, features):

			# Define the rectangle.
			x, y = emplacement

			crop = frame[y-5:y+2, x:x+width[c]]

			# Ajust luminosity.
			gamma_choice = 0.5 if self.color_mode == (255, 255, 255) else 1.4
			crop_gamma = self.adjust_gamma(crop, gamma=gamma_choice)

			# Glue the localisation by the gamma crop.
			frame[y-5:y+2, x:x+width[c]] = crop_gamma

			c+=1


	def drawing_around_the_hand(self, frame, hand_boxe_resized, features, mode):
		"""
			 _   _  
			|a   b|     
				   
			|_   _| 
			c
		"""

		face_id, label, hand_face_side, sign_detected = features[2:]

		(x, y), (w, h) = hand_boxe_resized


		add_x = self.percent_of(w, 2)
		add_y = self.percent_of(h, 6)

		slot1 = (x + add_x, y + add_y) # a
		#slot2 = (w - int(len(label) * 5.5), y + add_y) # b
		slot3 = (x + add_x, h - 10) # c

		w_threhsold, h_threshold = self.threshold_area_display_all_data_in_hand 

		if mode is "hand":
			(x, y), (w, h) = hand_boxe_resized
			crop = frame[y:h, x:w]
			gamma_choice = 0.7 if self.color_mode == (255, 255, 255) else 1.3
			hand_crop = self.adjust_gamma(crop, gamma=gamma_choice)

			frame[y:h, x:w] = hand_crop


		if abs(x-w) > w_threhsold or abs(y-h) > h_threshold:

			slots = [slot1, slot3]

			self.gamma_the_text_on_the_hand(frame, slots, features[3:])

			[cv2.putText(frame, text.capitalize(), emplacement, self.font, 0.3, self.color_mode) 
			for (emplacement, text) in zip(slots, features[3:])]

		else:

			(x, y), (w, h) = hand_boxe_resized

			length_label = 20
			height_label = 15

			mid_x = x + int( ( (w - x) - length_label ) / 2)
			mid_y = y + round( (h - y) / 2)

			#self.gamma_the_text_on_the_hand(frame, [(mid_x, mid_y)], [label])
			cv2.putText(frame, label.capitalize(), (mid_x, mid_y), self.font, 0.3, self.color_mode) 



	def drawing_lines(self, frame, hand_boxe_resized):
		"""Drawing 1, 2 & 3 lines

							*3
				*2 ____________| 
				/              |_______________|
			*1 /

		"""

		(x, y), (w, h) = hand_boxe_resized

		length_line = self.percent_of(w, 4)
		hight_line  = self.percent_of(h, 10)

		deaparture_line = (w, y) # *1
		angle_line = (w + length_line, y - hight_line) # *2
		end_line = (w + int(length_line * 4.5), y - hight_line) # *3

		coords = [deaparture_line, angle_line, end_line]
		[cv2.line(frame, coords[n], coords[n + 1], self.color_mode, 1) for n in range(2)]

		return coords


	def drawing_rectangle_sign_detected(self, frame, virgin, coord, hand_boxe_resized, sign_detected):
		"""
			 *3   _______________
			_____| SIGN DETECTED |
				 |_______________|

		"""

		(x, y), (w, h) = hand_boxe_resized

		margin = self.percent_of(w, 2)

		rect_x, end_y = coord[-1]
		rect_y = end_y - 10
		rect_h = end_y + 10
		rect_w = int(19.3 * 5)


		if not self.have_raise_frame:

			add = int(19.3 * 5)
			rect = (rect_x, rect_y, rect_w, rect_y + 45)
			crop_virgin = virgin[rect_y:rect_y + 45, rect_x:rect_x+add]
			frame[rect_y:rect_y + 45, rect_x:rect_x+add] = crop_virgin

			self.have_raise_frame = True

		crop = frame[rect_y:rect_h, rect_x:rect_x+rect_w]

		# Ajust luminosity in function of the color.
		gamma_choice = 0.6 if self.color_mode == (255, 255, 255) else 1.4
		crop_gamma = self.adjust_gamma(crop, gamma=gamma_choice)

		# Past the area (darkest - lightest) in the frame.
		frame[rect_y:rect_h, rect_x:rect_x+rect_w] = crop_gamma
		cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_h), self.color_mode, 1)

		emplacement_text = (rect_x, rect_y + 13)
		cv2.putText(frame, "SIGN DETECTED", emplacement_text, self.font, 0.4, self.color_mode)

		emplacement_text = (rect_x, rect_y + 35)
		if sign_detected.lower() in self.dico_sign:
			cv2.putText(frame, self.dico_sign[sign_detected.lower()], emplacement_text, self.font, 0.32, self.color_mode)



	def displaying_feature_of_the_hand(self, frame, virgin, features, hand_boxe_resized, mode):
		"""
								  _______________
					  ___________| SIGN DETECTED |
					 /           |_______________|
			 _   _  /            
			|a   b|               def of sign
				   
			|_   _|        *a id of the person.
			c              *b label of the hand (right - left).
						   *c face of the hand.
		"""

		_, _, _, sign_detected = features[2:]

		# Drawing corner around the hand.
		self.drawing_around_the_hand(frame, hand_boxe_resized, features, mode)

		# Sign is detected.
		if sign_detected is not "":
			# Drawing the line to the rectangle.
			coords = self.drawing_lines(frame, hand_boxe_resized)
			# Drawing the rectangle with the sign.
			self.drawing_rectangle_sign_detected(frame, virgin, coords, hand_boxe_resized, sign_detected)


	def hand_mode_display(self, frame, virgin, data_hand, data_hand_analyse, face_id, timer, mode):
		""" """

		for label in ["right", "left"]:

			# Points: [hand landmarks, hand_boxe] Display: [face_id, label, hand_face_side, sign_detected]
			features = self.recuperate_hand_data_interest(data_hand, data_hand_analyse, label, face_id, timer)

			hand_boxe = features[1]
			# If sign's detected modify the display.
			sign_detected = features[-1]
  
			if hand_boxe is not None:

				hand_boxe_resized = self.rectangle_around_the_hand(frame, hand_boxe, sign_detected)
				self.displaying_feature_of_the_hand(frame, virgin, features, hand_boxe_resized, mode)

		self.have_raise_frame = False
