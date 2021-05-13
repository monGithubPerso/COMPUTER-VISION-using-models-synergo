#!/usr/bin/python3
# -*- coding:utf-8 -*-


import cv2
import numpy as np


class Eyes_tracking:
    """Pupil detection by thresholding via dlib points."""

    def __init__(self):
        """Constructor"""

        # Frame dimensions.
        self.width = 640
        self.height = 360

        # Black picture.
        self.black_frame = np.zeros((self.height, self.width), np.uint8)

        # Eyes points landmarks.
        self.points_eyes = {"right": [36, 37, 38, 39, 40, 41], "left":[42, 43, 44, 45, 46, 47]}


    def rectangle_eye_area(self, img, eye, gray):
        """Recuperate contour of eyes in a box, make an egalizer,
        make a color and gray mask."""

        margin = 5
        x, y, w, h = cv2.boundingRect(np.array(eye))

        cropMask = gray[y-margin : (y+h)+margin, x-margin : (x+w)+margin]
        cropMask = cv2.equalizeHist(cropMask)

        cropImg = img[y-margin : (y+h)+margin, x-margin : (x+w)+margin]

        return cropMask, cropImg, (x-margin, y-margin)


    def eye_contour_masking(self, img, eye, gray):
        """Recuperate contour of eyes points, delimitate that
        recuperate color and gray mask."""

        margin = 5

        mask = np.full((self.height, self.width), 255, np.uint8)
        cv2.fillPoly(mask, [np.array(eye)], (0, 0, 255))
        mask = cv2.bitwise_not(self.black_frame, gray.copy(), mask=mask)

        x, y, w, h = cv2.boundingRect(np.array(eye))

        cropMask = mask[y-margin : y+h+margin, x-margin : x+w+margin]
        cropImg = img[y-margin : y+h+margin, x-margin : x+w+margin]

        return cropMask, cropImg


    def superpose_contour_eye_rectangle(self, mask_eyes_gray, crop):
        """Put gray pixel ( > 200 )
        to whites pixels in the color picture (crop)."""

        for i in range(mask_eyes_gray.shape[0]):
            for j in range(mask_eyes_gray.shape[1]):
                if mask_eyes_gray[i, j] > 200:
                    crop[i, j] = 255

        return crop


    def find_center_pupille_in_mask(self, crop, mask_eyes_img):
        """Gaussian filter, search the max solo contour on thresh,
        make an erod on 3 neighboors, find center of the contours."""

        out = None, None

        if crop is not None:

            gaussian = cv2.GaussianBlur(crop, (9, 9), 0)

            # Search threshold value.
            for thresh in range(0, 200, 5):
                _, threshold = cv2.threshold(gaussian, thresh, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 1:
                    break

            # Filters.
            _, threshold = cv2.threshold(gaussian, thresh - 10, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3,3), np.uint8)
            img_erosion = cv2.erode(threshold, kernel, iterations=1)

            # Recuperate contours.
            contours = cv2.findContours(img_erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            # Recuperate center of the contour.
            if len(contours) > 0:
                a = cv2.moments(contours[0])['m00']
                pupille_center = [(int(cv2.moments(contours[0])['m10']/a),
                                  int(cv2.moments(contours[0])['m01']/a)) for cnt in contours if a > 0]

                if pupille_center != []:
                    x_center, y_center = pupille_center[0][0], pupille_center[0][1]
                    out = x_center, y_center

        return out



    def eyes_landmarks(self, faces, frame, gray):
        """Recuperate the pupil center"""

        eyes_landmark = []
        for landmarks in faces:

            eyes_points = (cv2.convexHull(np.array([landmarks[36: 42]])),
                           cv2.convexHull(np.array([landmarks[42: 48]])))

            eyes_detected = []

            for eye in eyes_points:

                #Box egalized eyes areas.
                gray_crop, color_crop, coordinate = self.rectangle_eye_area(frame, eye, gray)

                #Contours of the broder of the eyes.
                mask_eyes_gray, mask_eyes_img = self.eye_contour_masking(frame, eye, gray)

                #Superpose box and contours.
                gray_crop = self.superpose_contour_eye_rectangle(mask_eyes_gray, gray_crop)

                #Define centers of pupils.
                x_center, y_center = self.find_center_pupille_in_mask(gray_crop, mask_eyes_img)

                try:
                    x, y = coordinate
                    x_center = x + x_center
                    y_center = y + y_center

                    eyes_detected += [(x_center, y_center)]

                except:
                    eyes_detected += [(None, None)]

            right_eye, left_eye = eyes_detected
            if None not in right_eye and None not in left_eye:
                eyes_landmark += [(right_eye, left_eye)]


        return eyes_landmark


    def eyes_association(self, face_person, eye_person, eyes):
        """Associate eye to face."""

        for face_id, data_face in face_person.items():

            is_detected = data_face["is_detected"]

            if is_detected:

                # Recuperate landmarks.
                landmarks_face = data_face["face_landmarks"]

                right_eye_landmarks = [landmarks_face[points] for points in self.points_eyes["right"]]

                x, y, w, h = cv2.boundingRect(np.array(right_eye_landmarks))

                for (right_eye, left_eye) in eyes:

                    # Right pupil in boxe of right eye face.
                    eyeX, eyeY = right_eye
                    if x <= eyeX <= x + w and y <= eyeY <= y + h:
                        eye_person[face_id]["eyes"] = (right_eye, left_eye)
                        break
