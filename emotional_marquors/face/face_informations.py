#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""face info"""


import os


import cv2
import numpy as np

import tensorflow.compat.v1 as tf

from keras.utils.data_utils import get_file
from keras.models import Model, load_model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from tensorflow.keras.models import Sequential as SeqTf
from tensorflow.keras.layers import Dense as DenseTf
from tensorflow.keras.layers import Flatten as FlattenTf
from tensorflow.keras.layers import Dropout as DropoutTf
from tensorflow.keras.layers import Conv2D as C2D
from tensorflow.keras.optimizers import Adam as ADAM
from tensorflow.keras.layers import MaxPooling2D as MAXP
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from utils.function_utils import Utils


def resize_pad_skin_color(img, size):
    """Add marge to img"""

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
  
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left,
                                    pad_right, cv2.BORDER_CONSTANT, 0)

    return scaled_img


def load_graph_skin_color(model_file):
    """Lunch a Tf session"""

    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def model_emotion(path_emotion):
    """Couche of emotion model"""

    model = SeqTf()

    model.add(C2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(C2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MAXP(pool_size=(2, 2)))
    model.add(DropoutTf(0.25))

    model.add(C2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MAXP(pool_size=(2, 2)))
    model.add(C2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MAXP(pool_size=(2, 2)))
    model.add(DropoutTf(0.25))

    model.add(FlattenTf())
    model.add(DenseTf(1024, activation='relu'))
    model.add(DropoutTf(0.5))
    model.add(DenseTf(7, activation='softmax'))

    model.load_weights(path_emotion)
    #graph = tf.get_default_graph()

    return model




class Face_information(Utils):
    """Recovery data on the face as the emotion, the skin color & the gender."""

    def __init__(self, path_emotion1, path_emotion2, path_skin_color,
                 path_skin_color_txt, genderModel, genderProto):

        """Constructor"""

        # Emotion
        self.model_emotion_1 = model_emotion(path_emotion1)
        self.label_emotion_1 = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                                4: "Neutral", 5: "Sad", 6: "Surprised"}

        self.model_emotion_2 = load_model(path_emotion2, compile=False)
        self.label_emotion_2 = ["Angry", "Disgusted", "Fearful",
                                "Happy", "Sad", "Surprised", "Neutral"]

        # Gender
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        # Skin color
        self.graph = load_graph_skin_color(path_skin_color)
        self.labels = [line.rstrip() for line in open(path_skin_color_txt, "r", encoding='cp1251')]

        input_name = "import/input_1"
        output_name = "import/softmax/Softmax"
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

        self.sess = tf.Session(graph=self.graph)
        self.sess.graph.finalize()

        self.class_object_data_hand = {}
        self.class_object_data_face = {}
        self.timer = 0

        # Min confidence for gender detection.
        self.prediction_threshold_gender = 0.8


    def getter_data_hand(self, data_hand):
        """Getter hand data."""
        self.class_object_data_hand = data_hand

    def getter_data_face(self, data_face):
        """Getter face data"""
        self.class_object_data_face = data_face

    def getter_timer(self, timer):
        """Getter time in video"""
        self.timer = timer


    def raise_data(self):
        """Raise data"""
        self.class_object_data_hand = {}
        self.class_object_data_face = {}


    @staticmethod
    def crop_out_border(frame, boxe):
        """Verify crop of the head isn't out the frame.
        Else put the frame dimension. For example of crop's: (-100, 10, 10, 10)
        pass to (0, 10, 10, 10)."""

        x, y, w, h = boxe

        height, width = frame.shape[:2]

        cropY = y if y >= 0 else 0
        cropH = h if h <= height else height
        cropX = x if x >= 0 else 0
        cropW = w if w <= width else width

        return cropX, cropW, cropY, cropH


    def getter_crops_for_first_model(self, gray, face_boxe):
        """Treatment of the picture for the model."""

        x, y, w, h = face_boxe

        # Delimitate crop & verify isn't out of the frame.
        cropX, cropW, cropY, cropH = self.crop_out_border(gray, boxe=(x, y, x + w, y + h))
        # Gray crop.
        roi_gray = gray[cropY:cropH, cropX:cropW]
        # Model emotion detection 1.
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        return roi_gray, cropped_img


    def getter_crops_for_second_model(self, roi_gray):
        """Treatment of the picture for the model."""
  
        # Model emotion detection 2.
        roi = cv2.resize(roi_gray, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        return roi


    def emotion_detection(self, gray):
        """Recuperate emotion with two models. If they give the same emotion
        save it else save neutral."""

        face_boxe = self.class_object_data_face["face_box"]

        if face_boxe is not None:
            try:
                # Getter face area in the frame.
                roi_gray, cropped_img = self.getter_crops_for_first_model(gray, face_boxe)

                # Get detection of the first model.
                prediction = self.model_emotion_1.predict(cropped_img)
                # Get the highter probability.
                maxIndex = int(np.argmax(prediction))
                # Get the label corresponding at the probability.
                detected1 = self.label_emotion_1[maxIndex]

                # Getter face area in the frame.
                roi = self.getter_crops_for_second_model(roi_gray)

                # Get detection of the second model.
                preds = self.model_emotion_2.predict(roi)[0]
                # Get the highter probability.
                emotion_probability = np.max(preds)
                # Get the label corresponding at the probability.
                detected2 = self.label_emotion_2[preds.argmax()]

                # If the two models has the same emotion detected save it.
                # Else put to neutral.
                self.class_object_data_face["emotions"] = detected1 if detected1 == detected2 else "Neutral"
            except:
                pass



    def recuperate_crop_for_gender(self, virgin_frame):
        """Recuperate the crop of the face with a margin of 20 pixels."""
 
        x, y, w, h = self.class_object_data_face["face_box"]
        boxe = (x - 20, y - 20, x + w + 20, y + h + 20)

        cropX, cropW, cropY, cropH = self.crop_out_border(virgin_frame, boxe)
        crop = virgin_frame[cropY:cropH, cropX:cropW]

        return crop


    def gender_detection(self, virgin_frame):
        """If data gender < 10.recuperate prediction."""

        gender_list = self.class_object_data_face["gender"]
        boxe_face = self.class_object_data_face["face_box"]

        there_isnt_data_yet = len(gender_list) <= 10
        there_is_face = boxe_face is not None

        if there_isnt_data_yet and there_is_face:
            try:
                crop = self.recuperate_crop_for_gender(virgin_frame)

                # Dnn blob
                blob=cv2.dnn.blobFromImage(crop, 1.0, (227, 227), self.MODEL_MEAN_VALUES, False)
                self.genderNet.setInput(blob)
                genderPreds = self.genderNet.forward().tolist()[0]

                # Prediction.
                prediction = max(genderPreds)
                if prediction > self.prediction_threshold_gender:
                    gender_list += [genderPreds.index(prediction)]
            except:
                pass


    def skin_color(self, frame):
        """Call skin predictor model on the first appear on the frame"""

        skin_color = self.class_object_data_face["skin_color"]
        x, y, w, h = self.class_object_data_face["face_box"]

        if skin_color is None:
            
            try:

                img = frame[y:y+h, x:x+w][:, :, ::-1]
                img = resize_pad_skin_color(img, (224, 224))

                # Add a forth dimension since Tensorflow expects a list of images
                img = np.expand_dims(img, axis=0)

                # Scale the input image to the range used in the trained network
                img = img.astype(np.float32)
                img /= 127.5
                img -= 1.

                results = self.sess.run(self.output_operation.outputs[0], {
                    self.input_operation.outputs[0]: img
                })
                results = np.squeeze(results)

                top_indices = results.argsort()[-1:][::-1]
                classes = [{"race": self.labels[ix], "prob": str(results[ix])} for ix in top_indices]

                self.class_object_data_face["skin_color"] = classes[0]['race']

            except:
                pass
