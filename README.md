# emotional-marquors human detection

Try to detect emotion marquors via body movements. Project in progress.

![a](https://user-images.githubusercontent.com/54853371/117892402-f2132680-b2b8-11eb-8d19-291852c411aa.png)


How it functions:

    lunch main.py --video {path_of_video}

    (video rules format's in main.py)


Requirements:
  
    - pip install tensorflow==2.4.1

    - pip install keras

    - pip install mediapipe==0.8.2

    - pip install opencv-python==4.5.1.48

    - pip install opencv-contrib-python==4.5.1.48

    - pip install numpy==1.19.3

    - pip install dlib==19.19.0
 
    - pip install face-recognition==1.2.3
 
Downloaw:
  
    - https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
    - And put it in emotional_marquors\data\models\dlib_model

    - https://github.com/eveningglow/age-and-gender-classification/blob/master/model/gender_net.caffemodel
    - And put it in emotional_marquors\data\models\genre

Used:

    - eyes model https://github.com/Chris10M/open-eye-detector
    - emotion : https://github.com/nileshbhadana/emotion_detection
    - color Original Author  Ernesto P. Adorio, Ph.D 
    - blink ear one https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
    - pupil detection https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
    - skin color : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf
    - face recognition https://www.youtube.com/watch?v=54WmrwVWu1w
    - head movement https://www.youtube.com/watch?v=ibuEFfpVWlU&t=661s


We don't know but in case you use it please cite __--XxXDarksMagicienDu26XxX--__
