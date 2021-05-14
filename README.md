# emotional-marquors human detection

Try to detect emotion marquors via body movements. Project in progress.

![a](https://user-images.githubusercontent.com/54853371/117892402-f2132680-b2b8-11eb-8d19-291852c411aa.png)


How it functions:

    In main verify path.

    lunch main.py --video {path_of_video}

    (video rules format's in main.py)


Requirements:
  
   1) conda install python=3.7.6
   2) pip install opencv-python
 
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
    
    TUTO used:

    - https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
    - https://www.pyimagesearch.com/
    - https://www.youtube.com/channel/UCYUjYU5FveRAscQ8V21w81A
    - https://www.youtube.com/channel/UC5hHNks012Ca2o_MPLRUuJw
    - https://www.youtube.com/channel/UCwlhFburhQNOsfgeGOyRujg


