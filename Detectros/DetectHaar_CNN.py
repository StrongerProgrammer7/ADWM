# -*- coding:UTF-8 -*-
import time

import numpy as np
from keras.models import load_model
import cv2


def Identify_hand(window_name,pathToMovie,nameTrainedModel,cascadeHaar,pathRecordMovieWithDetect=''):
    cv2.namedWindow(window_name)

    model = load_model(nameTrainedModel)

    cap = cv2.VideoCapture(pathToMovie)
    classifier = cv2.CascadeClassifier(cascadeHaar)
    color = (0, 255, 0)
    if pathRecordMovieWithDetect !='':
        wb = int(cap.get(3))
        hb = int(cap.get(4))

        output_video_path = 'pathRecordMovieWithDetect'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(5))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (wb, hb), isColor=True)
        if not out.isOpened():
            print("Error opening the output video file.")
            cap.release()
            exit()

    t0 = time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hands = classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=3)

        if len(hands) > 0:
            for hand in hands:
                x, y, w, h = hand

                image = frame[y:y + h, x:x + w]

                #greyI = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(image, (100, 100))
                img = np.expand_dims(resized_frame, axis=0)
                predictions = model.predict(img)
                print(predictions)
                if predictions[0, 0] > 0.98:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w, y + h), color, thickness=2)
                    cv2.putText(frame, 'Hand',
                                (x + 30, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 255),
                                2)
        if pathRecordMovieWithDetect != '':
            out.write(frame)
        t1 = time.time()
        #if t1 - t0 > 880:
           #break
        cv2.imshow(window_name,frame)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break

    if pathRecordMovieWithDetect != '':
        out.release()
    cap.release()
    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    Identify_hand('Movie','../../movies/hand2.mp4',"bestTrainedModel.h5",'./Cascade Haar/default_hand_cascade.xml')
