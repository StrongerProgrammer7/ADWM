import cv2
import numpy as np


if __name__ == '__main__':
    url = "../movies/hand2.mp4"
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        img_g = cv2.inRange(frame, (0, 193, 0), (9, 255, 255))

        kernel = np.ones((20, 20), np.uint8)

        opening = cv2.morphologyEx(img_g, cv2.MORPH_OPEN, kernel)

        cv2.imshow('IP Webcam Video', opening)
        cv2.imshow('Original', img_g)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
