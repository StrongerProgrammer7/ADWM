import cv2
import numpy as np


if __name__ == '__main__':
    url = "0"
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        #(0,135,120)(243,208,255)
        img_g = cv2.inRange(frame, (0,135,120),(243,208,255))

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(img_g, kernel, iterations=1)

        cv2.imshow('IP Webcam Video', erosion)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
