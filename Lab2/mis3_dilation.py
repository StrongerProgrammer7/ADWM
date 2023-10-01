import cv2
import numpy as np


if __name__ == '__main__':
    url = 0#"http://192.168.43.1:8080/video"
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        scale_percent = 80  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        img_g = cv2.inRange(frame, (0, 193, 0), (9, 255, 255))

        kernel = np.ones((20, 20), np.uint8)
        erosion = cv2.erode(img_g, kernel, iterations=1)
        dilation = cv2.dilate(img_g,kernel,iterations = 1)
        after_ers = cv2.dilate(erosion,kernel,iterations = 1)

        cv2.imshow('Eros', erosion)
        cv2.imshow('Dil', dilation)
        cv2.imshow('Eros & Dil', after_ers)
        cv2.imshow('Original', img_g)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
