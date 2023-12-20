import cv2
import numpy as np
import sys

if __name__ == '__main__':
    url = 0
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        img_g = cv2.inRange(hsv, (0, 193, 0), (9, 255, 255))

        contours, hierarchy = cv2.findContours(img_g.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)

        moments = cv2.moments(img_g, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea > 1000:
            x = int(dM10 / dArea)
            y = int(dM01 / dArea)
            dArea = int(dArea)
            if(x>=0 and y >=0):
                cv2.putText(frame, "Area: %d" % dArea, (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2,cv2.LINE_AA)

        cv2.imshow('IP Webcam Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
