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

       # cv2.drawContours(frame, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)

        moments = cv2.moments(img_g, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea > 10:
            x = int(dM10 / dArea)
            y = int(dM01 / dArea)
            center = (x,y)
            dArea = int(dArea)
            if(x>=0 and y >=0):
                #cv2.circle(frame, center, 5, (0,255,255), 1)
                cv2.line(frame, (x-15,y), (x + 15, y), (0, 255, 255), 2, 8, 0)
                cv2.line(frame,(x,y-15),(x,y+15),(0,255,255),2,8,0)
                x,y,w,h = cv2.boundingRect(img_g)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),thickness=2,lineType=8,shift=0)

        cv2.imshow('IP Webcam Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

