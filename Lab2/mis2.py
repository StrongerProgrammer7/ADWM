import cv2
import numpy as np

'''def nothing(args):pass

# создаем окно для отображения результата и бегунки
cv2.namedWindow("setup")
cv2.createTrackbar("b1", "setup", 0, 255, nothing)
cv2.createTrackbar("g1", "setup", 0, 255, nothing)
cv2.createTrackbar("r1", "setup", 0, 255, nothing)
cv2.createTrackbar("b2", "setup", 255, 255, nothing)
cv2.createTrackbar("g2", "setup", 255, 255, nothing)
cv2.createTrackbar("r2", "setup", 255, 255, nothing)
'''
if __name__ == '__main__':
    url = 0#"http://192.168.43.1:8080/video"
    cap = cv2.VideoCapture(url)

    while True:

        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        img_g = cv2.inRange(frame, (0,193,0),(9,255,255))
        # зададим порог
        thresh = 10

        # получим картинку, обрезанную порогом
        ret, thresh_img = cv2.threshold(img_g, thresh, 255, cv2.THRESH_BINARY)

        # надем контуры
        #contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # создадим пустую картинку
        #img_contours = np.zeros(frame.shape)

        # отобразим контуры
        #cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

        cv2.imshow('IP Webcam Video', thresh_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''
r1 = cv2.getTrackbarPos('r1', 'setup')
        g1 = cv2.getTrackbarPos('g1', 'setup')
        b1 = cv2.getTrackbarPos('b1', 'setup')
        r2 = cv2.getTrackbarPos('r2', 'setup')
        g2 = cv2.getTrackbarPos('g2', 'setup')
        b2 = cv2.getTrackbarPos('b2', 'setup')
        # собираем значения из бегунков в множества
        min_p = (g1, b1, r1)
        max_p = (g2, b2, r2)'''