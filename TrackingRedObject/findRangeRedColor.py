import cv2


'''def nothing(args):pass

# create a window for displaying the result and sliders
cv2.namedWindow("setup")
cv2.createTrackbar("b1", "setup", 0, 255, nothing)
cv2.createTrackbar("g1", "setup", 0, 255, nothing)
cv2.createTrackbar("r1", "setup", 0, 255, nothing)
cv2.createTrackbar("b2", "setup", 255, 255, nothing)
cv2.createTrackbar("g2", "setup", 255, 255, nothing)
cv2.createTrackbar("r2", "setup", 255, 255, nothing)
'''

if __name__ == '__main__':
    url = 0
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        '''
                r1 = cv2.getTrackbarPos('r1', 'setup')
                g1 = cv2.getTrackbarPos('g1', 'setup')
                b1 = cv2.getTrackbarPos('b1', 'setup')
                r2 = cv2.getTrackbarPos('r2', 'setup')
                g2 = cv2.getTrackbarPos('g2', 'setup')
                b2 = cv2.getTrackbarPos('b2', 'setup')
                # собираем значения из бегунков в множества
                min_p = (g1, b1, r1)
                max_p = (g2, b2, r2)
                img_g = cv2.inRange(frame, min_p,max_p)
        '''
        img_g = cv2.inRange(frame, (0,193,0),(9,255,255))
        thresh = 10

        # we'll get a picture cropped by the threshold
        ret, thresh_img = cv2.threshold(img_g, thresh, 255, cv2.THRESH_BINARY)

        # find and show contours
        #contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #img_contours = np.zeros(frame.shape)
        #cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

        cv2.imshow('IP Webcam Video', thresh_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

