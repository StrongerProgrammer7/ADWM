import cv2
import numpy as np
def nothing(args):
    pass
def step(args):
    print(args)
    if args % 2 == 0:
        args+=1

cv2.namedWindow("setup")
cv2.namedWindow("histSet")
cv2.createTrackbar("b1", "setup", 0, 255, nothing)
cv2.createTrackbar("r1", "setup", 0, 255, nothing)
cv2.createTrackbar("g1", "setup", 0, 255, nothing)
cv2.createTrackbar("b2", "setup", 255, 255, nothing)
cv2.createTrackbar("g2", "setup", 255, 255, nothing)
cv2.createTrackbar("r2", "setup", 255, 255, nothing)

cv2.createTrackbar("hs", "histSet", 0, 510, nothing)
cv2.createTrackbar("he", "histSet", 146, 512, nothing)
cv2.createTrackbar("ss", "histSet", 5, 510, nothing)
cv2.createTrackbar("se", "histSet", 259, 512, nothing)

cv2.createTrackbar("HBhs", "histSet", 0, 510, nothing)
cv2.createTrackbar("HBhe", "histSet", 149, 512, nothing)
cv2.createTrackbar("HBss", "histSet", 5, 510, nothing)
cv2.createTrackbar("HBse", "histSet", 249, 512, nothing)
cv2.createTrackbar("gauss", "histSet", 3, 11, step)

def setRangeForColor(pathMovie):
    url = pathMovie  # "http://192.168.43.1:8080/video"
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)

        r1 = cv2.getTrackbarPos('r1', 'setup')
        g1 = cv2.getTrackbarPos('g1', 'setup')
        b1 = cv2.getTrackbarPos('b1', 'setup')
        r2 = cv2.getTrackbarPos('r2', 'setup')
        g2 = cv2.getTrackbarPos('g2', 'setup')
        b2 = cv2.getTrackbarPos('b2', 'setup')
        min_p = (g1, b1, r1)
        max_p = (g2, b2, r2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        img_g = cv2.inRange(frame, min_p, max_p)

        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(img_g, kernel, iterations=1)
        # closing = cv2.morphologyEx(img_g, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

        cv2.imshow('IP Webcam Video', opening)
        cv2.imshow('Original', img_g)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def setRangeForHistogram(pathImg):
    img = cv2.imread(pathImg, 1)
    imgC = img.copy()
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    while True:
        hs = cv2.getTrackbarPos("hs", "histSet")
        he = cv2.getTrackbarPos("he", "histSet")
        ss = cv2.getTrackbarPos("ss", "histSet")
        se = cv2.getTrackbarPos("se", "histSet")

        HBhs = cv2.getTrackbarPos("HBhs", "histSet")
        HBhe = cv2.getTrackbarPos("HBhe", "histSet")
        HBss = cv2.getTrackbarPos("HBss", "histSet")
        HBse = cv2.getTrackbarPos("HBse", "histSet")

        gaus = cv2.getTrackbarPos('gauss', 'histSet')
        if HBhe == 0:
            HBse = 180
        if HBse == 0:
            HBse = 250
        histB = cv2.calcHist([frame], [0, 1], None, [180, 246], [HBhs, HBhe, HBss, HBse])
        # hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [0, 0, 0], [0, 180, 0, 256, 0, 256])
        # cv2.imshow('hist',hist)
        histB = cv2.normalize(histB, histB, 0, 255, cv2.NORM_MINMAX)

        if gaus % 2 == 0:
            gaus += 1
        if he == 0:
            he = 180
        if se == 0:
            se = 180

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hist = cv2.calcHist([frame], [0, 1], None, [180, 246], [hs, he, ss, se])
        # hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        back_proj = cv2.calcBackProject([frame], [0, 1], histB, [hs, he, ss, se], 1)
        mask = cv2.threshold(back_proj, 15, 255, cv2.THRESH_BINARY)[1]

        mask = cv2.GaussianBlur(mask, (gaus, gaus), 0)

        kernel = np.ones((gaus, gaus), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(imgC, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)

        cv2.imshow('Original', back_proj)
        cv2.imshow('HSV', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('Countour', imgC)
        imgC = img.copy()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
if __name__ == '__main__':
    setRangeForHistogram('1.png')
