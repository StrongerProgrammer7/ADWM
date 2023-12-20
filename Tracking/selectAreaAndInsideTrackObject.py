import cv2
import numpy as np


def nothing(args):
    pass

# cv2.namedWindow("setup")
# cv2.createTrackbar("b1", "setup", 0, 255, nothing)
# cv2.createTrackbar("r1", "setup", 0, 255, nothing)
# cv2.createTrackbar("g1", "setup", 0, 255, nothing)
# cv2.createTrackbar("b2", "setup", 255, 255, nothing)
# cv2.createTrackbar("g2", "setup", 255, 255, nothing)
# cv2.createTrackbar("r2", "setup", 255, 255, nothing)

drawing = False # True if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy,ex,ey = -1, -1, -1, -1
percentSize = 50

def draw_rectangle(event, x, y, img, img2):
    global ix, iy, ex,ey,drawing, mode, alpha,endx,endy
    overlay = img.copy()
    output = img.copy()
    alpha = 1

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing == True:
        cv2.rectangle(overlay, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, img2)
        cv2.imshow('Movie', img2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(overlay, (ix, iy), (x, y), (0, 255, 0),2)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, img2)
        ex,ey = x,y

def changeSize(percent,frame):
    scale_percent = percent  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return frame

def getTracker():
    r1 = cv2.getTrackbarPos('r1', 'setup')
    g1 = cv2.getTrackbarPos('g1', 'setup')
    b1 = cv2.getTrackbarPos('b1', 'setup')
    r2 = cv2.getTrackbarPos('r2', 'setup')
    g2 = cv2.getTrackbarPos('g2', 'setup')
    b2 = cv2.getTrackbarPos('b2', 'setup')
    return (g1, b1, r1), (g2, b2, r2)

if __name__ == '__main__':
        nameFile = "hand6"
        url = "../movies/" + nameFile+ ".mp4"  # "http://192.168.43.1:8080/video"
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print("error read movie")
            exit()
        ret, frame = cap.read()

        output_video_path = '../movies/hands/' + nameFile + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(5))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h), isColor=True)

        if not out.isOpened():
            print("Error opening the output video file.")
            cap.release()
            exit()

        cv2.namedWindow("Movie")

        img2 = frame.copy()
        cv2.setMouseCallback('Movie', lambda event, x, y,flags, param,: draw_rectangle(event, x, y, frame, img2))
        while (1):
            cv2.imshow('Movie', img2)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        #print(ix,iy)
        #print(ex,ey)
        while True:
                ret, frame = cap.read()

                if not ret:
                    break
               # frame = changeSize(percentSize,frame)

                #min_p,max_p = getTracker()
                # (0,0,95)
                # (17,120,255)
                if(ex < ix):
                    temp = ex
                    ex = ix
                    ix = temp
                if(ey < iy):
                    temp = ey
                    ey = iy
                    iy = temp
                cv2.rectangle(frame, (ix, iy), (ex, ey), (0, 255, 0), 2)
                cat1 = frame[iy:ey, ix:ex]  # cut image

                #cv2.imshow('CAT', cat1)
                hsv = cv2.cvtColor(cat1, cv2.COLOR_BGR2HSV)
                #cv2.imshow('HSV', hsv)

                # (179,0,12) (179,50,125)
                #3 (14,80,127),(31,121,132)
                #4 (0,108,113),(14,135,231)
                #img_g = cv2.inRange(hsv, min_p, max_p)
                #img_g = cv2.inRange(hsv, (0,108,113),(14,135,231))
                #img_g = cv2.inRange(hsv, (179,0,16),(179,50,245))

                img_g = cv2.inRange(hsv, (0,31,144), (99,116,255))
                kernel = np.ones((3, 3), np.uint8)
                erosion = cv2.erode(img_g, kernel, iterations=1)
                opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

                contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #(0,135,120),(243,208,255)
                #cv2.drawContours(cat1, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)

                moments = cv2.moments(opening, 1)
                dM01 = moments['m01']
                dM10 = moments['m10']
                dArea = moments['m00']

                if dArea > 10:
                        x = int(dM10 / dArea)
                        y = int(dM01 / dArea)
                        center = (x, y)
                        dArea = int(dArea)
                        if (x >= 0 and y >= 0):
                                cv2.line(cat1, (x - 15, y), (x + 15, y), (0, 255, 255), 2, 8, 0)
                                cv2.line(cat1, (x, y - 15), (x, y + 15), (0, 255, 255), 2, 8, 0)
                                x, y, w, h = cv2.boundingRect(opening)
                                cv2.rectangle(cat1, (x, y), (x + w, y + h), (0, 0, 0), thickness=2, lineType=8,
                                              shift=0)

                out.write(frame)
                cv2.imshow('Movie', frame)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
        cap.release()
        out.release()
        cv2.destroyAllWindows()