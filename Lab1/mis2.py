import cv2

def showMovie(MODE,WIDTH,HEIGHT,COLOR,FLIP):
    cap = cv2.VideoCapture(r'../movies/SampleVideo_1280x720_2mb.mp4', MODE)
    #fps = int(cap.get(cv2.CAP_PROP_FPS))
    #if WIDTH != None and HEIGHT != None:
     #   cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
     #   cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)
    ret, frame = cap.read()
    while True:
        if not ret:
            break
        color = cv2.cvtColor(frame, COLOR)
        if FLIP == True:
            frame = cv2.flip(frame,0)
        if WIDTH is not None and HEIGHT is not None:
            resize = cv2.resize(frame,(WIDTH,HEIGHT))
            cv2.imshow('frame', resize)
        else:
            cv2.imshow('frame', color)
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == '__main__':
    #showMovie(cv2.CAP_ANY,None,None,cv2.COLOR_BGRA2GRAY,False)
    #showMovie(cv2.CAP_ANY, 640, 480,None,False)
    #showMovie(cv2.CAP_ANY, 640, 480, None, True)
    showMovie(cv2.CAP_ANY, None, None, cv2.COLOR_BGR2HSV, False)