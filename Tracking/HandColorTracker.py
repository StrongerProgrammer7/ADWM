import time

import cv2
import numpy as np
from imutils.video import FPS

drawing = False
mode = True
ix, iy,ex,ey = -1, -1, -1, -1

class HandColorTracker:
    def __init__(self, alpha=0.5, threshold=10):
        self.alpha = alpha # для лучшей адаптации к цвету (ближе к 1 быстрее адапатируется, то есть частые изменения)
        self.threshold = threshold #порогового значения пикселям, значения которых превышают указанное пороговое значение, присваивается стандартное значение
        self.target_hist = None

    def set_target_color(self, frame, bbox):
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Вычисляем и нормализуем гистограмму
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 246], [0, 180, 0, 256])
        #hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [0, 0, 0], [0, 180, 0, 256, 0, 256])
        #cv2.imshow('hist',hist)
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        '''
          const int* channels: список каналов, используемых для вычисления гистограммы. Число каналов изменяется от 0 до 2. 
          •  InputArray	mask: необязательная маска, показывающая, какие пиксели учитывать при вычислении гистограммы. 
          •  OutputArray	hist: результирующая гистограмма. 
          •  int dims: позволяет задать размерность гистограммы. 
          •  const	int*	histSize: массив размеров гистограмм по каждому измерению. 
          •  const	float**	ranges: массив массивов, описывающих границы интервалов гистограммы по каждому измерению. 
          •  bool	uniform=true: по умолчанию этот параметр равен true. Он определяет, является ли гистограмма равномерной.
        '''
        self.target_hist = hist

    def track(self, frame):
        if self.target_hist is None:
            return None

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        #cv2.imshow('fr',hsv_frame)
        # Вычисляем обратную проекцию
        back_proj = cv2.calcBackProject([hsv_frame], [0, 1], self.target_hist,[0, 146, 5, 159], 1) # 7 159 ,209 hand2
        # Применяем порог
        mask = cv2.threshold(back_proj, self.threshold, 255, cv2.THRESH_BINARY)[1]

        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        kernel = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            return x, y, w, h

        return None

def draw_rectangle(event, x, y, img, img2):
    global ix, iy, ex,ey,drawing, mode
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

def prepareSaveMovie(saveMovie,cap):
    output_video_path = saveMovie
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(5))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h), isColor=True)

    if not out.isOpened():
        print("Error opening the output video file.")
        cap.release()
        exit()
    return out

def selectObjectOnFrame(frame,tracker):
    bbox = cv2.selectROI(frame, False)
    # Устанавливаем цвет цели для трекинга
    tracker.set_target_color(frame, bbox)
    return bbox

def fixErrorCoordinateRect(x,y,w,h):
    if x > w:
        temp = x
        x = w
        w = temp
    if y > h:
        temp = y
        y = h
        h = temp
    return x,y,w,h

def drawRectAndText(frame,bbox,fps):
    (H, W) = frame.shape[:2]
    x, y, w, h = bbox
    x, y, w, h = fixErrorCoordinateRect(x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    fps.update()
    fps.stop()
    # initialize the set of information we'll be displaying on
    # the frame
    info = [
        ("Tracker", 'HSHsTrack'),
        ("Success", "Yes"),
        ("FPS", "{:.2f}".format(fps.fps())),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def runTracker(moviePath,saveMove='',showMovie=True,alpha=0.8,threshold=15):
    cap = cv2.VideoCapture(moviePath)
    if not cap.isOpened():
        print("error read movie")
        exit()

    tracker = HandColorTracker(alpha=alpha, threshold=threshold)  # hand1 = 180 hand5=120

    ok, frame = cap.read()
    if not ok:
        print("error read frame")
        exit()

    if(saveMove!=''):
        out = prepareSaveMovie(saveMove, cap)

    bbox = selectObjectOnFrame(frame,tracker)

    fps = FPS().start()

    t0 = time.time()
    while True:
       # cv2.waitKey(50)
        ok, frame = cap.read()
        if not ok:
            break

        bbox = tracker.track(frame)

        if bbox is not None:
            drawRectAndText(frame,bbox,fps)


        if (saveMove != ''):
            out.write(frame)

        if showMovie == True:
            cv2.imshow("Tracking", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    t1 = time.time()
    print(moviePath + ' : ' + str(t1-t0))
    cap.release()
    if (saveMove != ''):
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    runTracker('A:\High-level prommaing\Python\ADWM\movies\hand6.mp4','hadn6.avi',threshold=28)