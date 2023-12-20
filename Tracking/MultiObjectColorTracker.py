import cv2
import numpy as np

class MultiObjectColorTracker:
    def __init__(self, alpha=0.5, threshold=30):
        self.alpha = alpha
        self.threshold = threshold
        self.target_hist1 = None
        self.target_hist2 = None

    def set_target_color(self, frame, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Извлекаем ROI для первого объекта
        roi1 = frame[y1:y1+h1, x1:x1+w1]
        # Преобразуем BGR в HSV
        hsv_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
        # Вычисляем и нормализуем гистограмму для первого объекта
        hist1 = cv2.calcHist([hsv_roi1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist1 = cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)
        self.target_hist1 = hist1

        # Извлекаем ROI для второго объекта
        roi2 = frame[y2:y2+h2, x2:x2+w2]
        # Преобразуем BGR в HSV
        hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
        # Вычисляем и нормализуем гистограмму для второго объекта
        hist2 = cv2.calcHist([hsv_roi2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)
        self.target_hist2 = hist2

    def track(self, frame):
        if self.target_hist1 is None or self.target_hist2 is None:
            return None, None

        # Преобразуем BGR в HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Вычисляем обратную проекцию для первого объекта
        back_proj1 = cv2.calcBackProject([hsv_frame], [0, 1], self.target_hist1, [0, 180, 0, 256], 1)
        # Бинаризация для первого объекта
        _, mask1 = cv2.threshold(back_proj1, self.threshold, 255, cv2.THRESH_BINARY)

        # Вычисляем обратную проекцию для второго объекта
        back_proj2 = cv2.calcBackProject([hsv_frame], [0, 1], self.target_hist2, [0, 180, 0, 256], 1)
        # Бинаризация для второго объекта
        _, mask2 = cv2.threshold(back_proj2, self.threshold, 255, cv2.THRESH_BINARY)

        mask1 = cv2.GaussianBlur(mask1, (5, 5), 0)

        # Устранение шумов и замыкание контуров
        kernel = np.ones((5, 5), np.uint8)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

        # Находим контуры
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cont1 = None
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cont1 =(x, y, w, h)

        mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)

        # Устранение шумов и замыкание контуров
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

        # Находим контуры
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cont2 = None
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cont2 = (x, y, w, h)

        return cont1, cont2

if __name__ == '__main__':
    cap = cv2.VideoCapture("hand2.mp4")
    if not cap.isOpened():
        print("error read movie")
        exit()

    tracker = MultiObjectColorTracker(alpha=0.8, threshold=10) #hand1 = 180 hand5=120

    ok, frame = cap.read()
    if not ok:
        print("error read frame")
        exit()

    bbox1 = cv2.selectROI(frame, False)
    bbox2 = cv2.selectROI(frame, False)

    tracker.set_target_color(frame, bbox1, bbox2)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bbox1, bbox2 = tracker.track(frame)
        print(bbox1)
        if bbox1 is not None:
            x, y, w, h = bbox1
            if x > w:
                temp = x
                x = w
                w = temp
            if y > h:
                temp = y
                y = h
                h = temp
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if bbox2 is not None:
            x, y, w, h = bbox2
            if x > w:
                temp = x
                x = w
                w = temp
            if y > h:
                temp = y
                y = h
                h = temp
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 125), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()