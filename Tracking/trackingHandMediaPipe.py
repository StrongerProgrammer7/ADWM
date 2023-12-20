import cv2
import mediapipe as mp


class handTracker():
    def __init__(self, mode=False, maxHands=8, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon # порог достоверности для обнаружения рук
        self.modelComplex = modelComplexity #сложность модели, целое значение
        self.trackCon = trackCon #порог достоверности для отслеживания рук
        self.mpHands = mp.solutions.hands #из библиотеки взяли модель рук
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                #print(lmlist[len(lmlist)-1])
           # if draw:
             #   cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist

def writeVideo(addressFile,nameFileForWrite):
    cap = cv2.VideoCapture(addressFile)
    if not cap.isOpened():
        print("error read movie")
        exit()
    print('../movies/hands/mediaPipe/'+nameFileForWrite+'.avi')
    output_video_path = '../movies/hands/mediaPipe/'+nameFileForWrite+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(5))
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=True)
    if not out.isOpened():
        print("Error opening the output video file.")
        cap.release()
        exit()

    tracker = handTracker()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        success, image = cap.read()
        if not success:
            break
        image = tracker.handsFinder(image)
        #lmList = tracker.positionFinder(image)
        # if len(lmList) != 0:
        # print(lmList[4])
        out.write(image)
       # cv2.imshow("Video", image)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture("../movies/hand5.mp4")
    tracker = handTracker()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        success, image = cap.read()
        if not success:
            break
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        #if len(lmList) != 0:
           # print(lmList[4])

        cv2.imshow("Video", image)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    writeVideo('../movies/hand5.mp4','out_hand5')
    writeVideo('../movies/hand1.mp4', 'out_hand1')
    writeVideo('../movies/hand2.mp4', 'out_hand2')
    writeVideo('../movies/hand3.mp4', 'out_hand3')
    writeVideo('../movies/hand4.mov', 'out_hand4')


