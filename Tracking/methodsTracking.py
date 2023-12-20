from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mil": cv2.TrackerMIL_create
        }

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

def selectFirstFrame(cap,tracker,frame=None):
    if(frame is None):
        ok, frame = cap.read()

        if ok is None:
            return None
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
    initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
    # start OpenCV object tracker using the supplied bounding box
    # coordinates, then start the FPS throughput estimator as well
    tracker.init(frame, initBB)
    fps = FPS().start()
    return fps,initBB

def calcTrackNextFrame(typeTrack,tracker,frame,fps):
    (H, W) = frame.shape[:2]
    # grab the new bounding box coordinates of the object
    (success, box) = tracker.update(frame)
    # check to see if the tracking was a success
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
    # update the FPS counter
    fps.update()
    fps.stop()
    # initialize the set of information we'll be displaying on
    # the frame
    info = [
        ("Tracker", typeTrack),
        ("Success", "Yes" if success else "No"),
        ("FPS", "{:.2f}".format(fps.fps())),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def runTracker(typeTrack,movieAllPath,saveMovie='',showMovie=True):
    tracker = OPENCV_OBJECT_TRACKERS[typeTrack]()

    cap = cv2.VideoCapture(movieAllPath)
    if(saveMovie!= ''):
        out = prepareSaveMovie(saveMovie,cap)

    fps,initBB = selectFirstFrame(cap,tracker)
    t0 = time.time()
    while True:
        ok, frame = cap.read()

        if ok is None or frame is None:
            break

        if initBB is not None:
            calcTrackNextFrame(typeTrack,tracker,frame,fps)

        if(saveMovie!=''):
            out.write(frame)
        if(showMovie):
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("s") and typeTrack!='kcf':
            fps, initBB = selectFirstFrame(cap, tracker)
        elif cv2.waitKey(1) & 0xFF == ord("q"):
            break

    t1 = time.time()
    print(str(t1-t0))
    if(saveMovie!=''):
        out.release()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    runTracker('mil', 'hand2.mp4')
    runTracker('csrt', 'hand2.mp4')
    runTracker('kcf', 'hand2.mp4')
    runTracker('mil','hand6.mp4')
    runTracker('mil', 'hand3.mp4')

