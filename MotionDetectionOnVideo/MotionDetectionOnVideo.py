import cv2

if __name__ == '__main__':
    input_video_path = '../movies/archive/main.mov'
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("error read movie")
        exit()

    output_video_path = '../movies/example.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(5))
    frame_size = (int(cap.get(3)), int(cap.get(4)))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size,isColor=True)
    if not out.isOpened():
        print("Error opening the output video file.")
        cap.release()
        exit()

    kernelSize = (3,3)
    sigma = 0
    ret, frame = cap.read()
    frame_count = 0
    max_area = 100
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        frame = cv2.GaussianBlur(frame, kernelSize, sigma)
        oldFrame = frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, kernelSize, sigma)

            frame_diff = cv2.absdiff(oldFrame,gray_frame)
            #print(frame_diff)
            thresh = cv2.threshold(frame_diff,25,255,cv2.THRESH_BINARY)[1]
            #print(thresh)
            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            #print(contours)

            for cnt in contours:
                if cv2.contourArea(cnt) > max_area:
                    out.write(frame)
                    #cv2.imshow('IP Webcam Video', frame)
                    cv2.imwrite(f'../movies/screen/motion_detected_{frame_count}.png', frame)
                    frame_count +=1
                    break
            oldFrame = gray_frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
