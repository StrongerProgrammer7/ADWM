import cv2

if __name__ == '__main__':
    url = "http://192.168.43.1:8080/video"
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('IP Webcam Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

