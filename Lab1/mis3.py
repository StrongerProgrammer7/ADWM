import cv2

if __name__ == '__main__':
    input_video_path = '../movies/SampleVideo_1280x720_2mb.mp4'
    cap = cv2.VideoCapture(input_video_path)

    output_video_path = '../movies/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Формат кодека видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Кадров в секунду
    frame_size = (int(cap.get(3)), int(cap.get(4)))  # Размер кадра

    # Создаем объект VideoWriter
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Закрываем объекты VideoCapture и VideoWriter
    cap.release()
    out.release()

    cv2.destroyAllWindows()
