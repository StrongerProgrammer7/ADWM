%using extra module for Matlab2023 (WEBCAM and RESNET)
webcamlist;
% Создание объекта для чтения с веб-камеры
cam = webcam('Logi'); 

% Создание объекта видеоплеера
videoPlayer = vision.VideoPlayer;
i = 0;
% Цикл для чтения и обработки каждого кадра
while i < 1000
    % Получение кадра с веб-камеры
    frame = snapshot(cam);
    closePreview(cam)
    % Обнаружение рук на кадре
    [bboxes, scores] = detect(detector, frame);

    % Вставка аннотации в видео
    if ~isempty(bboxes)
        frame = insertObjectAnnotation(frame, 'rectangle', bboxes, scores);
    end
    i = i + 1;
    % Отображение кадра с аннотациями
    step(videoPlayer, frame);
    preview(cam)
end

% Освобождение ресурсов
release(videoPlayer);
clear cam;
