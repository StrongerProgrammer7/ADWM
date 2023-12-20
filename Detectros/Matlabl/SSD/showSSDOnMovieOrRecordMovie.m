vid = VideoReader('A:\High-level prommaing\Python\ADWM\movies\hand2.mp4');
%outputVideoFile = 'A:\High-level prommaing\Python\ADWM\movies\hands\SSD\hand4.mp4';

% Создание объекта VideoWriter
%outputVideo = VideoWriter(outputVideoFile, 'MPEG-4');
%outputVideo.FrameRate = vid.FrameRate;  % Установка частоты кадров такой же, как у исходного видео
%open(outputVideo);

while hasFrame(vid)
    vf = readFrame(vid);
    [bboxes,scores] = detect(detector,vf);
    if(~isempty(bboxes))
        vf = insertObjectAnnotation(vf,"rectangle",bboxes,scores);
    end
    imshow(vf);
    %writeVideo(outputVideo, vf);
end
%close(outputVideo);