load('backupDetector.mat');
img = imread('1.jpg');
[bboxes,scores] = detect(detector,img);
if ~isempty(bboxes)
   img = insertObjectAnnotation(img, 'rectangle', bboxes, scores);
else
   img = insertText(img,[10 10],'No Detections');
end
figure
imshow(img)