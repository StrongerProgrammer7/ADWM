vid = VideoReader('hand2.mp4');
detector = vision.CascadeObjectDetector('trained_model5.xml');
while hasFrame(vid)
    vf = readFrame(vid);
    bbox = step(detector,vf);
    detectedImg = insertObjectAnnotation(vf,'rectangle',bbox,'Mathced Found');
    imshow(detectedImg);
end