pos_dir = fullfile('A:\High-level prommaing\Python\ADWM\Detectros\DATASET_TRAIN\DatasetForHaar_p');
addpath(pos_dir);

 
combinedTable.imageFilename = string(combinedTable.imageFilename);
imds = imageDatastore(combinedTable.imageFilename);
blds = boxLabelDatastore(combinedTable(:,2:end));
ds = combine(imds,blds);

baseNetwork = layerGraph(resnet50);

classNames = "Hand";
anchorBoxes = {[30 60; 60 30; 50 50; 100 100], ...
               [40 70; 70 40; 60 60; 120 120]};
           
layersToConnect =  ["activation_22_relu" "activation_40_relu"];

detector = ssdObjectDetector(baseNetwork,classNames,anchorBoxes, ...
           DetectionNetworkSource=layersToConnect);

options = trainingOptions('sgdm', ...
        MiniBatchSize = 4, ....
        InitialLearnRate = 1e-3, ...
        LearnRateSchedule = 'piecewise', ...
        LearnRateDropPeriod = 30, ...
        LearnRateDropFactor =  0.4, ...
        MaxEpochs = 30, ...
        VerboseFrequency = 50, ...        
        CheckpointPath = tempdir, ...
        Shuffle = 'every-epoch', ...
        Plots="training-progress");

[detector,info2] = trainSSDObjectDetector(ds,detector,options);


%--- check detector
[bboxes,scores] = detect(detector,img);
if ~isempty(bboxes)
    img = insertObjectAnnotation(img,'rectangle',bboxes,cellstr(labels));
else
   img = insertText(img,[10 10],'No Detections');
end
figure
imshow(img)
