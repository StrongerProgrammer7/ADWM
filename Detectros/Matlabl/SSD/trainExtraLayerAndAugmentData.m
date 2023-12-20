
%Data = load('Data2.mat').Data;
rng(0);
shuffledIndices = randperm(height(Data));
idx = floor(0.8 * length(shuffledIndices) );
trainingData = Data(shuffledIndices(1:idx),:);
testData = Data(shuffledIndices(idx+1:end),:);

imdsTrain = imageDatastore(trainingData{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingData(:,'Hand'));

imdsTest = imageDatastore(testData{:,'imageFilename'});
bldsTest = boxLabelDatastore(testData(:,'Hand'));

trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest, bldsTest);

%data = read(trainingData);
%I = data{1};
%bbox = data{2};
%annotatedImage = insertShape(I,'rectangle',bbox);
%annotatedImage = imresize(annotatedImage,2);
%figure
%imshow(annotatedImage)


net = resnet50();
lgraph = layerGraph(net);

inputSize = [300 300 3];

classNames = "Hand";

% Find layer index of 'activation_40_relu'
idx = find(ismember({lgraph.Layers.Name},'activation_40_relu'));

% Remove all layers after 'activation_40_relu' layer
removedLayers = {lgraph.Layers(idx+1:end).Name};
ssdLayerGraph = removeLayers(lgraph,removedLayers);

weightsInitializerValue = 'glorot';
biasInitializerValue = 'zeros';

% Append Extra layers on top of a base network.
extraLayers = [];

% Add conv6_1 and corresponding reLU
filterSize = 1;
numFilters = 256;
numChannels = 1024;
conv6_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv6_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu6_1 = reluLayer(Name = 'relu6_1');
extraLayers = [extraLayers; conv6_1; relu6_1];

% Add conv6_2 and corresponding reLU
filterSize = 3;
numFilters = 512;
numChannels = 256;
conv6_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Stride = [2, 2], ...
    Name = 'conv6_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu6_2 = reluLayer(Name = 'relu6_2');
extraLayers = [extraLayers; conv6_2; relu6_2];

% Add conv7_1 and corresponding reLU
filterSize = 1;
numFilters = 128;
numChannels = 512;
conv7_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv7_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu7_1 = reluLayer(Name = 'relu7_1');
extraLayers = [extraLayers; conv7_1; relu7_1];

% Add conv7_2 and corresponding reLU
filterSize = 3;
numFilters = 256;
numChannels = 128;
conv7_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Stride = [2, 2], ...
    Name = 'conv7_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu7_2 = reluLayer(Name = 'relu7_2');
extraLayers = [extraLayers; conv7_2; relu7_2];

% Add conv8_1 and corresponding reLU
filterSize = 1;
numFilters = 128;
numChannels = 256;
conv8_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv8_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu8_1 = reluLayer(Name = 'relu8_1');
extraLayers = [extraLayers; conv8_1; relu8_1];

% Add conv8_2 and corresponding reLU
filterSize = 3;
numFilters = 256;
numChannels = 128;
conv8_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv8_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu8_2 = reluLayer(Name ='relu8_2');
extraLayers = [extraLayers; conv8_2; relu8_2];

% Add conv9_1 and corresponding reLU
filterSize = 1;
numFilters = 128;
numChannels = 256;
conv9_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Name = 'conv9_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu9_1 = reluLayer('Name', 'relu9_1');
extraLayers = [extraLayers; conv9_1; relu9_1];

if ~isempty(extraLayers)
    lastLayerName = ssdLayerGraph.Layers(end).Name;
    ssdLayerGraph = addLayers(ssdLayerGraph, extraLayers);
    ssdLayerGraph = connectLayers(ssdLayerGraph, lastLayerName, extraLayers(1).Name);
end
detNetworkSource = ["activation_22_relu", "activation_40_relu", "relu6_2", "relu7_2", "relu8_2"];

anchorBoxes = {[60,30;30,60;60,21;42,30];...
               [111,60;60,111;111,35;64,60;111,42;78,60];...
               [162,111;111,162;162,64;94,111;162,78;115,111];...
               [213,162;162,213;213,94;123,162;213,115;151,162];...
               [264,213;213,264;264,151;187,213]};

detector = ssdObjectDetector(ssdLayerGraph,classNames,anchorBoxes,DetectionNetworkSource=detNetworkSource,InputSize=inputSize,ModelName='ssdHand'); 

augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},rectangle = data{2});
    reset(augmentedTrainingData);
end

%figure
%montage(augmentedData,BorderSize = 10)
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);
%I = data{1};
%bbox = data{2};
%annotatedImage = insertShape(I,'rectangle',bbox);
%annotatedImage = imresize(annotatedImage,2);
%figure
%imshow(annotatedImage)

options = trainingOptions('sgdm', ...
        MiniBatchSize = 4, ....
        InitialLearnRate = 1e-4, ...
        LearnRateSchedule = 'piecewise', ...
        LearnRateDropPeriod = 30, ...
        LearnRateDropFactor =  0.8, ...
        MaxEpochs = 40, ...
        VerboseFrequency = 50, ...        
        CheckpointPath = tempdir, ...
        Shuffle = 'every-epoch', ...
        ExecutionEnvironment = 'auto', ...
        Plots="training-progress");

[detector, info] = trainSSDObjectDetector(preprocessedTrainingData,detector,options);
function B = augmentData(A)
    % Apply random horizontal flipping, and random X/Y scaling. Boxes that get
    % scaled outside the bounds are clipped if the overlap is above 0.25. Also,
    % jitter image color.
    B = cell(size(A));
    I = A{1};
    sz = size(I);
    if numel(sz)==3 && sz(3) == 3
        I = jitterColorHSV(I,...
            Contrast = 0.2,...
            Hue = 0,...
            Saturation = 0.1,...
            Brightness = 0.2);
    end
    % Randomly flip and scale image.
    tform = randomAffine2d(XReflection = true, Scale = [1 1.1]);  
    rout = affineOutputView(sz,tform, BoundsStyle = 'CenterOutput');    
    B{1} = imwarp(I,tform,OutputView = rout);
    % Sanitize boxes, if needed. This helper function is attached as a
    % supporting file. Open the example in MATLAB to access this function.
    %A{2} = helperSanitizeBoxes(A{2});    
    % Apply same transform to boxes.
    [B{2},indices] = bboxwarp(A{2},tform,rout,OverlapThreshold = 0.25);    
    B{3} = A{3}(indices);  
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        B = A;
    end
end

function data = preprocessData(data,targetSize)
    % Resize image and bounding boxes to the targetSize.
    sz = size(data{1},[1 2]);
    scale = targetSize(1:2)./sz;
    data{1} = imresize(data{1},targetSize(1:2));
    % Sanitize boxes, if needed. This helper function is attached as a
    % supporting file. Open the example in MATLAB to access this function.
    %data{2} = helperSanitizeBoxes(data{2});
    % Resize boxes.
    data{2} = bboxresize(data{2},scale);
end

function p = iSamePadding(FilterSize)
    p = floor(FilterSize / 2);
end


