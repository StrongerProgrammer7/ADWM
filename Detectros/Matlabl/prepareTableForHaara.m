
imageFolder = 'A:\High-level prommaing\Python\ADWM\Tracking\Detectros\DATASET_TRAIN\DatasetForHaar_p';

imageFiles = dir(fullfile(imageFolder, '*.jpg'));
imageFiles = [imageFiles; dir(fullfile(imageFolder, '*.png'))];
len = length(imageFiles);%1000;

dt = table('Size', [len, 2], 'VariableTypes', {'char', 'cell'}, 'VariableNames', {'imageFilename', 'Hand'});
ind = 0;

for i = 1:length(imageFiles)
    if (ind == len)
        break
    end
    ind = ind + 1;   

    imagePath = fullfile(imageFolder, imageFiles(i).name);
    
    img = imread(imagePath);
    
    height = size(img, 1);
    width = size(img, 2);
    
    dt.imageFilename{i} = imagePath;
    dt.Hand(i) = {[10,10,width-10,height-10]}; 
end

% Отображение таблицы с результатами
%disp(dataTable2);
