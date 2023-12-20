% for prepare data using Image labeler or script prepareTableForHaara
positive_ins = resultTable;
pos_dir = fullfile('A:\High-level prommaing\Python\ADWM\Detectros\DATASET_TRAIN\DatasetForHaar_p');
addpath(pos_dir);

neg_dir = fullfile('A:\High-level prommaing\Python\ADWM\Tracking\Detectros\DATASET_TRAIN\negative');

trainCascadeObjectDetector('trained_model5.xml',positive_ins,neg_dir,...
    'NumCascadeStages',16,'FeatureType','Haar');

% 1790 vs. 1843 is a better model. 
% Requires more negative for better result training took 16
% of hours