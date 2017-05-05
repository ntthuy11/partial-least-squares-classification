close all;
warning off;
addpath('E:\\_Thuy\code\Matlab functions\toolbox\stats');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% BIOFILM DATA
excelFile       = 'E:\\_Thuy\\project\\01-BioFilm\\paper\\1-for biofilm data\\130925\\714 samples 143x9 features.xlsx';
sheetNo         = 1;
nColsFrom       = 1;
nCols           = 1287;
groupColIdx     = nCols + 1;
nComponents     = 60;

% MAMMOGRAPHIC MASS DATA
% excelFile       = 'E:\\_Thuy\\project\\01-BioFilm\\paper\\2-handling missing data\\code\\mammographic.data\\4-mammographic.data (predicted missing values).xlsx';
% sheetNo         = 1;
% nColsFrom       = 2;
% nCols           = 5;
% groupColIdx     = nCols + 1;
% nComponents     = 5;

% PIMA INDIAN DIABETES DATA
% excelFile       = 'E:\\_Thuy\\project\\01-BioFilm\\paper\\2-handling missing data\\code\\pima.data\\4-pima.data (predicted missing values).xlsx';
% sheetNo         = 1;
% nColsFrom       = 2;
% nCols           = 8;
% groupColIdx     = nCols + 1;
% nComponents     = 5;


% load and filter data
if (~exist('dataSet', 'var'))
    [dataSet, dataTxt, ~] = xlsread(excelFile, sheetNo);
    dataX                 = dataSet(:, nColsFrom:nColsFrom+nCols-1);
    dataY                 = dataSet(:, nColsFrom+groupColIdx-1);
    
    nTotalSamples         = size(dataX, 1);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHECK ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nRepeat       = 100;
learningRatio = round(0.2 * nTotalSamples); 

tic;
[meanAccuracy, stdAccuracy] = AccuracyNRepeat(dataX, dataY, learningRatio, nComponents, nRepeat);
sprintf('ALL FEATURES:     [Accuracy avg: %0.2f%%]     [Accuracy std: %0.2f]     [Time: %0.3f seconds]', meanAccuracy * 100, stdAccuracy * 100, toc)