close all;
warning off;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% BIOFILM DATA
excelFile       = '714 samples 143x9 features.xlsx';
sheetNo         = 1;
nColsFrom       = 1;
nCols           = 1287;
groupColIdx     = nCols + 1;
nComponents     = 60;


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
