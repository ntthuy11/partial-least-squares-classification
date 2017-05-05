function [meanAccuracyRepeat, stdAccuracyRepeat] = AccuracyNRepeat(dataX, dataY, learningRatio, nComponents, nRepeat) 
    nTotalSamples    = size(dataX, 1);   
    accuracyRepeat   = size(nRepeat, 1);

    % run
    for iRepeat = 1:nRepeat

        % sequential selection
%         learningSamples = dataX(1              : learningRatio,  :);
%         testingSamples  = dataX(learningRatio+1 : nTotalSamples, :);
%         learningGroups  = dataY(1              : learningRatio,  :);
%         testingGroups   = dataY(learningRatio+1 : nTotalSamples, :);
        
        % random selection (using randsample), without replacement
        randIdx         = randsample(1:nTotalSamples, nTotalSamples);           %  <-- from Statistics Toolbox: stats.randsample
        learningSamples = dataX(randIdx(1               : learningRatio), :);
        testingSamples  = dataX(randIdx(learningRatio+1 : nTotalSamples), :);
        learningGroups  = dataY(randIdx(1               : learningRatio), :);
        testingGroups   = dataY(randIdx(learningRatio+1 : nTotalSamples), :);

        % run PLS-DA
        accuracyRepeat(iRepeat) = AccuracyOneRepeat(learningSamples, learningGroups, testingSamples, testingGroups, nComponents);
    end

    % return
    meanAccuracyRepeat = mean(accuracyRepeat);
    stdAccuracyRepeat  = std(accuracyRepeat);
end


%% ========================== 


function accuracy = AccuracyOneRepeat(learningSamples, learningGroups, testingSamples, testingGroups, nComponents)

    % prepare for normalization of input
    learningMean = mean(learningSamples); 
    learningStd  = std(learningSamples);

    % normalize learning samples
    learningMeanArray           = learningMean(ones(size(learningSamples, 1), 1),:);
    learningStdArray            = learningStd(ones(size(learningSamples, 1), 1),:);
%     normalizedLearningSamples   = (learningSamples - learningMeanArray)./learningStdArray; % centralize
    normalizedLearningSamples   = exp( -(learningSamples - learningMeanArray).^2 ./ learningStdArray ); % kernelize

    % normalize testing samples
    testingMeanArray            = learningMean(ones(size(testingSamples, 1), 1),:);
    testingStdArray             = learningStd(ones(size(testingSamples, 1), 1),:);
%     normalizedTestingSamples    = (testingSamples - testingMeanArray)./testingStdArray; % centralize
    normalizedTestingSamples    = exp( -(testingSamples - testingMeanArray).^2 ./ testingStdArray); % kernelize
   
    % run PLS-DA
    plsdaModelPrediction        = PLSDAprediction(normalizedLearningSamples, learningGroups, nComponents, normalizedTestingSamples, testingGroups);
    accuracy                    = plsdaModelPrediction.predictAccuracy;
end