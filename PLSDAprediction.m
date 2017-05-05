function plsdaModelWithPrediction = PLSDAprediction(Xlearning, Ylearning, nComponents, Xtesting, Ytesting)
    
    % run PLSDA
    plsdaModel = plsda(Xlearning, Ylearning, nComponents);
    
    
    % calculate accuracy of prediction    
    predictY   = [ones(size(Xtesting, 1), 1) Xtesting] * plsdaModel.B;
    
    if (size(predictY, 2) > 1) % more than 2 classes
        [~, predictIndices] = min(abs(predictY - 1), [], 2); % map the predicted Y back to the labels
        predictDiff         = Ytesting - predictIndices;
        
    else % only 2 classes
        nPredictY   = size(predictY, 1);
        predictDiff = zeros(nPredictY, 1);
        
        uniqueY     = unique([Ylearning; Ytesting]);
        class1      = uniqueY(1);
        class2      = uniqueY(2);        
        
        for i = 1:size(predictY, 1)
            if (abs(predictY(i) - class1) > abs(predictY(i) - class2))
                predictDiff(i) = Ytesting(i) - class2;
            else
                predictDiff(i) = Ytesting(i) - class1;
            end
        end
    end
    predictAccuracy = sum(predictDiff == 0) / size(predictDiff, 1);
    
    
    % ----- assign results from plsdaModel to plsdaModelWithPrediction -----
    plsdaModelWithPrediction.T  = plsdaModel.T;
    plsdaModelWithPrediction.U  = plsdaModel.U;
    plsdaModelWithPrediction.P  = plsdaModel.P;
    plsdaModelWithPrediction.Q  = plsdaModel.Q;
    plsdaModelWithPrediction.W  = plsdaModel.W;
    plsdaModelWithPrediction.B  = plsdaModel.B;
    
    plsdaModelWithPrediction.predictY        = predictY;
    plsdaModelWithPrediction.predictAccuracy = predictAccuracy;
end


%% ========================================================================


function plsdaModel = plsda(X, oneColumnY, nComponents) 
    Y = transformToManyColumns(oneColumnY);
    
    
    % ----- run PLS regression -----
    
    % kernel methods
    plsModel = kernelXtYYtX(X, Y, nComponents);   % used when nFeatures << nSamples                       (F. Lindgren, 1997)
%     plsModel = kernelXXtYYt(X, Y, nComponents);   % used when nFeatures >> nSamples (SUPER SLOW)          (Aylin Alin, 2012)
%     plsModel = kernelYtXXtY(X, Y, nComponents);   % used when nFeatures << nSamples (faster than SIMPLS)  (S. de Jong, 1994)
%     plsModel = kernelYtXXtYModified1(X, Y, nComponents);  % <----- VERY FAST ------                       (B. S. Dayal, 1997)
%     plsModel = kernelYtXXtYModified2(X, Y, nComponents);  % <----- VERY FAST ------                       (B. S. Dayal, 1997)

    % NIPALS
%     plsModel = nipals(X, Y, nComponents);               % low accuracy for our data, SLOW                 (P. Geladi, 1986)
%     plsModel = nipalsModified(X, Y, nComponents);       % good, but VERY SLOW                             (B. S. Dayal, 1997)
    
    % SIMPLS
%     plsModel = simpls(X, Y, nComponents);         % used when nFeatures << nSamples                       (S. de Jong, 1993)

    % PLS1
%     plsModel = pls1NonOrthogonalScores(X, oneColumnY, nComponents);       % (M. Andersson, 2009)
%     plsModel = pls1BiDiag2(X, oneColumnY, nComponents);                   % (M. Andersson, 2009)
    
    % ------------------------------

    
    % assign results from pslRegresionModel to plsdaModel
    plsdaModel.T        = plsModel.T;
    plsdaModel.U        = plsModel.U;
    plsdaModel.P        = plsModel.P;
    plsdaModel.Q        = plsModel.Q;
    plsdaModel.W        = plsModel.W;
    plsdaModel.B        = plsModel.B;
     
%     plsdaModel.crossValPercentVarExplainedByX = plsModel.percentVarExplainedByX;
%     plsdaModel.crossValPercentVarExplainedByY = plsModel.percentVarExplainedByY;    
end


%% ========================================================================

% Transform 1-column Y to many-column Y
function Y = transformToManyColumns(oneColumnY)
    if (size(oneColumnY, 2) ~= 1)
        error('Incorrect Y!');
    else
        nClasses = size(unique(oneColumnY), 1);
        nSamples = size(oneColumnY, 1);
        
        Y        = zeros(nSamples, nClasses);
        for i = 1:nSamples
           Y(i, oneColumnY(i)) = 1;
        end
    end
end