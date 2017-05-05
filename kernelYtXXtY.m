%% Model for Kernel Partial Least Squares regression --> GOOD WHEN nFeatures << nSamples
% The algorithm is Kernel PLS, from the paper
%       S. de Jong, "Comments on the PLS kernel algorithm," J. Chemometrics, 1994
% implemented by Thuy Tuong Nguyen, Nov 18, 2013 
%
% This paper is the improvement/modification of the SIMPLS paper (S. de Jong, 1993).
%       It is also slower in the case nFeatures >> nSamples (because of utilizing X'X and X'Y)
%
%
% ----------------------------------
%
% kplsModel = kernelYtXXtY(X, Y, nComponents)
%           X = TP
%           Y = UQ
%
% INPUT:
%       X                   nSamples x nFeatures        for calibration (better when X is kernelized by a Gaussian kernel)
%       Y                   nSamples x nVariables       for regression, or
%                           nSamples x nClasses         for discriminant analysis. Classes must be greater than 0
%       nComponents                                     number of latent variables to model
% 
% OUTPUT:
%   kplsModel struct with:
%
%       T                   nSamples x nComponents      X-scores
%       U                   nSamples x nComponents      Y-scores
%       P                   nFeatures x nComponents     X-loadings
%       Q                   nVariables x nComponents    Y-loadings
%       W                   nFeatures x nComponents     X-weights
%       B                   (1+nFeatures) x nVariables  regression vectors
%       B0                  1 x nVariables              for regression intercept
%
%       percentVarExplainedByX     1 x nComponents             variance explained by X
%       percentVarExplainedByY     1 x nComponents             variance explained by Y
% 
function kplsModel = kernelYtXXtY(X, Y, nComponents)

    % ----- initilize parameters -----
    [nSamples, nFeatures] = size(X);        % X: nSamples x nFeatures
    [~, nDimY]            = size(Y);        % Y: nSamples x nDimY
    
    T = zeros(nSamples,      nComponents);
    U = zeros(nSamples,      nComponents);
    P = zeros(nFeatures,     nComponents);
    Q = zeros(nDimY,         nComponents);
    W = zeros(nFeatures,     nComponents);
       
    
    % ----- cross-product of X and Y -----
    Xt  = X';                               % Xt  : nFeatures x nSamples
    XtX = Xt * X;                           % XtX : nFeatures x nFeatures (covariance matrix)
    XtY = Xt * Y;                           % XtY : nFeatures x nDimY (covariance matrix)

    
    % ----- iteration -----
    for iComponent = 1:nComponents

        % -----
%         % singular value decomposition
%         [AP, BQ, CQt]    = svd(XtY, 'econ'); % A : nFeatures x nDimY     (eigenvectors of (XtY)*(XtY)' )
%                                              % B : a diagonal matrix, nDimY x nDimY, which diagonal elements are singular values (eigenvalues)
%                                              % Ct: nDimY x nDimY         (eigenvectors of (XtY)'*(XtY) )        
%                                              
%         % initialize the loadings for X and Y (based on SVD)
%         a1               = AP(:, 1);         % a1: nFeatures x 1        % a1 ~ inv(P) ~ W, but not used in this code
%         b1               = BQ(1);            % b1: a scalar
%         c1               = CQt(:, 1);        % c1: nDimY x 1            % c1 ~ inv(Q)


        % kernel used: (XtY)'*(XtY) = Y'XX'Y
        % so, instead of using SVD, we use EIG to find eigenvectors/eigenvalues
        [eigenvectors, eigenvalues] = eig(XtY' * XtY);   % (XtY' * XtY) = (nDimY x nFeatures) x (nFeatures x nDimY) = nDimY x nDimY
        eigenvaluesDiagonal         = diag(eigenvalues); % eigenvectors: nDimY x nVectors
        idxOfDominantVector         = (eigenvaluesDiagonal == max(eigenvaluesDiagonal));
        qq                          = eigenvectors(:, idxOfDominantVector); % nDimY x 1
        
        % -----
                
        
        % calculate X-weights 
        w                = XtY * qq;                 % w              : nFeatures x 1
        w                = w ./ sqrt(w' * XtX * w);  % (w' * XtX * w): (1 x nFeatures) x (nFeatures x nFeatures) x (nFeatures x 1) = (1 x 1)
        W(:, iComponent) = w;
        
        % calculate X-loadings
        p                = XtX * w;                  % p: nFeatures x 1
        P(:, iComponent) = p;
        
        % calculate Y-loadings
        q                = (w' * XtY)'; % XtY' * w;  % q: nDimY x 1
        Q(:, iComponent) = q;
        
        % calculate Y-scores
        U(:, iComponent) = Y * q;                    % U is not needed for calculating B (Thuy added)
        
        % calculate X-scores
        T(:, iComponent) = X * w;                    % T is not needed for calculating B (Thuy added)
        
        % deflate
        XtX              = XtX -  p * p'; 
        XtY              = XtY - (q * p')'; % XtY - p * q';
    end
    
    % calculate the regression matrix B
    R  = W * inverse(P' * W) * Q';                   % (P' * W)      : nComponents x nComponents;     R: nFeatures x nDimY
    B0 = mean(Y, 1) - mean(X, 1) * R;                % B0 (intercept): 1 x nDimY
       
    
    % ----- return the PLS model -----
    kplsModel.T = T;
    kplsModel.U = U;
    kplsModel.P = P;
    kplsModel.Q = Q;
    kplsModel.W = W;
    kplsModel.B = [B0; R];                           % B             : (nFeatures + 1) x nDimY  
%     kplsModel.percentVarExplainedByX = cumsum( (sum(P.^2, 1) ./ sum(sum(X.^2, 1))) * 100 );     % 1 x nComponents
%     kplsModel.percentVarExplainedByY = cumsum( (sum(Q.^2, 1) ./ sum(sum(Y.^2, 1))) * 100 );     % 1 x nComponents
end