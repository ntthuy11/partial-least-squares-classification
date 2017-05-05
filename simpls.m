%% Model for Partial Least Squares regression, originated by Cleiton A. Nunes (UFLA,MG,Brazil). --> GOOD WHEN nFeatures << nSamples
% -Algorithm is SIMPLS (Statistical Inspired Modification of Partial Least Squares), S. de Jong.
% -This code was cleaned, refactored, and utilized by Thuy Tuong Nguyen, Nov 13, 2013.
%
% More reference:
%       + Matlab plsregress
%       + Aylin Alin, "Improved straightforward implementation of a statistical inspired modification of the partial least squares algorithm," 
%                   Pakistan Journal of Statistics, 2012.
%
% Difference between SIMPLS and NIPALS (from Chapter 35, Vandeginste, B. G. M. Handbook of Chemometrics and Qualimetrics: Part B, 1997):
%       - The slight difference between the NIPALS implementation and the SIMPLS implementation is from the multivariate response (PLS2). 
%         (They give same result in PLS1, one Y)
%       - The restriction of SIMPLS is that any newly formed PLS factor ta = Xwa should be orthogonal to its predecessors tb = Xwb, where 1 <= b < a.
%         This appears to be equivalent to constraining the new weight vector xa to be orthogonal to prior loading vectors pb.
%       - It turns out that NIPALS does not truly maximize the covariance criterion after the first dimension whereas SIMPLS does.
%       - SIMPLS is less stable than NIPALS when the number of components is larger.
%
% SIMPLS is faster than NIPALS because of its straightforward calculation of weights W and Y-loadings Q from SVD(X'Y). The need is the deflation.
%       Herein, the algorithm will be slower in case nFeatures >> nSamples (we are utilizing X'Y)
%
%
% ----------------------------------
%
% plsModel = simpls(X, Y, nComponents)
%           X = TP
%           Y = UQ
%
%           W ~ inv(P)
%           B = [B0; WQ']
%
% INPUT:
%       X                   nSamples x nFeatures        for calibration (better when X is kernelized by a Gaussian kernel)
%       Y                   nSamples x nVariables       for regression, or
%                           nSamples x nClasses         for discriminant analysis. Classes must be greater than 0
%       nComponents                                     number of latent variables to model
% 
% OUTPUT:
%   plsModel struct with:
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
function plsModel = simpls(X, Y, nComponents)

    % ----- initilize parameters -----
    [nSamples, nFeatures] = size(X);        % X: nSamples x nFeatures
    [~, nDimY]            = size(Y);        % Y: nSamples x nDimY
    
    T = zeros(nSamples,      nComponents);
    U = zeros(nSamples,      nComponents);
    P = zeros(nFeatures,     nComponents);
    Q = zeros(nDimY,         nComponents);
    W = zeros(nFeatures,     nComponents);
    
    V = zeros(nFeatures,     nComponents); % for deflate 
    % (DEFLATE: to remove the already determined solution, while leaving the remainder solutions unchanged)
    
    
    % ----- cross-product of X and Y -----
    Xt  = X';                               % Xt  : nFeatures x nSamples
    XtY = Xt * Y;                           % XtY : nFeatures x nDimY (covariance matrix)
    
    
    % ----- iteration -----
    for iComponent = 1:nComponents
        
        % FLOW OF CALCULATION:     SVD(XtY)  -->  a1          -->  T                             -->  P    -->  Deflation of XtY
        %                           |---------->  a1, b1, c1  -->  W (from a1), Q (from b1, c1)  -->  WQt  -->  B
        % SVD(XtY) equals to  EIG((XtY)*(XtY)') = a1  and  EIG((XtY)'*(XtY)) = c1
        
        
        % singular value decomposition
        [AP, BQ, CQt]    = svd(XtY, 'econ'); % A : nFeatures x nDimY     (eigenvectors of (XtY)*(XtY)' )
                                             % B : a diagonal matrix, nDimY x nDimY, which diagonal elements are singular values (eigenvalues)
                                             % Ct: nDimY x nDimY         (eigenvectors of (XtY)'*(XtY) )
                                             % P and Q can be initialied/calculated using svd(XtY), that means P and Q are eigenvectors of (XtY)*(XtY)' and (XtY)'*(XtY), respectively.       
        
        % initialize the loadings for X and Y (based on SVD)
        ap               = AP(:, 1);        % ap: nFeatures x 1         % ap ~ inv(P) ~ W
        bq               = BQ(1);           % bq: a scalar
        cq               = CQt(:, 1);       % cq: nDimY x 1             % cq ~ inv(Q)
        
        % calculate X-scores
        t                = X*ap;            % t    : nSamples x 1
        tnorm            = norm(t);         % t1norm: a scalar
        t                = t ./ tnorm;
        T(:, iComponent) = t;               % T is not needed for DIRECTLY calculating B
        
        % calculate X-loadings
        P(:, iComponent) = Xt * t;
        
        % calculate Y-loadings
        q                = bq*cq / tnorm;   % q: nDimY x 1
        Q(:, iComponent) = q;
        
        % calculate Y-scores
        U(:, iComponent) = Y * q;           % U is not needed for calculating B
        
        % calculate X-weights
        W(:, iComponent) = ap ./ tnorm;
        
        % deflate 
        [dV, dXtY]       = deflateXtY(iComponent, V, XtY, P);
        V                = dV;
        XtY              = dXtY;
    end
    
    % calculate the regression matrix B
    WQt = W * Q';                           % WQt           : nFeatures x nDimY
    B0  = mean(Y, 1) - mean(X, 1) * WQt;    % B0 (intercept): 1 x nDimY
    
    % ADDITIONAL: 10. deflate U (for better U)
    U = deflateU(nComponents, U, T);
    
    
    % ----- return the PLS model -----
    plsModel.T = T;
    plsModel.U = U;
    plsModel.P = P;
    plsModel.Q = Q;
    plsModel.W = W;
    plsModel.B = [B0; WQt];                 % B             : (nFeatures + 1) x nDimY
%     plsModel.percentVarExplainedByX = cumsum( (sum(P.^2, 1) ./ sum(sum(X.^2, 1))) * 100 );     % 1 x nComponents
%     plsModel.percentVarExplainedByY = cumsum( (sum(Q.^2, 1) ./ sum(sum(Y.^2, 1))) * 100 );     % 1 x nComponents
end


%% ------------------------ Deflate the matrix XtY ------------------------

function [V, XtY] = deflateXtY(iComponent, previousV, previousXtY, P) 

    % initialize a vector for deflate
    V   = previousV;
    XtY = previousXtY;
    vi  = P(:, iComponent); % initialize orthogonal loadings
    
    % deflate vi
    for repeat = 1:2
       for j = 1:iComponent-1
          vj = V(:, j);
          vi = vi - (vi' * vj) * vj; % make v to be orthogonal to previous loadings
       end
    end
    
    % normalize vi
    vi               = vi ./ norm(vi);
    V(:, iComponent) = vi;
    
    % deflate XtY 
    XtY = XtY - vi * (vi' * XtY); 
    Vi  = V(:, 1:iComponent);
    XtY = XtY - Vi * (Vi' * XtY);
end


%% ------------------------ Deflate U ------------------------

function U = deflateU(nComponents, previousU, T)
    U = previousU;
    
    for iComponent = 1:nComponents
       ui = U(:, iComponent);
       
       for repeat = 1:2
          for j = 1:iComponent-1
             tj = T(:, j);
             ui = ui - (ui' * tj) * tj; % make u to be orthogonal to previous t values
          end
       end
       
       U(:, iComponent) = ui;
    end
end