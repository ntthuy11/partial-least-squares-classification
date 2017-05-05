%% Model for Partial Least Squares regression
% -Algorithm is implemented, by Thuy Tuong Nguyen (Nov 21, 2013) according to the paper
%       B. S. Dayal, "Improved PLS algorithms," J. Chemometrics, 1997
%
% -This modified kernel YtXXtY differs from the original kernel YtXXtY (S. de Jong, "Comments on the PLS kernel algorithm," J. Chemometrics, 1994) 
%  in the following steps:
%       1. X is not deflated
%       2. the additional vector r is computed according to equation (30)
%       3. t is computed using r rather than w
%
%       Equation (30):
%           r1 = w1
%           ra = wa - p1'*wa*r1 - p2'*wa*r2 - ... - p(a-1)'*wa*r(a-1),     a > 1
%
function kplsModel = kernelYtXXtYModified1(X, Y, nComponents) 

    % ----- initilize parameters -----
    [nSamples, nFeatures] = size(X);        % X: nSamples x nFeatures
    [~, nDimY]            = size(Y);        % Y: nSamples x nDimY
    
    T = zeros(nSamples,      nComponents);
    U = zeros(nSamples,      nComponents);
    P = zeros(nFeatures,     nComponents);
    Q = zeros(nDimY,         nComponents);
    W = zeros(nFeatures,     nComponents);
    
    R = zeros(nFeatures,     nComponents);
    
    
    % ----- cross-product of X and Y -----
    Xt  = X';                               % Xt  : nFeatures x nSamples
    XtY = Xt * Y;                           % XtY : nFeatures x nDimY (covariance matrix)

    
    % ----- iteration -----
    for iComponent = 1:nComponents
        
        % if there is single response variable, compute the X-weights as here
        if (nDimY == 1)
           w = XtY; 
        else
            
            % kernel used: (XtY)'*(XtY) = Y'XX'Y
            [eigenvectors, eigenvalues] = eig(XtY' * XtY);   % (XtY' * XtY) = (nDimY x nFeatures) x (nFeatures x nDimY) = nDimY x nDimY
            eigenvaluesDiagonal         = diag(eigenvalues); % eigenvectors: nDimY x nVectors
            idxOfDominantVector         = (eigenvaluesDiagonal == max(eigenvaluesDiagonal));
            qq                          = eigenvectors(:, idxOfDominantVector); % nDimY x 1
            
            % calcualte X-weights
            w = XtY * qq;                       % w: nFeatures x 1
        end
        
        % normalize X-weights
        w                = w ./ norm(w);
        W(:, iComponent) = w;
        
        % loop to calculate r, as equation (30)
        r = w;                                  % r: nFeatures x 1
        for j = 1:iComponent - 1
           r = r - P(:,j)' * w * R(:,j); 
        end
        R(:, iComponent) = r;
        
        % calculate X-scores
        t                = X * r;               % t: nSamples x 1
        T(:, iComponent) = t;
        tt               = t' * t;              % tt: 1 x 1
        
        % calculate X-loadings
        p                = (t'*X)' ./ tt; %(X'*t) ./ tt;       % p: nFeatures x 1
        P(:, iComponent) = p;
        
        % calculate Y-loadings
        q                = (r' * XtY)' ./ tt;   % q: nDimY x 1
        Q(:, iComponent) = q;
        
        % calculate Y-scores
        U(:, iComponent) = Y * q;               % U is not needed for calculating B (Thuy added)
        
        % deflate
        XtY              =  XtY - p * q' * tt;
    end
    
    % calculate the regression matrix B
    RQt  = R * Q';                              % RQt           : nFeatures x nDimY
    B0 = mean(Y, 1) - mean(X, 1) * RQt;         % B0 (intercept): 1 x nDimY
    
    
    % ----- return the PLS model -----
    kplsModel.T = T;
    kplsModel.U = U;
    kplsModel.P = P;
    kplsModel.Q = Q;
    kplsModel.W = W;
    kplsModel.B = [B0; RQt];
%     kplsModel.percentVarExplainedByX = cumsum( (sum(P.^2, 1) ./ sum(sum(X.^2, 1))) * 100 );     % 1 x nComponents
%     kplsModel.percentVarExplainedByY = cumsum( (sum(Q.^2, 1) ./ sum(sum(Y.^2, 1))) * 100 );     % 1 x nComponents    
end