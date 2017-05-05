%% Model for Kernel Partial Least Squares regression --> GOOD WHEN nFeatures << nSamples
% The algorithm is Kernel PLS, from the paper
%       F. Lindgren, "The kernel algorithm for PLS," J. Chemometrics, 1993  
% implemented by Thuy Tuong Nguyen (Nov 20, 2013) with reference from the paper 
%       S. de Jong, "Comments on the PLS kernel algorithm," J. Chemometrics, 1994
%
function kplsModel = kernelXtYYtX(X, Y, nComponents)
    
    % ----- initilize parameters -----
    [nSamples, nFeatures] = size(X);        % X: nSamples x nFeatures
    [~, nDimY]            = size(Y);        % Y: nSamples x nDimY
    
    T = zeros(nSamples,      nComponents);
    U = zeros(nSamples,      nComponents);
    P = zeros(nFeatures,     nComponents);
    Q = zeros(nDimY,         nComponents);
    W = zeros(nFeatures,     nComponents);
    
    
    % ----- cross-product of X and Y -----
    
    % create the kernel
    XY = [X Y];                                     % XY: nSamples x (nFeatures + nDimY)
    k  = size(XY, 2);                               % k = nFeatures + nDimY
    for i = 1:k
       for j = i:k
          z         = XY(:, i)' * XY(:, j);         % XY(:, i) : nSamples x 1
          ZtZ(i, j) = z;                            % z        : 1 x 1
          ZtZ(j, i) = z;
       end
    end
    XtX = ZtZ(1:nFeatures,      1:nFeatures);       % XtX : nFeatures x nFeatures
    %YtY = ZtZ(nFeatures + 1:k,  nFeatures + 1:k);  
    XtY = ZtZ(1:nFeatures,      nFeatures + 1:k);   % XtY : nFeatures x nDimY (covariance matrix) 
    XtYYtX = XtY * XtY';                            % (XtY * XtY') = (nFeatures x nDimY) x (nDimY x nFeatures) = nFeatures x nFeatures
    
        
    % ----- iteration -----
    for iComponent = 1:nComponents
        
        % calculate X-weights 
        w  = XtYYtX(:, 1);
        w0 = rand(nFeatures, 1);
        while(norm(w-w0) > 1e-10)
           w0 = w;
           w  = (w0' * XtYYtX)';
           w  = w ./ norm(w);
        end
        W(:, iComponent) = w;                       % w: nFeatures x 1
               
        scal             = inverse(diag(w' * XtX * w)); % scaling factor for P and Q: 1 x 1
        
        % calculate X-loadings
        p                = (w' * XtX)' * scal;      % p: nFeatures x 1
        P(:, iComponent) = p;
        
        % calculate Y-loadings
        q                = (w' * XtY)' * scal;      % q: nDimY x 1
        Q(:, iComponent) = q;
        
        % calculate Y-scores
        U(:, iComponent) = Y * q;                   % U is not needed for calculating B (Thuy added)
        
        % calculate X-scores
        T(:, iComponent) = X * w;                   % T is not needed for calculating B (Thuy added)
        
        % deflate
        up               = eye(nFeatures) - (w*p'); % (w*p'): nFeatures x nFeatures
        upt              = up';
        XtX              = upt * XtX * up;
        XtY              = upt * XtY;
        XtYYtX           = XtY * XtY';
    end
    
    % calculate the regression matrix B
    R  = W * inverse(P' * W) * Q';                  % (P' * W)      : nComponents x nComponents; R: nFeatures x nDimY
    B0 = mean(Y, 1) - mean(X, 1) * R;               % B0 (intercept): 1 x nDimY
    
    
    % ----- return the PLS model -----
    kplsModel.T = T;
    kplsModel.U = U;
    kplsModel.P = P;
    kplsModel.Q = Q;
    kplsModel.W = W;
    kplsModel.B = [B0; R];                          % B: (nFeatures + 1) x nDimY
%     kplsModel.percentVarExplainedByX = cumsum( (sum(P.^2, 1) ./ sum(sum(X.^2, 1))) * 100 );     % 1 x nComponents
%     kplsModel.percentVarExplainedByY = cumsum( (sum(Q.^2, 1) ./ sum(sum(Y.^2, 1))) * 100 );     % 1 x nComponents
end