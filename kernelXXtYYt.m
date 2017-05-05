%% Model for Kernel Partial Least Squares regression --> GOOD WHEN nFeatures >> nSamples
% from the paper: 
%       Aylin Alin, "Improved straightforward implementation of a statistical inspired modification of the partial least squares algorithm," 
%                   Pakistan Journal of Statistics, 2012.
%
% --> VERY SLOW <-- high number of eigenvectors from X(Y'X)'Y'
%
function kplsModel = kernelXXtYYt(X, Y, nComponents)

    % ----- initilize parameters -----
    [nSamples, nFeatures] = size(X);        % X: nSamples x nFeatures
    [~, nDimY]            = size(Y);        % Y: nSamples x nDimY
    
    T = zeros(nSamples,      nComponents);
    U = zeros(nSamples,      nComponents);
    P = zeros(nFeatures,     nComponents);
    Q = zeros(nDimY,         nComponents);
    W = zeros(nFeatures,     nComponents);
    
    V = zeros(nFeatures,     nComponents); % for deflate 
    
    
    % ----- cross-product of X and Y -----
    Xt  = X';                               % Xt  : nFeatures x nSamples
    XtY = (Y' * X)';                        % XtY = (Y'X)': nFeatures x nDimY (covariance matrix)
                                            %               because, for fast, we assume that nFeatures >> nSamples
    
    % ----- iteration -----
    for iComponent = 1:nComponents
        
        % kernel used: XX'YY' = X(Y'X)'Y'
        [eigenvectors, eigenvalues] = eig(X * XtY * Y');         % X(Y'X)'Y': nSamples x nSamples           % -->  nSamples number of eigenvectors  -->  too large
        eigenvaluesDiagonal         = diag(eigenvalues);         % eigenvectors: nSamples x 1
        idxOfDominantVector         = (eigenvaluesDiagonal == max(eigenvaluesDiagonal));
        
        % calculate X-scores
        t                = eigenvectors(:, idxOfDominantVector); % t: nSamples x 1
        T(:, iComponent) = t;                                    % t' = inv(t) because t is an orthogonal matrix
        
        % calculate X-loadings
        P(:, iComponent) = (t' * X)'; % P = X'T = (T'X)'         % (t' * X): 1 x nFeatures
        
        % calculate Y-loadings 
        q                = (t' * Y)'; %Y' * t;                   % q: nDimY x 1
        Q(:, iComponent) = q;
        
        % calculate X-weights 
        w                = XtY * q;                              % w        : nFeatures x 1
        w                = w ./ norm(w' * Xt);                   % (w' * Xt): 1 x nSamples
        W(:, iComponent) = w;
        
        % calcualte Y-scores
        u                = Y * q;                                % u: nSamples x 1
        if (iComponent > 1)
            U(:, iComponent) = u - T * (T' * u);                 % ADDITIONAL: 10. deflate U (for better U)
        else
            U(:, iComponent) = u;                                % U is not needed for calculating B
        end
        
        % deflate
        if (iComponent > 1)
            [dV, dXtY]       = deflateXtY(iComponent, V, XtY, P);
            V                = dV;
            XtY              = dXtY;
        end
    end
    
    % calculate the regression matrix B
    WQt = W * Q';                            % WQt           : nFeatures x nDimY
    B0  = mean(Y, 1) - mean(X, 1) * WQt;     % B0 (intercept): 1 x nDimY

    
    % ----- return the PLS model -----
    kplsModel.T = T;
    kplsModel.U = U;
    kplsModel.P = P;
    kplsModel.Q = Q;
    kplsModel.W = W;
    kplsModel.B = [B0; WQt];                 % B             : (nFeatures + 1) x nDimY
     
%     varX = (sum(P.^2, 1) ./ sum(sum(X.^2, 1))) * 100;
%     varY = (sum(Q.^2, 1) ./ sum(sum(Y.^2, 1))) * 100;
%     kplsModel.percentVarExplainedByX = cumsum( varX(2:length(varX)) ); % cumsum( diag(P'*P)/trace(Xt * X) * 100 )';     % 1 x nComponents
%     kplsModel.percentVarExplainedByY = cumsum( varY(2:length(varX)) ); % cumsum( diag(Q'*Q)/trace(Yt * Y) * 100 )';     % 1 x nComponents
end


%% ------------------------ Deflate the matrix XtY ------------------------

function [V, XtY] = deflateXtY(iComponent, previousV, previousXtY, P) 

    % initialize a vector for deflate
    V   = previousV;
    XtY = previousXtY;
    vi  = P(:, iComponent); % initialize orthogonal loadings

    % make v to be orthogonal to previous loadings
    vi  = vi - V * (V' * vi);

    % normalize vi
    vi               = vi ./ norm(vi);
    V(:, iComponent) = vi;

    % deflate XtY
    XtY = XtY - vi * (vi' * XtY);    
end