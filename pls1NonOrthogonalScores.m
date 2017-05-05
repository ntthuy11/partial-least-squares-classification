%% Algorithm is implemented by Thuy Tuong Nguyen (Dec 19, 2013) according to the paper
%       M. Andersson, "A comparison of nine PLS1 algorithm," J. Chemometrics, 2009

function pls1Model = pls1NonOrthogonalScores(X, oneColumnY, nComponents)

    if (size(oneColumnY, 2) == 1) % only process 1-column y 
        
        % ----- initilize parameters -----
        x  = X;             % nSamples x nFeatures        
        y  = oneColumnY;    % nSamples x 1
        y0 = y;
        
        
        [nSamples, nFeatures] = size(X);
    
        T = zeros(nSamples,  nComponents);
        %U = zeros(nSamples,  nComponents);
        P = zeros(nFeatures, nComponents);
        %Q = zeros(1,         nComponents);
        W = zeros(nFeatures, nComponents);        
        %B = zeros(nFeatures, nComponents);
    
        
        % ----- iteration -----
        for iComponent = 1:nComponents
            Xt               = x';
            XtY              = Xt * y;                  % (nFeatures x nSamples) x (nSamples x 1)
            YtX              = y' * x; % = XtY'         % (1 x nSamples) x (nSamples x nFeatures)
            
            % X-weights
            w                = XtY / sqrt(YtX*XtY);     % (nFeatures x 1) / (1 x 1)
            W(:, iComponent) = w; 
            
            % X-scores
            t                = x * W(:, iComponent);    % (nSamples x nFeatures) x (nFeatures x 1)
            T(:, iComponent) = t;
            
            % Y-loadings
            %TT               = T';                      % nComponents x nSamples
            %Qt               = inverse(TT*T) * TT * y0; % (nComponents x nComponents) x (nComponents x nSamples) x (nSamples x 1)
            Qt               = T\y0;                    % <--- uncomment for best stability (from the paper)
            
            % X-loadings                                % <--- THUY ADDED
            ttt              = t' * t;                  % (1 x nSamples) x (nSamples x 1)
            P(:, iComponent) = Xt * t ./ ttt;           % p: (nFeatures x nSamples) x (nSamples x 1) / (1 x 1)
                        
            % regression matrix
            %B(:, iComponent) = W * Qt;                  % b: (nFeatures x nComponents) x (nComponents x 1)
            
            % deflation
            x                = x  - t * w';             % (nSamples x nFeatures) - (nSamples x 1) x (1 x nFeatures)
            y                = y0 - T * Qt;             % (nSamples x 1)         - (nSamples x nComponents) x (nComponents x 1)
        end
        
        
        % calculate the regression matrix B
        Q  = Qt';                                       % 1 x nComponents
        R  = W * inverse(P' * W) * Qt;                  % (nFeatures x nComponents) x (nComponents x nComponents) x (nComponents x 1)
        B0 = mean(oneColumnY, 1) - mean(X, 1) * R;      % (1 x 1) - (1 x nFeatures) x (nFeatures x 1)
        
        
        % ----- return the PLS model -----
        pls1Model.T = T;
        pls1Model.U = y * Q; % (nSamples x 1) x (1 x nComponents)     <--- THUY ADDED (U is not needed for calculating B)
        pls1Model.P = P;
        pls1Model.Q = Q;
        pls1Model.W = W;
        pls1Model.B = [B0; R]; % (nFeatures + 1) x 1
        
    else
        error('y must not have more than 2 classes, in case of this algorithm');
    end
end