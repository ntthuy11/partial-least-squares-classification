function pls1Model = pls1BiDiag2(X, oneColumnY, nComponents)

    if (size(oneColumnY, 2) == 1) % only process 1-column y 
        
        % ----- initilize parameters -----
        x  = X;             % nSamples x nFeatures        
        y  = oneColumnY;    % nSamples x 1
        
        
        [nSamples, nFeatures] = size(X);
    
        T = zeros(nSamples,  nComponents);
        %U = zeros(nSamples,  nComponents);
        P = zeros(nFeatures, nComponents);
        %Q = zeros(1,         nComponents);
        W = zeros(nFeatures, nComponents);
        
        Bsqr = zeros(nComponents, nComponents);
        %B = zeros(nFeatures, nComponents);
        
        
        % ----- iteration -----
        Xt      = x';           % nFeatures x nSamples
        
        w       = Xt * y;       % nFeatures x 1
        w       = w ./ norm(w);
        W(:, 1) = w;
        
        t       = x * w;        % nSamples x 1
        b       = norm(t);
        Bsqr(1, 1) = b;
        T(:, 1) = t ./ b;
        
        P(:, 1) = Xt * t ./ (t' * t); % <--- THUY ADDED
        
        % first iteration
        for iComponent = 2:nComponents
            
            % X-weights
            w                              = Xt * T(:, iComponent-1) - Bsqr(iComponent-1, iComponent-1) * W(:, iComponent-1);     % (nFeatures x nSamples) x (nSamples x 1) - (nFeatures x 1)
            b                              = norm(w);
            Bsqr(iComponent-1, iComponent) = b;
            W(:, iComponent)               = w ./ b;               % w: nFeatures x 1
            
            % X-scores
            t                              = x * w - Bsqr(iComponent-1, iComponent) * T(:, iComponent-1);   % (nSamples x nFeatures) x (nFeatures x 1) - (nSamples x 1)
            b                              = norm(t);
            Bsqr(iComponent, iComponent)   = b;
            T(:, iComponent)               = t ./ b;               % t: nSamples x 1
            
            % X-loadings                                           % <--- THUY ADDED
            P(:, iComponent)               = Xt * t ./ (t' * t);   % p: (nFeatures x nSamples) x (nSamples x 1) / (1 x 1)
        end
        
        % second iteration
        Q       = y' * T;                                          % (1 x nSamples) x (nSamples x nComponents)
        %invBsqr = inverse(Bsqr);                                  % NOT USED (from the paper)
        %for iComponent = 1:nComponents
        %   B(:, iComponent) = W(:, 1:iComponent) * (invBsqr(1:iComponent, 1:iComponent) * Q(1:iComponent)');   % (nFeatures x iComponent) x ( (iComponent x iComponent) x (iComponent x 1) )
        %end
        
        
        % calculate the regression matrix B
        %R  = W * inverse(P' * W) * Q';                            % (nFeatures x nComponents) x (nComponents x nComponents) x (nComponents x 1)
        % (or)
        R  = P * Bsqr * Q';                                        % (nFeatures x nComponents) x (nComponents x nComponents) x (nComponents x 1)
        B0 = mean(oneColumnY, 1) - mean(X, 1) * R;                 % (1 x 1) - (1 x nFeatures) x (nFeatures x 1)
        
        
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