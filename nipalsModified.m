%% Model for Partial Least Squares regression
% -Algorithm is implemented, by Thuy Tuong Nguyen (Nov 21, 2013) according to the paper
%       B. S. Dayal, "Improved PLS algorithms," J. Chemometrics, 1997
%
% -This modified NIPALS differs from the original only in the following steps:
%       1. X is not deflated
%       2. the additional vector r is computed according to equation (30)
%       3. t is computed using r rather than w
%
%       Equation (30):
%           r1 = w1
%           ra = wa - p1'*wa*r1 - p2'*wa*r2 - ... - p(a-1)'*wa*r(a-1),     a > 1
%
function plsModel = nipalsModified(X, Y, nComponents)

    % ----- initilize parameters -----
    [nSamples, nFeatures] = size(X);        % X: nSamples x nFeatures
    [~, nDimY]            = size(Y);        % Y: nSamples x nDimY
    
    T = zeros(nSamples,      nComponents);
    U = zeros(nSamples,      nComponents);
    P = zeros(nFeatures,     nComponents);
    Q = zeros(nDimY,         nComponents);
    W = zeros(nFeatures,     nComponents);
    
    R = zeros(nFeatures,     nComponents);
    
    
    % ----- iteration -----
    for iComponent = 1:nComponents
        [~, uidx] = max(std(Y)); % assign the Y with the largest variance to initialize the Y-scores U
        u         = Y(:, uidx);
        
        % iteration for outer modeling until convergence:   u -> w -> r -> t -> q -> u
        diff      = 1;
        while (diff > 1e-10)
            u0    = u;                              % u: nSamples x 1
            
            % calculate X-weight
            w     = X'*u ./ (u'*u);                 % w: nFeatures x 1
            w     = w ./ norm(w);
            
            % eq. (30)
            r     = w;                              % r: nFeatures x 1
            if (iComponent > 1)
                for j = 1:iComponent-1
                    r = r - P(:,j)' * w * R(:,j);
                end
            end
            
            % calculate X-scores t, Y-loadings q, Y-scores u
            t      = X * r;                         % t: nSamples x 1
            q      = Y'*t ./ (t'*t);                % q: nDimY    x 1
            u      = Y*q ./ (q'*q);                 % u: nSamples x 1
            
            %
            diff   = norm(u0-u) / norm(u);
        end
        
        % calculate X-loadings y
        p = X'*t ./ (t'*t);                          % q: nFeatures x 1
        
        % only Y block is deflated
        Y = Y - t*q';
        
        % assign vectors to matrices
        T(:, iComponent) = t;
        P(:, iComponent) = p;
        U(:, iComponent) = u;
        Q(:, iComponent) = q;
        W(:, iComponent) = w;        
        R(:, iComponent) = r;
    end
    
    % ----- return the PLS model -----
    RQt = R*Q';                            % RQt           : nFeatures x nDimY
    B0  = mean(Y, 1) - mean(X, 1) * RQt;   % B0 (intercept): 1 x nDimY
    
    plsModel.T = T;
    plsModel.U = U;
    plsModel.P = P;
    plsModel.Q = Q;
    plsModel.W = W;
    plsModel.B = [B0; RQt];
%     plsModel.percentVarExplainedByX = cumsum( (sum(P.^2, 1) ./ sum(sum(X.^2, 1))) * 100 );     % 1 x nComponents
%     plsModel.percentVarExplainedByY = cumsum( (sum(Q.^2, 1) ./ sum(sum(Y.^2, 1))) * 100 );     % 1 x nComponents
end