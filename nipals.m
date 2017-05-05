%% Model for Partial Least Squares regression
% - Algorithm is NIPALS (Nonlinear Iterative Partial Least-Squares)
% - This code is from http://www.mathworks.com/matlabcentral/fileexchange/18760-partial-least-squares-and-discriminant-analysis
% - It was slightly modified by Thuy Tuong Nguyen (Nov 20, 2013) for the PLS model framework
%
%
% -----------------------
%
% The difference between the total least squares regression and partial least squares regression can be explained as follows:
%
%       For given independent data X and dependent data Y, to fit a model:                                                                                      Y = X*B + E
%       the total least squares regression solves the problem to minimize the error in least squares sense:                                                     J = E'*E
%
%       Instead of directly fitting a model between X and Y, the PLS decomposes X and Y into low-dimensional space (so called laten variable space) first:      X = T*P' + E0, and 
%                                                                                                                                                               Y = U*Q' + F0
%       where P and Q are orthogonal matrices, i.e. P'*P=I, Q'*Q=I, T and U has the same number of columns, a, which is much less than the number of 
%       columns of X. Then, a least squares regression is performed between T and U:                                                                            U = T*B + F1
%
%       At the end, the overall regression model is:                                                                                                            Y = X*(P*B*Q') + F
%       i.e. the overall regression coefficient is P*B*Q'.
%
%       The reason to perform PLS instead of total LS regression is that the data sets X and Y may contain random noises, which should be excluded from 
%       regression. Decomposing X and Y into laten space can ensure the regression is performed based on most reliable variation.
%
%
% -----------------------
%
% [T, P, U, Q, B, Q] = pls(X, Y, tol) performs particial least squares regression between the independent variables, X and dependent Y as
%       X = T*P' + E;
%       Y = U*Q' + F = T*B*Q' + F1;
%
% Inputs:
% X     data matrix of independent variables
% Y     data matrix of dependent variables
% tol   the tolerant of convergence (defaut 1e-10)
% 
% Outputs:
% T     score matrix of X
% P     loading matrix of X
% U     score matrix of Y
% Q     loading matrix of Y
% B     matrix of regression coefficient
% W     weight matrix of X
%
% Using the PLS model, for new X1, Y1 can be predicted as                       Y1 = (X1*P)*B*Q' = X1*(P*B*Q')
%                                                      or                       Y1 = X1*(W*inv(P'*W)*inv(T'*T)*T'*Y)
%
% Without Y provided, the function will return the principal components as      X = T*P' + E
%
%
% -----------------------
%
% Example: taken from Geladi, P. and Kowalski, B.R., "An example of 2-block predictive partial least-squares regression with simulated data", Analytica Chemica Acta, 185(1996) 19--32.
%{
    x = [4 9 6 7 7 8 3 2;6 15 10 15 17 22 9 4;8 21 14 23 27 36 15 6; 10 21 14 13 11 10 3 4; 12 27 18 21 21 24 9 6; 14 33 22 29 31 38 15 8; 16 33 22 19 15 12 3 6; 18 39 26 27 25 26 9 8;20 45 30 35 35 40 15 10];
    y = [1 1;3 1;5 1;1 3;3 3;5 3;1 5;3 5;5 5];

    % leave the last sample for test
    N  = size(x,1);
    x1 = x(1:N-1,:);
    y1 = y(1:N-1,:);
    x2 = x(N,:);
    y2 = y(N,:);

    % normalization
    xmean = mean(x1);
    xstd  = std(x1);
    ymean = mean(y1);
    ystd  = std(y);
    X     = (x1-xmean(ones(N-1,1),:))./xstd(ones(N-1,1),:);
    Y     = (y1-ymean(ones(N-1,1),:))./ystd(ones(N-1,1),:);

    % PLS model
    [T, P, U, Q, B, W] = pls(X, Y);

    % Prediction and error
    yp = (x2-xmean)./xstd * (P*B*Q');
    fprintf('Prediction error: %g\n',norm(yp-(y2-ymean)./ystd));
%}
%
%
% -----------------------
%
% By Yi Cao at Cranfield University on 2nd Febuary 2008
%
% Reference:
% Geladi, P and Kowalski, B.R., "Partial Least-Squares Regression: A Tutorial", Analytica Chimica Acta, 185 (1986) 1--7.
%
function plsModel = nipals(X, Y, nComponents)

    % ----- Input check -----
    narginchk(1, 3);
    nargoutchk(0, 6);
    
    % if there is no Y
    if nargin < 2
        Y = X;
    end
    
    
    % ----- Define parameters -----
    toleranceT = 1e-10;
    toleranceY = 1e-10;

    % Size of x and y
    [nSamples, nFeatures] = size(X);
    [rY, nDimY] = size(Y);
    assert(nSamples == rY, 'Sizes of X and Y mismatch.');

    
    % ----- Allocate memory to the maximum size -----
    T = zeros(nSamples,      nComponents);
    U = zeros(nSamples,      nComponents);
    P = zeros(nFeatures,     nComponents);
    Q = zeros(nDimY,         nComponents);
    W = zeros(nFeatures,     nComponents);
    
    B = zeros(nComponents,   nComponents);

    iComponent = 0;
    
    % iteration loop if residual is larger than specfied
    while (norm(Y) > toleranceY) && (iComponent < nComponents)
        [~, tidx] = max(sum(X.*X)); % choose the column of x has the largest square of sum as t
        [~, uidx] = max(sum(Y.*Y)); % choose the column of y has the largest square of sum as u        
        t0 = X(:, tidx);            % t0: current value of t, for the loop
        u  = Y(:, uidx);            % u : nSamples x 1 
        t  = zeros(nSamples, 1);    % t : nSamples x 1

        % iteration for outer modeling until convergence:   u -> w -> t -> q -> u
        while norm(t0-t) > toleranceT
            w  = (u'*X)'; %X'*u;    % w: nFeatures x 1      u -> w
            w  = w ./ norm(w);
            t  = t0;
            t0 = X * w;             %                       w -> t
            q  = (t0'*Y)'; %Y'*t0;  % q: nDimY x 1          t -> q
            q  = q ./ norm(q);
            u  = Y * q;             %                       q -> u
        end
        
        % update p based on t
        t     = t0;
        tt    = t'*t;
        p     = (t'*X)' ./ tt; %X'*t ./ tt;               % t -> q
        pnorm = norm(p);
        p     = p ./ pnorm;
        t     = t * pnorm;
        w     = w * pnorm;

        % regression and residuals
        b = (t'*u)' ./ tt; %u'*t ./ tt;     % U = TB
        X = X - t*p';                       % deflate X
        Y = Y - b*t*q';                     % deflate Y

        % save iteration results to outputs:
        iComponent = iComponent+1;
        T(:, iComponent) = t;
        P(:, iComponent) = p;
        U(:, iComponent) = u;
        Q(:, iComponent) = q;
        W(:, iComponent) = w;
        B(iComponent, iComponent) = b;
        
        % uncomment the following line if you wish to see the convergence
        %disp(norm(Y))
    end
    
    T(:, iComponent+1:end) = [];
    P(:, iComponent+1:end) = [];
    U(:, iComponent+1:end) = [];
    Q(:, iComponent+1:end) = [];
    W(:, iComponent+1:end) = [];
    
    B = B(1:iComponent, 1:iComponent);
    
    
    % ----- return the PLS model -----
    PBQt = P*B*Q';                           % PBQt          : nFeatures x nDimY
    B0   = mean(Y, 1) - mean(X, 1) * PBQt;   % B0 (intercept): 1 x nDimY
    
    plsModel.T = T;
    plsModel.U = U;
    plsModel.P = P;
    plsModel.Q = Q;
    plsModel.W = W;
    plsModel.B = [B0; PBQt];
%     plsModel.percentVarExplainedByX = cumsum( (sum(P.^2, 1) ./ sum(sum(X.^2, 1))) * 100 );     % 1 x nComponents
%     plsModel.percentVarExplainedByY = cumsum( (sum(Q.^2, 1) ./ sum(sum(Y.^2, 1))) * 100 );     % 1 x nComponents
end