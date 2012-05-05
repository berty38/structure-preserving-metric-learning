function model = spml(X, A, params)

%%% function model = spml(X, A, params)
%
%
% SPML v0.1
% learns an M matrix
% the constraints are generated using mini-batches of triplet constraints
%
% X - (d x n) node features
% A - (n x n) node connectivity
%
% params
% lambda - regularization parameter
% maxNumMinutes - maximum cpu time for run
% maxNumIts - maximum number of iterations
% margin - target minimum difference between connected and disconnected distance
% miniBatchSize - number of triplets to consider each iterations
% diagonal - true if M is constrained to be nonzero only along main
% diagonal, otherwise full M matrix is computed
% printEvery - number of iterations between graphical output
%
% returns:
% model
% M - learned metric
% numImpIts - number of impostors found at each iteration
% elapsedSec
% params
%
% Copyright 2011-2012 Blake Shaw and Bert Huang
%

% set params if not set
if ~isfield(params, 'lambda')
    params.lambda = 1e-10;
end

if ~isfield(params, 'maxNumMinutes')
    params.maxNumMinutes = 7*24*60;
end

if ~isfield(params, 'maxIter')
    params.maxIter = 1e5;
end

if ~isfield(params, 'margin')
    params.margin = 1e-2;
end

if ~isfield(params, 'miniBatchSize')
    params.miniBatchSize = 10;
end

if ~isfield(params, 'diagonal')
    params.diagonal = true;
end

if ~isfield(params, 'printEvery')
    params.printEvery = 0;
end

% maximum size graph that we still count imposters on
MAX_COUNT_IMPOSTERS = 6000;


% load params into short variables
lambda = params.lambda;
maxNumMinutes = params.maxNumMinutes;
T = params.maxIter;
margin = params.margin;
miniBatchSize = params.miniBatchSize;


% setup model output
model.algo = 'SPML';
model.params = params;
model.predictor = @predictorMetrics;


% initialize

[D, N] = size(X);

ddd = rand(D, 1);
M = sparse(1:D, 1:D, ddd, D, D, D);

if N < MAX_COUNT_IMPOSTERS
    model.initialNumImpostors = countNumImpostors(X, speye(D), A);
end

tic

for t=1:T
    
    eta = 1/(lambda * t);
    
    C = sparse([],[],[],N,N,miniBatchSize*9);
    scores(t) = 0;
    
    for bb = 1:miniBatchSize
        
        i = randi(N);
        
        delta = zeros(1, N);
        delta(i) = 1;
        
        [~, conIDX] = find(A(:,i) == 1);
                
        if ~isempty(conIDX) && length(conIDX) < N
            
            idx = randi(length(conIDX));
            j = conIDX(idx);
            
            if length(conIDX) < N/100 % randomly sample to find disconnected nodes
                k = randi(N);
                while ismember(k, conIDX)
                    k = randi(N);
                end
            else
                [~, disIDX] = find((A(i, :) + delta) == 0); %remove O(n) time
                idx = randi(length(disIDX));
                k = disIDX(idx);
            end
            
            %% old easier to read kernel value computation
            %             Kii = X(:, i)' * M * X(:, i);
            %             Kjj = X(:, j)' * M * X(:, j);
            %             Kkk = X(:, k)' * M * X(:, k);
            %             Kij = X(:, i)' * M * X(:, j);
            %             Kik = X(:, i)' * M * X(:, k);
            %             Kji = X(:, j)' * M * X(:, i);
            %             Kki = X(:, k)' * M * X(:, i);
            
            
            %% faster kernel value computation
            XiM = (M * X(:,i))';
            XjM = (M * X(:,j))';
            XkM = (M * X(:,k))';
            
            Kii = XiM * X(:,i);
            Kjj = XjM * X(:,j);
            Kkk = XkM * X(:,k);
            Kij = XiM * X(:,j);
            Kji = Kij;
            Kik = XiM * X(:,k);
            Kki = Kik;
            
            
            distk = Kii + Kkk - Kik - Kki;
            distj = Kii + Kjj - Kij - Kji;
            
            if (distk <= distj + margin)
                %% old easier to read version
                %                 C(j, j) = C(j, j) + 1;
                %                 C(i, j) = C(i, j) - 1;
                %                 C(j, i) = C(j, i) - 1;
                %                 C(i, k) = C(i, k) + 1;
                %                 C(k, i) = C(k, i) + 1;
                %                 C(k, k) = C(k, k) - 1;
                
                %% sparse version
                C = C + sparse([j i j i k k], [j j i k i k], [1 -1 -1 1 1 -1], N, N);
                
                
                scores(t) = scores(t) + 1;
            end
        end
    end
    
    C = sparse(C);
    
    if params.diagonal
        dGrad = sum((X*C) .* X, 2);
        grad = sparse(1:D, 1:D, dGrad, D, D, D) + lambda * M;
    else
        grad = X * C * X' + lambda * M;
    end
    
    M0 = M;
    
    M = M0 - eta * grad;
    
    if (mod(t, params.printEvery) == 0)
        figure(12); plot(1:t, smooth(scores(1:t), 500, 'moving'));
        title('Number of Impostors Per Node');
        xlabel('Iterations');
        ylabel('Number of Impostors Found');
        drawnow;
    end
    
    if (toc > maxNumMinutes*60)
        break
    end
    
end

model.M = M;
model.numImpIts = scores;
model.elapsedSec = toc;
if N < MAX_COUNT_IMPOSTERS
    model.afterNumImpostors = countNumImpostors(X, M, A);
end