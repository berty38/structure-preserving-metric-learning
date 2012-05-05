%% countNumImpostors
% 
% for each nodes computes the number of disconnected nodes which are closer
% than the furthest connected neighbor for that node and returns the sum of
% the impostor value over all nodes
%
% Computes the full node by node matrix, so only use on small problems
% 
% X - features for each node
% M - metric to be used, identity is euclidean
% A - Adjacency matrix
% 
%
%


function numImpostors = countNumImpostors(X, M, A)

N = size(X, 2);

K = X' * M * X;

margin = 0;

numImpostors = 0;


for i=1:N
    
    idx1 = i;
    
    ii = zeros(1, N);
    ii(idx1) = 1;
    
    [~, conIDX] = find(A(idx1, :) == 1);
    [~, disIDX] = find((A(idx1, :) + ii) == 0);
    
    Kii = K(idx1, idx1);
    Kij = K(idx1, :);
    Kjj = diag(K)';
    
    dists = Kii + Kjj - 2*Kij;
    
    [conDistance, idx] = max(dists(conIDX));
    
    if (length(idx) > 0)    
        j = conIDX(idx);

        [disconDistances, idx] = find(dists(disIDX) < (conDistance + margin));

        numImpostors = numImpostors + length(idx);
    end

end