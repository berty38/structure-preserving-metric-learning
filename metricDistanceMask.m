function D = metricDistanceMask(X, M, mask)

% computes the distance matrix of X only in nonzeros of mask

n = size(X,2);

XM = X'*M;

if nargin==3 && issparse(mask)
    % if you can't compile the mex file spouterprod2, use the matlab version
    % spouterprod.m
    %K = spouterprod(mask, XM, X')+mask;
    K = spouterprod2(mask, XM', X)+mask;
else
    K = XM*X;
end

XX = sum(XM'.*X)';

[I,J] = find(mask);
V = XX(I) + XX(J);

if (nnz(mask) > nnz(K))
    fprintf('Something is wrong\n');
else
    V = V - 2*(nonzeros(K)-1);
    clear K;
end
[I,J] = find(mask);
D = sparse(I,J,V,n,n);

