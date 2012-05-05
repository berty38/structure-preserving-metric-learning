function X = spouterprod(mask, U, V)

%function X = spouterprod(mask, U, V)
% computes a sparse outer product of U and V with nonzeros only where mask
% is true
% Equivalent to X = mask .* (U*V'), but skips computation of false entries
% in mask

U = U';
V = V';
% computes mask .* U*V'

[I,J] = find(mask);

vals = zeros(nnz(mask),1);
%
% if (nnz(mask)>1e6)
%     fprintf('Computing large sparse outer product\n');
% end

tic;

start = 1;

for col = 1:size(mask,2)
    if (J(start) == col)
        inds = start : start + find(J(start:end)==col, 1, 'last') - 1;
        
        vals(inds) = U(:, I(inds))' * V(:,col);
        
        start = inds(end)+1;
    end
    if toc > 5
        if (nnz(mask) > 1e6)
            fprintf('Completed column %d of %d (%f)\n', ...
                col, size(mask,2), col/size(mask,2));
        end
        tic;
    end
    
end

X = sparse(I,J,vals,size(mask,1),size(mask,2));
