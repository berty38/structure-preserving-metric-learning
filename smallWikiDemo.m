%% Load data

load data/graphTheory;
%load data/searchEngines;
%load data/philConcepts;

%% Visualize data

figure(1);
imagesc(X);
title('Word representations of articles');
xlabel('word');
ylabel('document');

figure(2);
spy(A);
title('Wiki Links');
ylabel('From Document');
xlabel('To Document');

%% sample 10 holdout documents

test_inds = randperm(length(names));
test_inds = test_inds(1:10);
holdout = false(length(names),1);
holdout(test_inds) = true;

%% remove holdout set

Xtr = X(~holdout,:);
Atr = A(~holdout, ~holdout);

%% run spml

params = [];
params.lambda = 1e-5;
params.maxIter = 5000;
params.printEvery = 1;

model = spml(Xtr', Atr, params);

%% compute distances

n = length(names);

for i = 1:length(test_inds)
    I = test_inds(i) * ones(n,1);
    J = 1:n;
    D = metricDistanceMask(X', model.M, sparse(I,J,true,n,n));
    
    [~,inds] = sort(D(test_inds(i),:),'ascend');
    
    fprintf('Closest articles to %s:\n', names{test_inds(i)});
    disp(names(inds(1:10)));
    
    pause;
end