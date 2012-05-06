%% Load data

load data/graphTheory;
%load data/searchEngines;
%load data/philConcepts;

%% Visualize data

figure(1);
imagesc(X>0);
title('Word representations of articles');
xlabel('word');
ylabel('document');

figure(2);
spy(A);
title('Wiki Links');
ylabel('From Document');
xlabel('To Document');

%% remove stop words and rare words
% recommended if using full metric matrix (see params.diagonal below)
freqs = sum(X>0)./size(X,1);
inds = freqs > .1 & freqs < .9;
X = X(:,inds);

%% normalize data

X = bsxfun(@rdivide, X, sum(X,2));
X(isnan(X)) = 0;

%% symmetrize links

A = A+A';

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
params.lambda = 1e-6;
params.maxIter = 5000;
params.printEvery = 100;
params.project = 'final';
params.diagonal = true;
% turn off 'diagonal' full matrix (richer metric)
%params.diagonal = false;


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