clear;clc;close all;

load motorcycle.mat

%% Split training-testing data
rate = 0.6;
rand('seed',12345);
randn('seed',12345);
[n d] = size(X);                 % samples x bands
r = randperm(n);                 % random index
ntrain = round(rate*n);          % #training samples
Xtrain = X(r(1:ntrain),:);       % training set
Ytrain = y(r(1:ntrain),:);       % observed training variable
Xtest  = X(r(ntrain+1:end),:);   % test set
Ytest  = y(r(ntrain+1:end),:);   % observed test variable

% model = trainKNNR(Xtrain,Ytrain)
% Ypred = testKNNR(model,Xtest)
% assessment(Ytest,Ypred,'regress')
% break

k = 5;
[IDX,D] = knnsearch(Xtrain,Xtest,'K',k);
whos

% knn regression 
Ypredtest = mean(Ytrain(IDX),2);
assessment(Ytest,Ypredtest,'regress')

% weighted knn regression
    W = 1./D; % compute the inverse distances to each neighbor
    W = W./repmat(sum(W,2),1,k); % normalize in rows to obtain the relative inverse distance to each neighbor
    Ypredtest = nansum(Ytrain(IDX) .* W, 2);
    assessment(Ytest,Ypredtest,'regress')

% 
% [IDX,D] = knnsearch(Xtrain,X,'K',k);
% whos
% Ypred = nansum(Ytrain(IDX)./D(IDX),2);
% assessment(Ytest,Ypredtest,'regress')
% figure,plot(X,y,'k.',X,Ypred,'b')
