function model = trainWKNNR(X,Y)

[n d] = size(X);
rate = 0.66;      % Use 2/3 - 1/3 for xvalidation
ntrain = round(rate*n);
r = randperm(n);
Xtrain = X(r(1:ntrain),:);
Ytrain = Y(r(1:ntrain),:);
Xtest  = X(r(ntrain+1:end),:);
Ytest  = Y(r(ntrain+1:end),:);

K = 1:20;

res = Inf;
for k=K
    [IDX,D] = knnsearch(Xtrain,Xtest,'K',k);
    W = 1./(D+eps); % compute the inverse distances to each neighbor
    W = W./repmat(sum(W,2),1,k); % normalize in rows to obtain the relative inverse distance to each neighbor
    Ypred = nansum(Ytrain(IDX) .* W, 2);
    res(k) = norm(Ytest-Ypred,'fro');
end
% figure,plot(K,res,'ko-'), grid

[val idx] = min(res);
model.k = idx;
model.Xtrain = Xtrain; % should we save X? ;)
model.Ytrain = Ytrain; % should we save Y? ;)

