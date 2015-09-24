function Ypred = testWKNNR(model,X)

[n d] = size(X);
[IDX,D] = knnsearch(model.Xtrain,X,'K',model.k);
W = 1./(D+eps); % compute the inverse distances to each neighbor
W = W./repmat(sum(W,2),1,model.k); % normalize in rows to obtain the relative inverse distance to each neighbor
Ypred = nansum(model.Ytrain(IDX) .* W, 2);
