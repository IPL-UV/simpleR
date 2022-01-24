function model = trainRLR(X,Y,max_gamma)

[n,d] = size(X);
X = [ones(n,1) X];
cv = cvpartition(n, 'KFold', 4);

Ctrain = X' * X;
if ~exist('max_gamma', 'var')
    max_gamma = floor(log10(max(Ctrain(:))));
else
    max_gamma = log10(max_gamma);
end
gamma = logspace(-10,max_gamma,50);

cvgamma = zeros(1,cv.NumTestSets);
for k = 1:cv.NumTestSets
    idxtrn = training(cv,k);
    Xtrain = X(idxtrn,:);
    Ytrain = Y(idxtrn,:);
    idxtst = test(cv,k);
    Xtest = X(idxtst,:);
    Ytest = Y(idxtst,:);    
    res = nan(1,numel(gamma));
    for lg = 1:numel(gamma)
        W = (Xtrain'*Xtrain + gamma(lg)*eye(d)) \ (Xtrain'*Ytrain);
        Ypred = Xtest * W;
        res(lg) = mean(sqrt(mean((Ytest-Ypred).^2)));
    end
    [~, idx] = nanmin(res);
    cvgamma(k) = gamma(idx);
end
% figure,semilogx(gamma,res,'ko-'), grid

bestgamma = median(cvgamma);
W = (X'*X + bestgamma*eye(d)) \ (X'*Y);
model.W = W;
model.gamma = bestgamma;
