function model = trainLWP(X,Y)

[n,~] = size(X);
rate = 0.66;      % Use 2/3 - 1/3 for xvalidation
ntrain = round(rate*n);
r = randperm(n);
Xtrain = X(r(1:ntrain),:);
Ytrain = Y(r(1:ntrain),:);
Xtest = X(r(ntrain+1:end),:);
Ytest = Y(r(ntrain+1:end),:);

ALPHAS = logspace(-3,2,10);
ORDERS = 1:5;

m=0;
for p=ORDERS
    for alfa = ALPHAS
    m=m+1;
    MSE = lwptest(Xtrain, Ytrain, Xtest, Ytest, p, alfa);
    results(m,:) = [p alfa MSE];
    end
end

[val idx] = min(results(:,3)); 
model.MSE = results(idx,3);
model.p   = results(idx,1);
model.alpha = results(idx,2);
model.X = Xtrain;
model.Y = Ytrain;



