function model = trainMSVR(X,Y)

% 2/3 / 1/3 classic train / validation partition
n = size(X,1);
rate = 0.66; 
ntrain = round(rate*n);
r = randperm(n);
Xtrain = X(r(1:ntrain),:);
Ytrain = Y(r(1:ntrain),:);
Xtest = X(r(ntrain+1:end),:);
Ytest = Y(r(ntrain+1:end),:);

tol = 1e-10;
eps = [0.001 0.01 0.1:0.1:0.5];
C = logspace(0,3,10);

% First guess for the sigma parameter
meanSigma = mean(pdist(X));
sigmaMin = log10(meanSigma * 0.1);
sigmaMax = log10(meanSigma * 10);
sigma = logspace(sigmaMin, sigmaMax, 20);

bestEps = 1;
bestC = 1;
bestSigma = 1;
bestRMSE = Inf;

for ss = 1:numel(sigma)
    Ktest = kernelmatrix('rbf', Xtest', Xtrain', sigma(ss));
    for cc = 1:numel(C)
        for ee = 1:numel(eps)
            Beta = msvr(Xtrain, Ytrain, 'rbf', C(cc), eps(ee), sigma(ss), tol);
            Ypred = Ktest * Beta;
            RMSE = mean(sqrt(mean((Ytest - Ypred).^2)));
            if RMSE < bestRMSE
                bestRMSE = RMSE;
                bestEps = ee;
                bestC = cc;
                bestSigma = ss;
            end
        end
    end
end

% Final model with all training points and best parameters found
[Beta,NSV] = msvr(X, Y, 'rbf', C(bestC), eps(bestEps), sigma(bestSigma), tol);

% Here we can return only SVs using the (non-used) i1 value returned by the
% msvr function, i.e. Beta(i1,:), X(i1,:).
model.Beta = Beta;
model.NSV = NSV;
model.sigma = sigma(bestSigma);
model.C = C(bestC);
model.Xtrain = X;
