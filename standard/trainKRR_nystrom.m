function model = trainKRR_nystrom(X,Y,k)

% Default rank
if ~exist('k', 'var')
    k = 5000;
end

n = size(X,1);
rate = 0.66;      % Use 2/3 - 1/3 for xvalidation
ntrain = round(rate*n);
r = randperm(n);
Xtrain = X(r(1:ntrain),:);
Ytrain = Y(r(1:ntrain),:);
Xtest = X(r(ntrain+1:end),:);
Ytest = Y(r(ntrain+1:end),:);

% [samples outdim] = size(Ytrain);

meanSigma = mean(pdist(X));
sigmaMin = log10(meanSigma*0.1);
sigmaMax = log10(meanSigma*10);
sigma = logspace(sigmaMin,sigmaMax,20);
gamma = logspace(-7,0,20);

rmse = Inf;
for ls = 1:numel(sigma)

    Kt = kernelmatrix('rbf', Xtrain', Xtrain', sigma(ls));
    Kv = kernelmatrix('rbf', Xtest', Xtrain', sigma(ls));

    if n <= k
        warning('simpleR:warning', 'Number of samples (%d) is less than k (%d), using full kernel', n, k)
    else
        [U,D] = nys(Kt, k);
        Kt = U * D * U';
    end

    for lg = 1:numel(gamma)

        % Train
        % 1/ Slow: compute the inverse of the regularized kernel matrix:
        % alpha = inv(gamma(lg) * eye(size(yt,1)) + Kt) * yt;
        % 2/ Faster: solve the linear problem:
        % alpha = (gamma(lg) * eye(size(yt,1)) + Kt) \ yt;
        % 3/ Even faster: Cholesky decomposition
        R = chol(gamma(lg) * eye(size(Ytrain,1)) + Kt);
        alpha = R\(R'\Ytrain);

        % Validate
        yp = Kv * alpha;

        % Error
        res = mean(sqrt(mean((Ytest-yp).^2)));

        if res < rmse
            model.sigma = sigma(ls);
            model.gamma = gamma(lg);
            rmse = res;
        end
    end
end

% Final model
model.x = X;
K = kernelmatrix('rbf', model.x', model.x', model.sigma);
model.alpha = (model.gamma * eye(size(Y,1)) + K) \ Y;
