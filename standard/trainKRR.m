function model = trainKRR(X,Y)

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

    for lg = 1:numel(gamma)

        % Train
        % 1/ Slow: compute the inverse of the regularized kernel matrix:
        % alpha = inv(Kt + gamma(lg) * eye(size(yt,1))) * yt;
        % 2/ Faster: solve the linear problem:
        % alpha = (Kt + gamma(lg) * eye(size(yt,1))) \ yt;
        % 3/ Even faster: Cholesky decomposition
        [R, p] = chol(Kt + gamma(lg) * eye(size(Ytrain,1)));
        if p > 0
            alpha = (Kt + gamma(lg) * eye(size(yt,1))) \ yt;
        else
            alpha = R\(R'\Ytrain);
        end

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
