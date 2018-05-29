function model = trainKRR_nystrom(X,Y,k,rate)

% Default rank
if ~exist('k', 'var')
    k = 5000;
end

if ~exist('rate', 'var')
    rate = 0.7;      % For xvalidation
end

n = size(X,1);
ntrain = round(rate*n);
if k < ntrain
    nk = 1:k;
else
    nk = 1:ntrain;
end

r = randperm(n);
idxTrain = r(1:ntrain);
idxTest = r(ntrain+1:end);

% meanSigma = mean(pdist(Xtrain(nk,:)));
meanSigma = mean(pdist(X(idxTrain(nk),:)));
sigmaMin = log10(meanSigma * 0.01);
sigmaMax = log10(meanSigma * 10);
sigma = logspace(sigmaMin, sigmaMax, 20);
gamma = logspace(-7, 0, 20);

rmse = Inf;
for ls = 1:numel(sigma)
    
    Kt = kernelmatrix('rbf', X(idxTrain,:)', X(idxTrain(nk),:)', sigma(ls));
    Kv = kernelmatrix('rbf', X(idxTest,:)', X(idxTrain(nk),:)', sigma(ls));

    KtT_Kt = Kt' * Kt;
    KtT_Yt = Kt' * Y(idxTrain,:);
    
    for lg = 1:numel(gamma)

        % Train
        % 1/ Slow and less accurate: compute the inverse of the regularized kernel matrix:
        % alpha = inv(gamma(lg) * Kt(nk,:) + Kt' * Kt) * (Kt' * Ytrain);
        
        % 2/ Faster: solve the linear problem:
        % alpha = (gamma(lg) * Kt(nk,:) + Kt' * Kt) \ (Kt' * Ytrain);
        
        % 3/ Even faster: Cholesky decomposition
        % Sometimes this rises 'matrix must be positive definite'
        [R,p] = chol(gamma(lg) * Kt(nk,:) + KtT_Kt);
        if p == 0
            alpha = R \ (R' \ KtT_Yt);
        else
            alpha = (gamma(lg) * Kt(nk,:) + KtT_Kt) \ KtT_Yt;
        end

        % Evaluate
        yp = Kv * alpha;
        res = mean(sqrt(mean((Y(idxTest,:) - yp).^2)));
        if res < rmse
            model.sigma = sigma(ls);
            model.gamma = gamma(lg);
            model.alpha = alpha;
            rmse = res;
        end
        
        % model.res(ls,lg) = res;
        % [ls/numel(sigma) lg/numel(gamma)]
    end
end

% Final model
model.x = X(idxTrain(nk),:);
