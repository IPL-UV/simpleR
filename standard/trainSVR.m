function model = trainSVR(X,Y)

vf  = 3;
eps = [0.001 0.01 0.1:0.1:0.5];
C   = logspace(0,3,10);

% First guess for the sigma parameter
meanSigma = mean(pdist(X));
sigmaMin = log10(meanSigma * 0.1);
sigmaMax = log10(meanSigma * 10);
sigma = logspace(sigmaMin, sigmaMax, 20);
gamma = 1./sqrt(2*sigma);

bestEps = 1;
bestC = 1;
bestGamma = 1;
bestMse = Inf;

for gg = 1:numel(gamma)
    % K = kernelmatrix('rbf', X', X', sigma(gg));
    for cc = 1:numel(C)
        for ee = 1:numel(eps)
            % params = sprintf('-s 3 -t 4 -c %f -p %f -v %d', C(cc), eps(ee), vf);
            params = sprintf('-s 3 -t 2 -g %f -c %f -p %f -v %d', gamma(gg), C(cc), eps(ee), vf);
            mse = svmtrain(Y, X ,params);
            if mse < bestMse
                bestMse = mse;
                bestEps = ee;
                bestC = cc;
                bestGamma = gg;
            end
        end
    end
end

% K = kernelmatrix('rbf', X', X', sigma(bestGamma));
% params = sprintf('-s 3 -t 4 -c %f -p %f', C(bestC), eps(bestEps));
params = sprintf('-s 3 -t 2 -g %f -c %f -p %f', gamma(bestGamma), C(bestC), eps(bestEps));
model = svmtrain(Y, X, params);
% model.sigma = sigma(bestGamma);
% model.C = C(bestC);
% model.Xtrain = X;
