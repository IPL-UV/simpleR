function [XPLOTS PPLOTS] = partialplots(METHOD,model,X)
% Partial dependence plot of X
% tilde{f}(x) = frac{1}{n} sum_{i=1}^n f(x, x_{iC}),

[n d]  = size(X);
P      = 20;         % number of evaluation points
PPLOTS = zeros(d,P);
XPLOTS = zeros(d,P);

for var = 1:d        % we evaluate all input variables
    m  = min(X(:,var));
    M  = max(X(:,var));
    xc = linspace(m,M,P);
    XPLOTS(var,:) = xc;
    for i = 1:P
        DELTA = ones(n,d);
        DELTA(:,var) = xc(i)*ones(n,1);
        xx = X.*DELTA;
        eval(['Yp = test' METHOD '(model,xx);']);       % Test the model
        PPLOTS(var,i) = mean(Yp);
    end
end
