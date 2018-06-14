function [Yp,quantiles,densities] = testWGPR(model,Xtest)

% set quantiles for predictive density
alpha = 0.1:0.1:0.9;

% rescale targets
tmax   = model.tmax; tmin = model.tmin;
Y      = (model.Ytrain-tmin)/(tmax-tmin) - 0.5;

% NOTE: supply scaled targets t and ttest here
% [quant,mea,dens]= gpw01pred(model.Xw,Xtest,Y,intest,alpha,Ytest);
[Yp,quantiles,densities] = gpw01pred(model.Xw,model.Xtrain,Y,Xtest,alpha);

% rescale back
Yp = (Yp + 0.5)*(tmax-tmin) + tmin;
% Yp = real(Yp);
