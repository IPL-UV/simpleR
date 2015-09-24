% Example script for making predictions from warped GP. Replace 01
% with 03 for NN covariance instead of Gaussian

load results

alpha = [0.1:0.1:1-0.1]; % set quantiles for predictive density

% NOTE: supply scaled targets t and ttest here
[quant,mea,dens]= gpw01pred(Xw,input,t,intest,alpha,ttest);

% rescale back
quant = (quant+0.5)*(tmax-tmin) + tmin;
mea = (mea+0.5)*(tmax-tmin) + tmin;
dens = log(tmax-tmin) + dens;