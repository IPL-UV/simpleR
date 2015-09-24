% Example script for training warped GP. Replace 01 with 03 to use
% NN covariance rather than Gaussian, and increase length of Xw by 1

clear

input = load('train_inputs');
tt = load('train_outputs');
intest = load('train_inputs');
tttest = load('train_outputs');

% rescale targets
tmax = max(tt); tmin = min(tt);
t = (tt-tmin)/(tmax-tmin) - 0.5;
ttest = (tttest-tmin)/(tmax-tmin) - 0.5;

num = 5; % set number of tanh functions in warping
[n,D] = size(input);
Xw = randn(D+2+num*3,1); % randomly initialise parameters for gpw01lik
% Xw = randn(D+3+num*3,1); % randomly initialise parameters for gpw03lik
			 
% Alternate between 'slow' full gradient optimization and 'fast'
% warping parameter only optimization
for j = 1:5
  [Xw,fX,i] = minimize(Xw,'gpw01lik',-100,input,t);
  [Xw,fX,i] = minimize(Xw,'gpw01fastlik',-200,input,t);
end

save results

plotwarp01(Xw,D,t,'r') % visualize warping function to see if it
                       % has done anything sensible!