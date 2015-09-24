function [quant,mea,dens] = gpw03pred(X,input,t,testin,alpha,testout)

% gpw03pred: Make predictions for the warped Gaussian process with
% neural net covariance. Requires Carl Edward Rasmussen's gp03pred
% code for making predictions from a standard GP with neural net
% covariance.
% 
% X:       vector of warping parameters and covariance hyperparameters
%          (see gpw03lik)
% input:   (n x D) matrix of training inputs
% t:       (n x 1) vector of training targets
% testin:  (N x D) matrix of test inputs
% alpha:   (1 x Na) vector of quantiles (0 < alpha < 1)
% testout: OPTIONAL (N x 1) vector of test targets for computation
%          of negative log predictive density
% 
% quant:   (N x Na) matrix of locations of quantiles specified by alpha
% mea:     (N x 1) means of predictive densities
% dens:    (N x 1) negative log predictive densituies evaluated at
%          test targets

N = size(testin,1);
[n,D] = size(input);
num = (length(X) - D - 3)/3; 
for i = 1:num
    ea(i,1) = exp(X(D+3+i)); eb(i,1) = exp(X(D+3+num+i));
    c(i,1) = X(D+3+2*num+i);
end

z = warp(t,ea,eb,c); % warp training targets to latent space
[sortz,I] = sort(z); sortt = t(I);

% Make predictions from standard GP
[muz,s2z] = gp03pred(X(1:D+3),input,z,testin);
s2z = s2z + exp(X(D+3));


% find locations in latent 'z' space of quantiles specified by alpha
q = repmat(sqrt(2*s2z),1,length(alpha)).* ... 
	   repmat(erfinv(2*alpha-1),N,1) ...
	    + repmat(muz,1,length(alpha));
% pass quantiles through inverse warp function to find locations in
% observation 't' space
quant = invert(q, ea, eb, c, sortz, sortt);


% quadrature to compute mean of predictive density

%quadr = [0.3811870 1.1571937 1.9816568 2.9306374];
quadr = [0.3429013 1.0366108 1.7566836 2.5327317 3.4361591];
quadr = [-quadr(end:-1:1) quadr];
%H = [0.6611470 0.2078023 0.0170780 0.0001996];
H = [0.6108626 0.2401386 0.0338744 0.0013436 .00000076];
H = [H(end:-1:1) H];

mea = invert(sqrt(2*s2z)*quadr + ... 
	     repmat(muz,1,length(quadr)),ea,eb,c,sortz,sortt);
mea = mea*H'/sqrt(pi);


% if test targets are supplied, compute neg. log predictive density

if nargin > 5
  w = ones(length(testout),1);
  for i = 1:num
    s{i} = eb(i)*(testout + c(i));
    r{i} = tanh(s{i});
    y{i} = 1 - r{i}.^2;
    w = w + ea(i)*eb(i).*y{i};
  end
  dens = 0.5*log(2*pi*s2z) + 0.5*(warp(testout,ea,eb,c)-muz).^2./s2z - log(w);
else
  dens = [];
end


function newt = invert(newz, ea, eb, c, sortz, sortt)
% invert warp function for vector newz. sortz and sortt provide
% very good initial starting points for Newton iterations 

for j = 1:size(newz,1)
  for k = 1:size(newz,2)
    if newz(j,k) > sortz(end)
      t0(j,k) = sortt(end);
    elseif newz(j,k) < sortz(1)
      t0(j,k) = sortt(1);
    else
      I = find(sortz > newz(j,k)); I = [I(1)-1;I(1)];
      t0(j,k) = mean(sortt(I));
    end
  end
end

newt = warpinv(newz,ea,eb,c,t0,8); % may need to adjust no. of
                                   % iterations to ensure convergence

