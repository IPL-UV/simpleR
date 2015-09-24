function [fX, dfX] = gp01lik(X, input, target);

% gp01lik: Compute minus log likelihood and its derivatives with respect to
% hyperparameters for a Gaussian Process for regression.
%
% gp01lik is an implementation of Gaussian Process for Regression, using a
% simlpe Gaussian covariance function and a noise term. The covariance
% function is controlled by hyperparameters.
%
% usage: [fX dfX] = gp01lik(X, input, target)
%
% where:
%
% X      is a (column) vector (of size D+2) of hyperparameters
% input  is a n by D matrix of training inputs
% target is a (column) vector (of size n) of targets
% fX     is the returned value of minus log likelihood
% dfX    is a (column) vector (of size D+2) of partial derivatives
%        of minus the log likelihood wrt each of the hyperparameters
%
% The form of the covariance function is
%
% C(x^p,x^q) = v1 * exp( -0.5 * sum_{d=1..D} w_d * (x^p_d - x^q_d)^2 )
%            + v0 * delta_{p,q}
%
% where the first term is Gaussian and the second term with the kronecker
% delta is the noise contribution. In this function, the hyperparameters w_i,
% v1 and v0 are collected in the vector X as follows:
%
% X = [ log(w_1)
%       log(w_2) 
%        .
%       log(w_D)
%       log(v1)
%       log(v0) ]
%
% Note: the reason why the log of the parameters are used in X is that we may
% then used unconstrained optimisation, while the hyperparameters themselves
% must, naturally, always be positive.
%
% This function can conveniently be used with the "minimize" function to train
% a Gaussian process:
%
% [X, fX, i] = minimize(X, 'gp01lik', length, input, target)
%
% See also: minimize, gp01pred
%      
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen (2001-05-25).


[n, D] = size(input);         % number of examples and dimension of input space
expX = exp(X);              % exponentiate the hyperparameters once and for all


% first, we write out the covariance matrix Q

Z = zeros(n,n);
for d = 1:D
  Z = Z + (repmat(input(:,d),1,n)-repmat(input(:,d)',n,1)).^2*expX(d);
end
Z = expX(D+1)*exp(-0.5*Z);
Q = Z + expX(D+2)*eye(n);                             % Gaussian term and noise


% then, we compute the negative log likelihood ...

invQ = inv(Q);
invQt = invQ*target;
logdetQ = 2*sum(log(diag(chol(Q))));    % don't compute det(Q) directly
fX = 0.5*logdetQ + 0.5*target'*invQt + 0.5*n*log(2*pi);


% ... and its partial derivatives

dfX = zeros(D+2,1);                     % set the size of the derivative vector

for d = 1:D
  V = (repmat(input(:,d),1,n)-repmat(input(:,d)',n,1)).^2.*Z;
  dfX(d) = expX(d)*(invQt'*V*invQt - sum(sum(invQ.*V)))/4;
end 
dfX(D+1) = 0.5*sum(sum(invQ.*Z)) - 0.5*invQt'*Z*invQt;
dfX(D+2) = 0.5*trace(invQ)*expX(D+2) - 0.5*invQt'*invQt*expX(D+2);
