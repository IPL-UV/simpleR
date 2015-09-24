function [fX, dfX] = gp03lik(X, input, target);

% gp03lik: Compute minus log likelihood and its derivatives with respect to
% hyperparameters for a Gaussian Process for regression.
%
% gp03lik is an implementation of Gaussian Process for Regression, using a
% non-stationary covariance function derived from feed forward neural networks
% with erf hidden units. See C. K. I. Williams, "Computation with infinite 
% Neural Networks", Neural Computation, vol 10, number 5, pp 1203-1216, 1998
% or http://www.ncrg.aston.ac.uk/Papers/postscript/NCRG_97_025.ps.Z
% Note, that since the covariance function is non-stationary translation of the
% inputs will affect the solution.
%
% usage: [fX dfX] = gp03lik(X, input, target)
%
% where:
%
% X      is a (column) vector (of size D+3) of hyperparameters
% input  is a n by D matrix of training inputs
% target is a (column) vector (of size n) of targets
% fX     is the returned value of minus log likelihood
% dfX    is a (column) vector (of size D+3) of partial derivatives
%        of minus the log likelihood wrt each of the hyperparameters
%
% The form of the covariance function is
%
% C(x^p,x^q) = v1 * asin( x^p' S x^q / sqrt((1 + x^p' S x^p)*(1 + x^q' S x^q)))
%            + v0 * delta_{p,q}
%
% where in the above formula, the inputs x have been augmented by a unit entry
% (playing the role of bias in the neural network analogy; this is purely a
% notational trick, you do not have to add these 1's to the input when calling
% the function!). The S (covariance) matrix is diagonal with (positive) entries
% S_1,...,S_D+1. The second term with the kronecker delta is the noise
% contribution. In this function, the hyperparameters S_i, v1 and v0 are
% collected in the vector X as follows:
%
% X = [ log(S_1)
%       log(S_2) 
%        .
%       log(S_D+1)
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
% [X, fX, i] = minimize(X, 'gp03lik', length, input, target)
%
% See also: minimize, gp03pred
%      
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen (2001-05-29).

[n, D] = size(input);         % number of examples and dimension of input space
expX = exp(X);              % exponentiate the hyperparameters once and for all


% first, we write out the covariance matrix Q

Q = repmat(expX(1:D)',n,1).*input*input';
Z = (expX(D+1)+Q)./(sqrt(1+expX(D+1)+diag(Q))*sqrt(1+expX(D+1)+diag(Q)'));
Q = expX(D+2)*asin(Z) + expX(D+3)*eye(n);


% then, we compute the negative log likelihood ...

invQ = inv(Q);
invQt = invQ*target;
logdetQ = 2*sum(log(diag(chol(Q))));            % don't compute det(Q) directly
fX = 0.5*logdetQ + 0.5*target'*invQt + 0.5*n*log(2*pi);


% ... and its partial derivatives

dfX = zeros(D+3,1);                     % set the size of the derivative vector

Q = repmat(expX(1:D)',n,1).*input*input';
for d = 1:D
  v = input(:,d).^2./(1+expX(D+1)+diag(Q));
  V = (input(:,d)*input(:,d)'./(sqrt(1+expX(D+1)+diag(Q))*sqrt(1+expX(D+1)+diag(Q)'))-Z.*(repmat(v,1,n)+repmat(v',n,1))/2)./sqrt(1-Z.^2); 
  dfX(d) = expX(D+2)*expX(d)*(sum(sum(invQ.*V)) - invQt'*V*invQt)/2;
end 

v = 1./(1+expX(D+1)+diag(Q));
V = (1./(sqrt(1+expX(D+1)+diag(Q))*sqrt(1+expX(D+1)+diag(Q)'))-Z.*(repmat(v,1,n)+repmat(v',n,1))/2)./sqrt(1-Z.^2);
dfX(D+1) = expX(D+2)*expX(D+1)*(sum(sum(invQ.*V)) - invQt'*V*invQt)/2;

V = asin(Z);
dfX(D+2) = expX(D+2)*(sum(sum(invQ.*V)) - invQt'*V*invQt)/2;
dfX(D+3) = expX(D+3)*(trace(invQ) - invQt'*invQt)/2;


