function [mu, S2, deriv, S2deriv] = gp03pred(X, input, target, test);

% gp03pred: Compute (marginal) predictions based on hyperparameters, training
% inputs and targets and test inputs - from 1 to 4 outputs may be requested.
%
% usage: [mu S2 deriv S2deriv] = gp03pred(X, input, target, test) 
%
% where: 
%
%   X       is a (column) vector (of size D+3) of hyperparameters
%   input   is a n by D matrix of training inputs
%   target  is a (column) vector (of size n) of targets
%   test    is a nn by D matrix of test inputs
%   mu      is a (column) vector (of size nn) of prediced means
%   S2      is a (column) vector (of size nn) of predicted variances
%   deriv   is a n by D matrix of mean partial derivatives
%   S2deriv is a n by D matrix of variances on the partial derivatives
%
% Note that the reported variances in S2 are for the noise-free signal; to get
% the noisy variance, simply add the noise variance log(X(D+3)).
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
% which corresponds exactly to the covariance function used by the other gp03 
% programs.
%
% See also: gp03lik
%      
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen (2001-07-26).

[n, D] = size(input);   % number of training cases and dimension of input space
[nn, D] = size(test);       % number of test cases and dimension of input space
expX = exp(X);              % exponentiate the hyperparameters once and for all


% first, we write out the covariance matrix Q for the training inputs ...

Q = repmat(expX(1:D)',n,1).*input*input';
Z = (expX(D+1)+Q)./(sqrt(1+expX(D+1)+diag(Q))*sqrt(1+expX(D+1)+diag(Q)'));
Q = expX(D+2)*asin(Z) + expX(D+3)*eye(n);


% ... then we compute the covariance between training and test inputs ...

v = expX(D+1)+sum(repmat(expX(1:D)',nn,1).*test.^2,2);

Z = sqrt(1+expX(D+1)+sum(repmat(expX(1:D)',n,1).*input.^2,2)) * sqrt(1+v');
ZZ = (expX(D+1)+repmat(expX(1:D)',n,1).*input*test')./Z;
a = expX(D+2)*asin(ZZ);


% ... and covariance between the test input and themselves 

b = expX(D+2)*asin(v./(1+v));


% Now, write out the desired terms

if nargout == 1
  mu = a'*(Q\target);     % don't compute invQ explicitly if we only need means
else
  invQ = inv(Q);
  invQt = invQ*target;
  mu = a'*invQt;
  S2 = b - sum(a.*(invQ*a),1)';
end


if nargout > 2
  for d = 1:D
    c = (repmat(input(:,d),1,nn)./Z-ZZ.*repmat((test(:,d)./(1+v))',n,1)) ...
                                                              ./ sqrt(1-ZZ.^2);
    deriv(1:nn,d) = expX(D+2)*expX(d)*c'*invQt;
    if nargout == 4
       f = (1-(2*expX(d)*test(:,d).^2)./(1+2*v))./sqrt(1+2*v);
       S2deriv(1:nn,d) = expX(D+2)*expX(d) * ...
                                     (f-expX(D+2)*expX(d)*sum(c.*(invQ*c),1)');
    end
  end
end
