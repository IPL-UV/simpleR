function [mu, S2, deriv, S2deriv, dummy] = gp01pred(X, input, target, test);

% gp01pred: Compute (marginal) predictions based on hyperparameters, training
% inputs and targets and test inputs - from 1 to 4 outputs may be requested;
% if a fifth return value is requested, this prompts the program to return
% the S2deriv as a 3 dimensional array (full covariance per test point).
%
% usage: [mu S2 deriv S2deriv dummy] = gp01pred(X, input, target, test) 
%
% where: 
%
%   X       is a (column) vector (of size D+2) of hyperparameters
%   input   is a n by D matrix of training inputs
%   target  is a (column) vector (of size n) of targets
%   test    is a nn by D matrix of test inputs
%   mu      is a (column) vector (of size nn) of prediced means
%   S2      is a (column) vector (of size nn) of predicted variances
%   deriv   is a n by D matrix of mean partial derivatives
%   S2deriv is a n by D (or n by D by D) matrix of (co-)variances on the
%           partial derivatives
%   dummy   a dummy variable whose presence make S2deriv 3 dimensional
%
% Note that the reported variances in S2 are for the noise-free signal; to get
% the noisy variance, simply add the noise variance log(X(D+2)).
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
% which corresponds exactly to the covariance function used by the other gp01 
% programs.
%
% See also: gp01lik
%      
% (C) Copyright 1999, 2000, 2001 & 2002, Carl Edward Rasmussen (2002-01-22).

[n, D] = size(input);   % number of training cases and dimension of input space
[nn, D] = size(test);       % number of test cases and dimension of input space
expX = exp(X);              % exponentiate the hyperparameters once and for all


% first, we write out the covariance matrix Q for the training inputs ...

Q = zeros(n,n);                                         % create and zero space
for d = 1:D                                           % non-linear contribution
  Q = Q + expX(d)*(repmat(input(:,d),1,n)-repmat(input(:,d)',n,1)).^2;
end
Q = expX(D+1)*exp(-0.5*Q) + expX(D+2)*eye(n);


% ... then we compute the covariance between training and test inputs ...

a = zeros(n, nn);                                       % create and zero space
for d = 1:D
  a = a + expX(d)*(repmat(input(:,d),1,nn)-repmat(test(:,d)',n,1)).^2;
end
a = expX(D+1)*exp(-0.5*a);


% ... and covariance between the test input and themselves 

b = expX(D+1);


% Now, write out the desired terms

if nargout == 1
  mu = a'*(Q\target);     % don't compute invQ explicitly if we only need means
else
  invQ = inv(Q);
  invQt = invQ*target;
  mu = a'*invQt;                                              % predicted means
  S2 = b - sum(a.*(invQ*a),1)';                            % predicted variance
end

if nargout > 2
  deriv = zeros(nn,D);                                           % create space
  if nargout == 4
    S2deriv = zeros(nn,D);
  elseif nargout == 5
    S2deriv = zeros(nn,D,D); dummy = [];        % assign dummy to avoid warning
  end
  for d = 1:D
    c = a.*(repmat(input(:,d),1,nn)-repmat(test(:,d)',n,1));
    deriv(1:nn,d) = expX(d)*c'*invQt;                      % derivative of mean
    if nargout == 4
      S2deriv(1:nn,d) = expX(d)*(expX(D+1)-expX(d)*sum(c.*(invQ*c),1)');    
    elseif nargout == 5
      ainvQc = a.*(invQ*c);
      for e = 1:D
        S2deriv(1:nn,d,e) = -expX(d)*expX(e)* ...
              sum(ainvQc.*(repmat(input(:,e),1,nn)-repmat(test(:,e)',n,1)),1)';
      end
      S2deriv(1:nn,d,d) = S2deriv(1:nn,d,d) + expX(d)*expX(D+1);
    end
  end
end
