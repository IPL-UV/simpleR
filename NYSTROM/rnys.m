function [U,D,t,e] = rnys(G,k,m,idx,p,q)
%   [U,D] = rnys(G,k) : rank-k nystrom approximation of the PSD matrix G such
%   that G ~= U*D*U'. 
%   
%   [U,D,t,e] = rnys(G,k) : returns the computation time t (not includes
%   computing the error) and the approximation error e=||G-U*D*U'||_F in
%   Frobenius norm.
%
%   the other input options:
%
%   m   : samples m columns of G; default = k
%
%   idx : samples the m columns specified by idx(1:m); default is by uniform
%   sampling without replacement
% 
%   p   : over-sampling parameter of the randomized SVD; defaults is 5
%
%   q   : power parameter of the randomized SVD; defaults is 2
%
%   Reference: M. Li, J.T. Kwok, B. Lu. Making large-scale Nystrom approximation possible. ICML, 2010. 

%   Author: Mu Li (limu.cn@gmail.com)
%   Date:   04/18/2010

%%%%%%%% check argin %%%%%%%%
narginchk(2, 6)

n = size(G,1);
if n ~= size(G,2)
    error('G, which size is (%d,%d), should be square and symmetric.', n, size(G,2));
end

if nargin < 3
    m = k;
end

if m < k || m > n
    error('should sample m (now is %d) columns not less than the rank k(=%d)', m, k);
end

if nargin < 4 
    idx = [];
elseif length(idx) < m && ~isempty(idx)
    error('the column indexes are not enough: length(idx) = %d < m = %d', ...
          length(idx), m);
end

if nargin < 5
    p = 5;
end

if nargin < 6
    q = 2;
end

%%%%%%%% argout %%%%%%%%
nargoutchk(0, 4);

%%%%%%%% main algorithm %%%%%%%%

tstart = tic;

%%%% sampling

if isempty(idx)
    idx = randperm(n);                  % uniform sampling without
                                        % replacement 
end

cols = idx(1:m);                        % m columns indexes
C = G(:,cols);                          % m columns
W = C(cols,:);                          % m-by-m intersection matrix

%%%% perform truncated SVD on m-by-m matrix W

[V,D] = rsvd(W,k,p,q);

%%%% form the approximation

U = C * ( sqrt(m/n) * V );
D = (n/m) * diag(diag(D).^-1);

t = toc(tstart);

%%%% compute the approximation error
if nargout == 0 || nargout == 4
    e = norm ( G - U * D * U', 'fro');
end

%%%% display the result
if nargout == 0
    fprintf('Computational time:   %.3f s\n', t);
    fprintf('Approximation error:  %.3f (F-norm)\n', e);
end
