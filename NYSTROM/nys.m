function [U,D,t,e] = nys(G,k,m,idx)
% NYS   rank-k nystrom approximation to PSD matrix G
%   [U,D] = nys(G,k) : rank-k nystrom approximation of the PSD matrix G such
%   that G ~= U*D*U'. 
%   
%   [U,D,t,e] = nys(G,k) : returns the computation time t (not including
%   computing the error) and the approximation error e=||G-U*D*U'||_F in Frobenius norm.
%
%   [U,D,t,e] = nys(G,k,m) samples m columns of G; default = k
%
%   [U,D,t,e] = nys(G,k,m,idx) samples the m columns specified by the index
%   idx(1:m); default is uniform sampling without replacement.
%
%   See also RSVD RNYS
%
%   Reference: Williams, C.K.I. and Seeger, M. "Using the Nystrom method to
%   speed up kernel machines", NIPS, 2001

%   Author: Mu Li (limu.cn@gmail.com)
%   Date:   04/18/2010

%%%%%%%% check argin %%%%%%%%
narginchk(2, 4)

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
elseif length(idx) < m
    error('the column indexes are not enough: length(idx) = %d < m = %d', ...
          length(idx), m);
end

%%%%%%%% argout %%%%%%%%
nargoutchk(0, 4);

%%%%%%%% main algorithm %%%%%%%%

% eps = 1e-12;

tstart = tic;

%%%% sampling %%%%

if isempty(idx)
    idx = randperm(n);                  % uniform sampling without replacement 
end

cols = idx(1:m);                        % m columns indexes
C = G(:,cols);                          % m columns
W = C(cols,:);                          % m-by-m intersection matrix

%%%% perform truncated SVD on m-by-m matrix W %%%%

% W = (W + W')/2;                         
% [V,D] = eigs(W,k);
% [D,ix] = sort(diag(D),'descend');       
% r = nnz(D>eps);                         % numerical rank

% if r < k
%    warning('The numerical rank of W (=%d) is less than k (=%d)', r, k);
%    k = r;
% end
% D = D(1:k);
% V = V(:,ix(1:k));

[V,D] = svd(W,'econ');
D = diag(D(1:k,1:k));
V = V(:,1:k);

%%%% form the approximation

U = C * ( sqrt(m/n) * V * diag(D.^-1) );
D = (n/m) * diag(D);

t = toc(tstart);

%%%% compute the approximation error
if nargout == 0 || nargout == 4
    e = norm ( G - U * D * U', 'fro');
end

% if nargout == 1, U = D; end % in case you forget the ';'

%%%% display the result
if nargout == 0
    fprintf('Computational time:   %.3f s\n', t);
    fprintf('Approximation error:  %.3f (F-norm)\n', e);
end
