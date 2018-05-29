function K = rbf(A,B,bt)

if nargin < 3
    bt = rbf_bt(A);
end

sbt = sqrt(bt);
A = A / sbt;

if issparse(A)
    A = full(A);
end

if nargin < 2
    B = A;
else
    B = B / sbt;
    if issparse(B)
        B = full(B);
    end
end

% if size(A,2) < 50 %-- such as covtype
%     K = rbf_mex(A',B',1/bt);
%     return
% end

SA = (sum(A.^2, 2));
SB = (sum(B.^2, 2));

%--- time mainly spend on 2*Da*Db' and exp()
%--- exp take memory, can divide it using for !TODO
% K = exp(bsxfun(@minus,bsxfun(@minus,2*A*B', SA), SB')/bt);

K = bsxfun(@minus,bsxfun(@minus,2*A*B', SA), SB'); %/bt;
% exp_mex(K);
K = exp(K);

