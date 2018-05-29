function bt = rbf_bt(D)
% D is a sample x feature matrix
% Compute beta for the RBF kernel
% 
% Assume D = [x_1', ..., x_n']', then 
% beta = the average square sample distance
%      = \sum_{i,j=1}^n ||x_i-x_j||^2/n/(n-1)
%      = \sum_i 2*||x_i||/(n-1) - \sum_{i,j} 2*x_i'x_j/n/(n-1)
%      = 2*||D||^2_F/(n-1) - sum(sum(D*D'))*2/n/(n-1) (matlab notation)
%      = 2*||D||^2_F/(n-1) - sum(sum(D)*D')*2/n/(n-1)
%      = 2*||D||^2_F/(n-1) - sum(D)*sum(D)'*2/n/(n-1)


n = size(D,1);
s = sum(D);
bt = 2*norm(D,'fro')^2/(n-1) - (s*s')*2/n/(n-1);
