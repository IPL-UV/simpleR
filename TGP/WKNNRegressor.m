function result = WKNNRegressor(TestInput, Input, Target, Knn)

% Regression using the Nearest neighbor algorithm
% Inputs:
% 	Input	   - Train samples
%	Target	   - Train labels
%   TestInput  - Test  samples
%	Knn		   - Number of nearest neighbors 
%
% Outputs
%	result	- Predicted targets

L			= length(Target);

if (L < Knn),
   error('You specified more neighbors than there are points.')
end

N                   = size(TestInput,1);
d                   = size(Target,2);
result              = zeros(N,d); 
for i = 1:N,
    dist            = sum((Input - ones(L,1)*TestInput(i,:)).^2,2);
%     dist = sum(abs(Input - ones(L,1)*TestInput(i,:)),2);
    [val, indices]  = sort(dist);
    index = indices(1:Knn);
    W = 1./dist(index);
    W = W./sum(W);
    result(i,:)     = sum((W*ones(1,d)).*Target(index,:));
end