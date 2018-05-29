%% preparing data

% load a dataset ( sample x feature )
load('satimage');

% construct a Gaussian kernel matrix
fprintf('constructing a %d x %d Gaussian matrix\n', size(D,1),size(D,1));
G = rbf(D);

% for spectral embedding, using A = sqrt(sum(G)); G = A * G * A; 

% the rank 
k = 100; 

% uniform sampling without replacement
idx = randperm(size(G,1)); 

%% to get a more accurate and stable time, please run this cell twice
m = 200;
fprintf('\n*** forming %d-rank approximation, sampling %d columns ***\n',k,m)
fprintf('\nstandard nystrom...\n')
[U,D,t] = nys(G,k,m,idx);
e = norm(G-U*D*U', 'fro'); % error
fprintf('Computational time:   %.3f s\n', t);
fprintf('Approximation error:  %.3f (F-norm)\n', e);

fprintf('\nnystrom + randomized SVD...\n')
rnys(G,k,m,idx);

p = 10; q = 3;
fprintf('\nusing a more accurate randomized SVD... p=%d, q=%d\n',p,q)
rnys(G,k,m,idx,p,q);

m = 400;
fprintf('\n*** sampling more columns m = %d ***\n',m)
fprintf('\nstandard nystrom...\n')
nys(G,k,m,idx);

fprintf('\nnystrom + randomized SVD...\n')
rnys(G,k,m,idx);