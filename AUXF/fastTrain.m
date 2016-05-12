function model = fastTrain(method,X,Y)

% METHODS = {'RLR' 'LASSO' 'ELASTICNET' 'TREE' 'BAGTREE' 'BOOST' 'NN' 'ELM' 'SVR' 'KRR' 'RVM' 'GPR' 'VHGPR'}

[N d] = size(X);

if N<2000
    ordre = ['model = train' method '(X,Y);'];
    eval(ordre)
else
    m = round(N^0.45);
    NpB = round(N/m);
    
    if NpB>2000
        disp('Too many data points even for this fast implementation ...')
    else
        
        % Split in m disjoint subsets
        indices = crossvalind('Kfold',N,m);
        % Train a KRR for each subset
%         h = waitbar(0,'Please wait while training...');
        for i = 1:m
            train = (indices == i);
            Xtrain = X(train,:);
            Ytrain = Y(train,:);
            ordre = ['model{i} = train' method '(Xtrain,Ytrain);'];
            eval(ordre)
%             waitbar(i/m,h)
        end
        
    end
end







