%
% Simple implementation of http://www.cs.berkeley.edu/~yuczhang/files/colt13_fastkrr.pdf
%
% METHODS = {'RLR' 'LASSO' 'ELASTICNET' 'TREE' 'BAGTREE' 'BOOST' 'NN' 'ELM' 'SVR' 'KRR' 'RVM' 'GPR' 'VHGPR'}
%

function model = fastRegressionTraining(method,X,Y)

[N d] = size(X);

if N<2000
    ordre = ['model = train', method, '(X,Y);'];
    eval(ordre)
else
    m = round(N^0.45);
    NpB = round(N/m);
    
    if NpB>2000
        disp('Too much samples even for this fast implementation')
    else
        % Split the dataset in m disjoint subsets
        indices = crossvalind('Kfold',N,m);
        % Train a model for each subset
        for i = 1:m
            fprintf([num2str(i) ' models out of ' num2str(m) ' using ' num2str(NpB) ' samples each ...\n'])
            train = (indices == i);
            Xtrain = X(train,:);
            Ytrain = Y(train,:);
            ordre = ['model{i} = train', method, '(Xtrain,Ytrain);'];
            eval(ordre);
        end
    end
    
end
