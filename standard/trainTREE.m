function model_TREE = trainTREE(Xtrain,Ytrain)

strver = version;
if str2double(strver(1)) <= 7
    model_TREE = treefit(Xtrain, Ytrain, 'method', 'regression');
    % Uncomment this if you want to perform xval pruning ...
    [~,~,~,best] = treetest(model_TREE, 'cross', Xtrain, Ytrain);
    model_TREE   = treeprune(model_TREE, 'level', best);
else
    model_TREE = fitrtree(Xtrain,Ytrain);
end
