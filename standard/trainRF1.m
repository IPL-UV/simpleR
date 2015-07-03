function model = trainRF1(x,y)

% function model = trainRF1(x,y)
%
% Train a RFs using MATLAB's TreeBagger.
%
% This will train as many RFs as output variables are (size(y,2)).
% This may take a long time.

%% Finding optimal leaf size
% leaf = [1:5 10 20];
% col = 'rgbcmyk';
% for i = 1:length(leaf)
%     b = TreeBagger(100, x(idxtrain,:), yp(idxtrain,1), 'method', 'r', ...
%         'oobPred', 'on', 'minleaf', leaf(i));
%     plot(oobError(b), col(i)), hold on
% end
% legs = num2cell(leaf);
% for i = 1:numel(legs), legs{i} = num2str(legs{i}); end
% legend(legs)

% leaf: use '1' or '2'

%% Estimating feature importance
% b = TreeBagger(100, x, y, 'method', 'r', 'oobvarimp', 'on', 'minleaf', 1);
% plot(oobError(b))
% bar(b.OOBPermutedVarDeltaError)

%% RF train
model = cell(1,size(y,2));
for i = 1:length(model)
    fprintf('  RF on output var %d\n', i);
    model{i} = TreeBagger(100, x, y(:,i), 'method', 'r', ...
        'oobvarimp', 'on', 'minleaf', 1, 'nvartosample', 'all');
end

%% See feature importance
% for i = 1:length(model)
%     subplot(4,2,i)
%     bar(model{i}.OOBPermutedVarDeltaError)
% end
