function model = trainRF1(x, y, ntrees, nvartosample, oobvarimp)

% function model = trainRF1(x, y, ntrees, nvartosample, oobvarimp)
%
% Train a RFs using MATLAB's TreeBagger.
%
% This will train as many RFs as output variables are (size(y,2)).
% This may take a long time.

if ~exist('ntrees', 'var')
    ntrees = 200;
end
if ~exist('nvartosample', 'var')
    % nvartosample = ceil(size(x,2)/3); % default RF Breiman's value for regr.
    nvartosample = size(x,2); % Empirically using all variables work best
end
if ~exist('oobvarimp', 'var')
    oobvarimp = 'off';
end

%% Finding optimal leaf size
% leaf = [1:5 10 20];
% col = 'rgbcmyk';
% for i = 1:length(leaf)
%     b = TreeBagger(ntrees, x(idxtrain,:), yp(idxtrain,1), 'method', 'r', ...
%         'oobPred', 'on', 'minleaf', leaf(i));
%     plot(oobError(b), col(i)), hold on
% end
% legs = num2cell(leaf);
% for i = 1:numel(legs), legs{i} = num2str(legs{i}); end
% legend(legs)

% minleaf: use between 1 and 5
minleaf = 1;

% nvartosample = 'all'; % if not set, 1/3 of the variables for regression
% nvartosample = ceil(size(x,2) / 3); % default original RF setting
% nvartosample = size(x,2) - 1; % any positive integer invokes Breiman's 'random forest' algorithm
% As in canonical correlation forests
% nvartosample = ceil(log2(size(x,2)) + 1);

%% Estimating feature importance
% b = TreeBagger(100, x, y, 'method', 'r', 'oobvarimp', 'on', 'minleaf', 1);
% plot(oobError(b))
% bar(b.OOBPermutedVarDeltaError)

%% RF train
model = cell(1,size(y,2));
for i = 1:length(model)
    fprintf('  RF on output var %d\n', i);
    model{i} = TreeBagger(ntrees, x, y(:,i), 'method', 'r', ...
        'oobvarimp', oobvarimp, 'minleaf', minleaf, 'nvartosample', nvartosample);
    % model{i} = model{i}.compact();
end

%% See feature importance
% for i = 1:length(model)
%     subplot(4,2,i)
%     bar(model{i}.OOBPermutedVarDeltaError)
% end
