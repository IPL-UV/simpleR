function [yt,sd] = testRF1(model,xt)

% function y = testRF(model,xt)
%
% Predict using previously trained RF in model

%% RF predict
yt = zeros(size(xt,1),length(model));
sd = zeros(size(xt,1),length(model));

if nargout > 1 && strcmp(class(model{1}), 'TreeBagger')
    dostd = true;
else
    dostd = false;
end

for i = 1:length(model)
    if dostd,
        [yt(:,i),sd(:,i)] = predict(model{i}, xt);
    else
        yt(:,i) = predict(model{i}, xt);
    end
end
