function Yp    = fastTest(method,model,Xtest)

[Ntest d] = size(Xtest);
m = numel(model);

% h = waitbar(0,'Please wait while testing...');

Yp = zeros(Ntest,1);
for i = 1:m
    ordre = ['yp = test' method '(model{i},Xtest);'];
    eval(ordre)
    Yp = Yp + yp;
%     waitbar(i/m,h)
end

Yp = Yp/m;
