function prank = permutation(METHOD,model,Xtr,Ytr)

nperms  = 10;
nX      = length(Xtr(1,:));
prank   = zeros(nperms,nX);

for i=1:nX
    cXtr  = Xtr;
    cX    = Xtr(:,i);
    nS    = size(cX,1);

    for j=1:nperms
        cX=cX(randperm(nS));
        cXtr(:,i)=cX;
%         cXtr = zscore(cXtr);
%         Ypred = testRF1(model,cXtr);
        eval(['Yp = test' METHOD '(model,cXtr);']);       % Test the model
        prank(j,i) = mean((Yp-Ytr).^2);
    end
end 

