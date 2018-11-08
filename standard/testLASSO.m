function Yp = testLASSO(model_LASSO, Xtest)

B = model_LASSO.B;
S = model_LASSO.S;

Xplus = [ones(size(Xtest,1),1) Xtest];
Yp = Xplus * [S.Intercept(S.Index1SE) ; B(:,S.Index1SE)];
