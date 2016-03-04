%% RANKING FEATURES with DIFFERENT METHODS

d = size(Xtest,2);

% LR
figure, bar(model_RLR.W(2:end)), set(gca,'Xtick',1:d,'XtickLabel',VARIABLES)
grid,title('LR Feature relevance.')

% LASSO

% Display the weights
figure, bar(model_LASSO.W), set(gca,'Xtick',1:d,'XtickLabel',VARIABLES)
grid,title('LASSO Feature relevance.')

% Display a trace plot of the lasso fits.
axTrace = lassoPlot(model_LASSO.B,model_LASSO.S);

% Display the sequence of cross-validated predictive MSEs.
axCV = lassoPlot(model_LASSO.B,model_LASSO.S,'PlotType','CV');

% ELASTICNET
figure, bar(model_ElasticNet.W), set(gca,'Xtick',1:d,'XtickLabel',VARIABLES)
grid,title('ELASTIC NET Feature relevance.')

% Display a trace plot of the lasso fits.
axTrace = lassoPlot(model_ElasticNet.B,model_ElasticNet.S);

% Display the sequence of cross-validated predictive MSEs.
axCV = lassoPlot(model_ElasticNet.B,model_ElasticNet.S,'PlotType','CV');

% TREE
view(model_TREE)
figure, bar(varimportance(model_TREE)), set(gca,'Xtick',1:d,'XtickLabel',VARIABLES)
grid,title('TREE Feature relevance.')

% GPR
figure, bar(1./exp(model_GP.loghyper(1:d))), set(gca,'Xtick',1:d,'XtickLabel',VARIABLES)
grid,title('GPR Feature relevance.')

% VHGPR
figure, bar(1./exp(model_VHGP.loghyper(1:d))), set(gca,'Xtick',1:d,'XtickLabel',VARIABLES)
grid,title('VHGPR Feature relevance.')
