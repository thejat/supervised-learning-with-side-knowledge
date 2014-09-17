%Generate multiple random fullsamples (training, test, side knowledge) with
%the same coefficients and run regression with side knowledge while learning.

clc;close all;clear all;

%% Generate simulated data

s = RandStream('mcg16807','Seed',0) ; RandStream.setGlobalStream(s);
nRuns       = 1;
nBeta       = 100; %fixed dimension. Higher means, more data is required.
betaTrue    = randn(nBeta,1);
nTrainArray = [1000];
nTest       = mean(nTrainArray);%can be anything.
nKnowledge  = 100;

%The features need not be random here. Only the residuals need to be random.

% featureSigma = randn(nBeta+1); %+1 for covariance with residuals.
% featureSigma = featureSigma'*featureSigma;
featureSigma = gallery('randcorr',nBeta+1);
sampleTestX = randn(nTest,nBeta+1)*chol(featureSigma); %kept same for a given beta
sampleTestY = sampleTestX*[betaTrue;0];
sampleTestX = sampleTestX(:,1:end-1); %removed the residuals column
sampleKnowledgeX = randn(nKnowledge,nBeta+1)*chol(featureSigma); %kept same for a given beta
sampleKnowledgeY = sampleKnowledgeX*[betaTrue;0];
sampleKnowledgeX = sampleKnowledgeX(:,1:end-1); %removed the residuals column

for i=1:nRuns
    sampleTrainX = randn(max(nTrainArray),nBeta+1)*chol(featureSigma);
    sampleTrainY = sampleTrainX*[betaTrue;0];
    sampleTrainX = sampleTrainX(:,1:end-1);
end


%% Fit model

nfolds   = 5;
nrepeats = 3;
X_trn    = sampleTrainX;%to be celled, indexed by size
Y_trn    = sampleTrainY;
X_test   = sampleTestX;
Y_test   = sampleTestY;

X_trn = [X_trn ones(size(X_trn,1),1)];%Add a vector of ones to the feature mat to capture intecepts if any
X_test = [X_test ones(size(X_test,1),1)];

%%
betaEst = X_trn\Y_trn;

%%
coeffrange = 2^([-3:1:0]);
[Y_hat_val,betaCVX,regularize_coeff,cv_matrix,Y_hat_trn] = cvalidated_model('Ridge',coeffrange,nfolds,nrepeats,X_trn,Y_trn,X_val,model_disable_CV,0.0001);
metrics = metric_of_success(Y_test,Y_hat_val,length(X_trn(1,:)),'Val',str_dependent,'Ridge',plot_enable);


%{

plot of performance (measured using RMSE) as a function of training sample
size.

For st plot performance for without knowledge case with error bars.

%}
